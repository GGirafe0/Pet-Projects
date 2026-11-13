import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import time
import textwrap
import tkinter as tk
import threading
import queue
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# --- Настройки ---
MODEL_SIZE = "small"
SAMPLE_RATE = 16000
DEVICE = "cpu"
FONT_SIZE = 16
TRANSPARENT_COLOR = "#010203"

# VAD
VAD_FRAME_MS = 30
VAD_MODE = 2
SILENCE_END_MS = 800
MIN_SPEECH_MS = 300
MAX_PHRASE_SEC = 15

# Текст
MAX_LINES = 4
WRAP_CHARS = 60


class SpeechTranslatorApp:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.ready = False
        self.paused = False          # флаг паузы (читается из аудио-потока)

        self._drag_x = 0
        self._drag_y = 0

        self.lines = []

        self._build_ui()

        self.ready = True
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

        self.root.after(30, self._check_queues)
        self.root.mainloop()

    # ---------- Интерфейс ----------
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Speech Translator")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.hidden_root = tk.Toplevel(self.root)
        self.hidden_root.title("Speech Translator")
        self.hidden_root.geometry("1x1+-10000+-10000")
        self.hidden_root.iconify()

        try:
            self.root.attributes("-transparentcolor", TRANSPARENT_COLOR)
        except tk.TclError:
            self.root.attributes("-alpha", 1.0)

        self.root.configure(bg=TRANSPARENT_COLOR)

        self.win_w, self.win_h = 700, 160
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = 20
        y = screen_h - self.win_h - 60
        self.root.geometry(f"{self.win_w}x{self.win_h}+{x}+{y}")

        self.canvas = tk.Canvas(self.root, bg=TRANSPARENT_COLOR,
                                highlightthickness=0, bd=0)
        self.canvas.pack(expand=True, fill="both")

        # Значок-ручка
        self._create_outlined_text(12, 6, "⠿", 18, "drag_handle")
        self.canvas.tag_bind("drag_handle", "<Button-1>", self._start_drag)
        self.canvas.tag_bind("drag_handle", "<B1-Motion>", self._do_drag)

        # Кнопка паузы
        self._create_outlined_text(42, 6, "⏸", 14, "pause_btn")
        self.canvas.tag_bind("pause_btn", "<Button-1>", lambda e: self._toggle_pause())
        self.canvas.tag_bind("pause_btn", "<Enter>",
                             lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind("pause_btn", "<Leave>",
                             lambda e: self.canvas.config(cursor=""))

        # Кнопка закрытия
        self._create_outlined_text(72, 6, "✕", 14, "close_btn")
        self.canvas.tag_bind("close_btn", "<Button-1>", lambda e: self._quit())
        self.canvas.tag_bind("close_btn", "<Enter>",
                             lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind("close_btn", "<Leave>",
                             lambda e: self.canvas.config(cursor=""))

        self._draw_text_with_outline("Говорите...")
        self.root.update_idletasks()
        print("[GUI] интерфейс готов")

    def _create_outlined_text(self, x, y, text, size, tags):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.canvas.create_text(
                x + dx, y + dy, text=text, anchor="nw",
                fill="black", font=("Arial", size, "bold"), tags=tags,
            )
        self.canvas.create_text(
            x, y, text=text, anchor="nw",
            fill="white", font=("Arial", size, "bold"), tags=tags,
        )

    def _start_drag(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _do_drag(self, event):
        self.root.geometry(
            f"+{event.x_root - self._drag_x}+{event.y_root - self._drag_y}"
        )

    # ---------- Пауза ----------
    def _toggle_pause(self):
        self.paused = not self.paused
        # Перерисовываем значок: ⏸ ↔ ▶
        self.canvas.delete("pause_btn")
        symbol = "▶" if self.paused else "⏸"
        self._create_outlined_text(42, 6, symbol, 14, "pause_btn")
        # tag_bind нужно перепривязать (удалили старые элементы)
        self.canvas.tag_bind("pause_btn", "<Button-1>",
                             lambda e: self._toggle_pause())
        self.canvas.tag_bind("pause_btn", "<Enter>",
                             lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind("pause_btn", "<Leave>",
                             lambda e: self.canvas.config(cursor=""))
        print(f"[PAUSE] {'включена' if self.paused else 'выключена'}")

    def _quit(self):
        try:
            self.hidden_root.destroy()
        except Exception:
            pass
        self.root.destroy()
        os._exit(0)

    # ---------- Текст ----------
    def _draw_text_with_outline(self, text):
        self.canvas.delete("text")
        text_x, text_y = 20, 34
        w = self.win_w - 40
        font = ("Arial", FONT_SIZE, "bold")

        measure_id = self.canvas.create_text(
            text_x, text_y, text=text, anchor="nw",
            font=font, width=w, justify="left", fill="", tags="measure",
        )
        bbox = self.canvas.bbox(measure_id)
        self.canvas.delete("measure")

        if bbox:
            x1, y1, x2, y2 = bbox
            pad = 6
            self.canvas.create_rectangle(
                x1 - pad, y1 - pad, x2 + pad, y2 + pad,
                fill="black", outline="black", stipple="gray50", tags="text",
            )

        self.canvas.create_text(
            text_x, text_y, text=text, anchor="nw",
            fill="white", font=font, width=w, justify="left", tags="text",
        )

    def _append_translation(self, phrase):
        wrapped = textwrap.wrap(phrase, width=WRAP_CHARS) or [phrase]
        self.lines.extend(wrapped)
        if len(self.lines) > MAX_LINES:
            self.lines = self.lines[-MAX_LINES:]
        self._draw_text_with_outline("\n".join(self.lines))

    # ---------- Очереди ----------
    def _check_queues(self):
        try:
            while True:
                phrase = self.text_queue.get_nowait()
                self._append_translation(phrase)
        except queue.Empty:
            pass
        self.root.after(30, self._check_queues)

    # ---------- Аудио ----------
    def _audio_worker(self):
        while not self.ready:
            time.sleep(0.05)

        try:
            dev = sd.query_devices(kind="input")
            print(f"[AUDIO] Микрофон: {dev['name']}")
        except Exception as e:
            print(f"[AUDIO] Ошибка: {e}")

        print("[AUDIO] Загрузка модели...")
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")
        print("[AUDIO] Модель готова. Слушаю...")

        vad = webrtcvad.Vad(VAD_MODE)
        frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        frame_bytes = frame_samples * 2

        audio_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            audio_q.put(pcm16)

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=frame_samples,
                callback=callback,
            )
            stream.start()
        except Exception as e:
            print(f"[AUDIO] Не удалось открыть поток: {e}")
            return

        speech_buffer = []
        speaking = False
        silence_frames = 0
        silence_frames_to_end = int(SILENCE_END_MS / VAD_FRAME_MS)
        min_speech_frames = int(MIN_SPEECH_MS / VAD_FRAME_MS)
        max_speech_frames = int(MAX_PHRASE_SEC * 1000 / VAD_FRAME_MS)

        try:
            while True:
                try:
                    pcm16 = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                # --- ПАУЗА: сбрасываем состояние и игнорируем звук ---
                if self.paused:
                    if speaking:
                        speaking = False
                        speech_buffer = []
                        silence_frames = 0
                        print("[VAD] пауза — сброс текущей фразы")
                    continue

                if len(pcm16) != frame_bytes:
                    continue

                is_speech = vad.is_speech(pcm16, SAMPLE_RATE)

                if is_speech:
                    if not speaking:
                        speaking = True
                        silence_frames = 0
                        print("[VAD] речь началась")
                    speech_buffer.append(pcm16)
                else:
                    if speaking:
                        silence_frames += 1
                        if silence_frames < silence_frames_to_end:
                            speech_buffer.append(pcm16)
                        else:
                            speaking = False
                            silence_frames = 0
                            frames_count = len(speech_buffer)
                            if frames_count >= min_speech_frames:
                                print(f"[VAD] фраза завершена, кадров={frames_count}")
                                self._transcribe_and_emit(speech_buffer, model)
                            else:
                                print(f"[VAD] слишком короткая ({frames_count}), пропуск")
                            speech_buffer = []

                if speaking and len(speech_buffer) >= max_speech_frames:
                    print("[VAD] превышен лимит длины — режем")
                    speaking = False
                    silence_frames = 0
                    self._transcribe_and_emit(speech_buffer, model)
                    speech_buffer = []
        finally:
            stream.stop()
            stream.close()

    def _transcribe_and_emit(self, frames_list, model):
        audio_bytes = b"".join(frames_list)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

        try:
            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                task="translate",
                language=None,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
                condition_on_previous_text=False,
            )
            text = " ".join(s.text for s in segments).strip()
            if text:
                print(f"[TRANSLATE] {text}")
                self.text_queue.put(text)
            else:
                print("[TRANSLATE] пусто")
        except Exception as e:
            print(f"[TRANSLATE] ошибка: {e}")


if __name__ == "__main__":
    SpeechTranslatorApp()