import queue
import threading
import time
import sys

import pyttsx3

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import pythoncom


class Speaker:
    """
    Non-blocking text-to-speech with de-duplication.

    Windows/SAPI5 quirk: reusing one pyttsx3 engine instance across
    multiple runAndWait() calls in a loop is unreliable — the first
    utterance works, later ones are silently swallowed with no
    exception raised. The fix that reliably works is to create a new
    engine instance for every utterance and dispose of it right after.
    Slightly wasteful, but this is what actually speaks every time.
    """

    def __init__(self, rate=170, volume=1.0, min_repeat_interval=4.0):
        self._rate = rate
        self._volume = volume
        self.min_repeat_interval = min_repeat_interval

        self._queue = queue.Queue()
        self._last_spoken = None
        self._last_spoken_time = 0.0
        self._stop_flag = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _speak_once(self, text: str):
        """Creates a fresh engine, speaks one utterance, tears it down."""
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            del engine

    def _worker(self):
        if IS_WINDOWS:
            pythoncom.CoInitialize()

        while not self._stop_flag.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:  # sentinel to stop
                break

            try:
                self._speak_once(text)
            except Exception as e:
                print(f"[Speaker] Error speaking '{text[:50]}': {e}")

        if IS_WINDOWS:
            pythoncom.CoUninitialize()

    def say(self, text: str, force: bool = False):
        if not self._thread.is_alive():
            print("[Speaker] Warning: worker thread is dead, cannot speak.")
            return False

        now = time.time()
        is_repeat = (text == self._last_spoken and
                     now - self._last_spoken_time < self.min_repeat_interval)

        if is_repeat and not force:
            return False

        self._last_spoken = text
        self._last_spoken_time = now
        self._queue.put(text)
        return True

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self):
        self._stop_flag.set()
        self._queue.put(None)
        self._thread.join(timeout=2)