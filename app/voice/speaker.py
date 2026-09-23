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

    Two rate limits, not one:
    - min_repeat_interval: cooldown for ROUTINE narration (e.g. 4s)
    - urgent_repeat_interval: cooldown for FORCED/high-risk warnings
      (e.g. 2s). force=True does NOT mean "speak every single frame" —
      it means "use the shorter cooldown and jump ahead of the queue,"
      not "ignore rate limiting entirely." Without this, a risk that
      stays high for several consecutive frames (very common — a person
      standing near the camera stays 'high risk' for as long as they're
      there) would queue dozens of identical utterances per second,
      creating a backlog that keeps playing stale warnings long after
      the actual scene has changed or the person has left frame.

    Also: an urgent (force=True) call clears any already-queued text
    first. This means a NEW urgent message always interrupts stale
    queued ones instead of stacking behind them.
    """

    def __init__(self, rate=170, volume=1.0, min_repeat_interval=4.0,
                 urgent_repeat_interval=2.5, max_queue_size=3):
        self._rate = rate
        self._volume = volume
        self.min_repeat_interval = min_repeat_interval
        self.urgent_repeat_interval = urgent_repeat_interval
        self.max_queue_size = max_queue_size

        self._queue = queue.Queue()
        self._last_spoken = None
        self._last_spoken_time = 0.0
        self._last_urgent_time = 0.0
        self._stop_flag = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _speak_once(self, text: str):
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

            if text is None:
                break

            try:
                self._speak_once(text)
            except Exception as e:
                print(f"[Speaker] Error speaking '{text[:50]}': {e}")

        if IS_WINDOWS:
            pythoncom.CoUninitialize()

    def _clear_queue(self):
        """Drops any pending, not-yet-spoken utterances. Used when an
        urgent message arrives, so stale warnings don't keep playing
        after the situation has already changed."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def say(self, text: str, force: bool = False):
        if not self._thread.is_alive():
            print("[Speaker] Warning: worker thread is dead, cannot speak.")
            return False

        now = time.time()

        if force:
            # Urgent path: still rate-limited (just a shorter cooldown),
            # NOT fired every frame. This is the actual bug fix.
            if now - self._last_urgent_time < self.urgent_repeat_interval:
                return False

            self._last_urgent_time = now
            self._last_spoken = text
            self._last_spoken_time = now

            # Interrupt: throw away anything stale still waiting to play,
            # so this urgent message is heard promptly, not queued behind
            # a backlog of now-outdated warnings.
            self._clear_queue()
            self._queue.put(text)
            return True

        # Routine path: same de-dup as before.
        is_repeat = (text == self._last_spoken and
                     now - self._last_spoken_time < self.min_repeat_interval)
        if is_repeat:
            return False

        # Safety net: even for routine speech, never let the queue grow
        # unbounded if TTS is somehow falling behind real-time.
        if self._queue.qsize() >= self.max_queue_size:
            return False

        self._last_spoken = text
        self._last_spoken_time = now
        self._queue.put(text)
        return True

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self):
        self._stop_flag.set()
        self._clear_queue()
        self._queue.put(None)
        self._thread.join(timeout=2)