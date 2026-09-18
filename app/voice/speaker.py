import queue
import threading
import time

import pyttsx3


class Speaker:
    """
    Non-blocking text-to-speech with de-duplication.

    Two problems this solves:
    1. pyttsx3's .say()+.runAndWait() BLOCKS the calling thread until
       speech finishes. Call it directly in your main loop and your
       camera feed freezes for the duration of every sentence.
    2. Without de-duplication, an unchanged scene re-triggers the same
       sentence every single frame -> the queue explodes instantly.
    """

    def __init__(self, rate=170, volume=1.0, min_repeat_interval=4.0):
        self._queue = queue.Queue()
        self._last_spoken = None
        self._last_spoken_time = 0.0
        self.min_repeat_interval = min_repeat_interval  # seconds

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        self._rate = rate
        self._volume = volume

    def _worker(self):
        # pyttsx3 engine must be created and used on the SAME thread
        # it's driven from -> create it inside the worker thread.
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)

        while not self._stop_flag.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:  # sentinel to stop
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, text: str, force: bool = False):
        """
        Queues text to be spoken, unless:
        - it's identical to the last thing spoken, AND
        - less than `min_repeat_interval` seconds have passed
        `force=True` bypasses de-duplication (for urgent warnings later).
        """
        now = time.time()
        is_repeat = (text == self._last_spoken and
                     now - self._last_spoken_time < self.min_repeat_interval)

        if is_repeat and not force:
            return False

        self._last_spoken = text
        self._last_spoken_time = now
        self._queue.put(text)
        return True

    def stop(self):
        self._stop_flag.set()
        self._queue.put(None)
        self._thread.join(timeout=2)