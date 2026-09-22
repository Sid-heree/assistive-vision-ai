import time
from collections import deque, defaultdict
from typing import List, Optional
from app.core.types import Detection


class TrackHistory:
    """
    Keeps a short rolling history of each tracked object's distance and
    position, and derives:

    1. Motion ("approaching" / "receding" / "stationary") — by comparing
       the average nearness of the older half of the history window
       against the newer half. Nearness is 0..1, higher = closer, so a
       rising trend means the object is getting closer.

    2. Smoothed position — a majority vote over the last few frames'
       position, instead of trusting a single frame. This is what fixes
       the left/center flicker noted as a known issue back on Day 2:
       a track sitting near a zone boundary now reports whichever side
       it's been on MOST of the time recently, not whichever side one
       noisy frame happened to land on.

    Tracks not seen for `stale_after` seconds are dropped, so IDs that
    walk out of frame don't leak memory forever.
    """

    def __init__(self, max_len: int = 10, stale_after: float = 2.0,
                 approach_threshold: float = 0.03, min_samples: int = 4):
        self.max_len = max_len
        self.stale_after = stale_after
        self.approach_threshold = approach_threshold
        self.min_samples = min_samples

        self._distance_history = defaultdict(lambda: deque(maxlen=max_len))
        self._position_history = defaultdict(lambda: deque(maxlen=max_len))
        self._last_seen = {}

    def update(self, detections: List[Detection]):
        """Records this frame's raw distance/position for every tracked
        detection. Call this BEFORE annotate(), using each detection's
        freshly-computed (unsmoothed) values."""
        now = time.time()
        for det in detections:
            if det.track_id is None:
                continue
            if det.distance is not None:
                self._distance_history[det.track_id].append(det.distance)
            if det.position is not None:
                self._position_history[det.track_id].append(det.position)
            self._last_seen[det.track_id] = now
        self._cleanup(now)

    def _cleanup(self, now: float):
        stale_ids = [tid for tid, t in self._last_seen.items()
                     if now - t > self.stale_after]
        for tid in stale_ids:
            self._distance_history.pop(tid, None)
            self._position_history.pop(tid, None)
            self._last_seen.pop(tid, None)

    def motion_for(self, track_id: int) -> Optional[str]:
        history = self._distance_history.get(track_id)
        if not history or len(history) < self.min_samples:
            return None  # not enough data yet to judge a trend

        values = list(history)
        half = len(values) // 2
        older, newer = values[:half], values[half:]
        if not older or not newer:
            return None

        older_avg = sum(older) / len(older)
        newer_avg = sum(newer) / len(newer)
        delta = newer_avg - older_avg  # positive = nearness increasing = approaching

        if delta > self.approach_threshold:
            return "approaching"
        if delta < -self.approach_threshold:
            return "receding"
        return "stationary"

    def smoothed_position(self, track_id: int, fallback: Optional[str]) -> Optional[str]:
        history = self._position_history.get(track_id)
        if not history:
            return fallback
        counts = {}
        for p in history:
            counts[p] = counts.get(p, 0) + 1
        return max(counts, key=counts.get)

    def annotate(self, detections: List[Detection]) -> List[Detection]:
        """Fills in .motion and overwrites .position with the smoothed
        value, for every tracked detection. Call this AFTER update()."""
        for det in detections:
            if det.track_id is None:
                continue
            det.motion = self.motion_for(det.track_id)
            det.position = self.smoothed_position(det.track_id, det.position)
        return detections

    def active_track_count(self) -> int:
        return len(self._last_seen)