import time
from app.core.types import Detection
from app.tracking.history import TrackHistory


def make_det(track_id, distance, position="center"):
    return Detection(
        class_name="person", class_id=0, confidence=0.9,
        bbox=[100, 100, 200, 300], frame_width=640, frame_height=480,
        track_id=track_id, distance=distance, position=position,
    )


def test_no_motion_before_min_samples():
    hist = TrackHistory(min_samples=4)
    hist.update([make_det(1, 0.3)])
    assert hist.motion_for(1) is None  # only 1 sample, not enough yet


def test_approaching_detected_from_rising_distance():
    hist = TrackHistory(min_samples=4, approach_threshold=0.03)
    for d in [0.2, 0.25, 0.4, 0.5, 0.6]:
        hist.update([make_det(1, d)])
    assert hist.motion_for(1) == "approaching"


def test_receding_detected_from_falling_distance():
    hist = TrackHistory(min_samples=4, approach_threshold=0.03)
    for d in [0.7, 0.6, 0.4, 0.3, 0.2]:
        hist.update([make_det(1, d)])
    assert hist.motion_for(1) == "receding"


def test_stationary_when_distance_flat():
    hist = TrackHistory(min_samples=4, approach_threshold=0.05)
    for d in [0.4, 0.41, 0.39, 0.40, 0.41]:
        hist.update([make_det(1, d)])
    assert hist.motion_for(1) == "stationary"


def test_small_noise_does_not_trigger_approaching():
    """A threshold exists precisely so single-pixel-level depth jitter
    doesn't get reported as 'approaching' every other frame."""
    hist = TrackHistory(min_samples=4, approach_threshold=0.05)
    for d in [0.40, 0.41, 0.40, 0.42, 0.41]:
        hist.update([make_det(1, d)])
    assert hist.motion_for(1) == "stationary"


def test_smoothed_position_majority_vote():
    hist = TrackHistory()
    for pos in ["left", "center", "left", "left", "center"]:
        hist.update([make_det(1, 0.3, position=pos)])
    assert hist.smoothed_position(1, fallback="center") == "left"


def test_smoothed_position_falls_back_for_unknown_track():
    hist = TrackHistory()
    assert hist.smoothed_position(999, fallback="right") == "right"


def test_stale_tracks_are_forgotten():
    hist = TrackHistory(stale_after=0.05)
    hist.update([make_det(1, 0.3)])
    assert hist.active_track_count() == 1
    time.sleep(0.1)
    hist.update([])  # nothing seen this frame -> triggers cleanup
    assert hist.active_track_count() == 0


def test_different_tracks_are_independent():
    hist = TrackHistory(min_samples=4, approach_threshold=0.03)
    for d in [0.2, 0.3, 0.4, 0.5]:
        hist.update([make_det(1, d)])       # approaching
    for d in [0.5, 0.4, 0.3, 0.2]:
        hist.update([make_det(2, d)])       # receding
    assert hist.motion_for(1) == "approaching"
    assert hist.motion_for(2) == "receding"


def test_annotate_fills_motion_and_smooths_position():
    hist = TrackHistory(min_samples=2, approach_threshold=0.02)
    hist.update([make_det(1, 0.2, position="left")])
    hist.update([make_det(1, 0.3, position="left")])
    det = make_det(1, 0.4, position="center")
    hist.update([det])
    hist.annotate([det])
    assert det.motion in ("approaching", "stationary")  # deterministic given data
    assert det.position == "left"  # majority of last 3 frames was "left"