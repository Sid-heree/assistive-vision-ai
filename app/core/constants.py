# Relative-depth thresholds. These are NOT meters — see depth_estimator.py
# docstring. Tune these by eye against your own camera/room during testing.
DEPTH_BUCKETS = [
    (0.15, "very_near"),
    (0.35, "near"),
    (0.60, "medium"),
    (1.01, "far"),   # anything above previous thresholds
]

# Classes that represent a physical hazard if they're close and
# moving toward the user (vehicles), or a fall hazard regardless of
# motion (stairs — a custom-trained class, not part of COCO).
# Adjust freely — this is a judgment call, not a fixed fact.
DANGER_CLASSES = {
    "car", "bus", "truck", "train", "motorcycle", "bicycle",
    "stairs",   # custom class — falling hazard, high risk even without motion
}

# Words that, if found in OCR text, make that text safety-relevant
# regardless of size/position (an EXIT sign matters even if it's small
# and off to the side).
SAFETY_KEYWORDS = {
    "exit", "stop", "danger", "warning", "caution",
    "stairs", "emergency", "fire", "wet floor",
}