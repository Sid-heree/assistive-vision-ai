# Relative-depth thresholds. These are NOT meters — see depth_estimator.py
# docstring. Tune these by eye against your own camera/room during testing.
DEPTH_BUCKETS = [
    (0.15, "very_near"),
    (0.35, "near"),
    (0.60, "medium"),
    (1.01, "far"),   # anything above previous thresholds
]