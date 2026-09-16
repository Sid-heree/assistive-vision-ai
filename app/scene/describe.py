from collections import defaultdict
from typing import List
from app.core.types import Detection

SIDE_PHRASE = {
    "left": "on your left",
    "right": "on your right",
    "center": "ahead",
}


def summarize(detections: List[Detection], max_items: int = 4) -> str:
    """Rule-based scene sentence. Deliberately minimal — Day 5 replaces
    this with a real analyzer that reasons about priority and depth."""
    if not detections:
        return "No objects detected."

    # Biggest boxes first as a stand-in for 'nearest'. Day 3 swaps this
    # for true depth ordering.
    ranked = sorted(detections, key=lambda d: d.area_ratio, reverse=True)

    # Group by (class, side) so three chairs become "3 chairs on your left"
    groups = defaultdict(int)
    order = []
    for det in ranked:
        key = (det.class_name, det.position)
        if key not in groups:
            order.append(key)
        groups[key] += 1

    parts = []
    for name, side in order[:max_items]:
        count = groups[(name, side)]
        noun = name if count == 1 else f"{name}s"
        article = "a " if count == 1 else f"{count} "
        parts.append(f"{article}{noun} {SIDE_PHRASE[side]}")

    return "There is " + ", ".join(parts) + "."