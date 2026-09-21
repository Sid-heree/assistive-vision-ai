from typing import List, Tuple
from app.core.types import Detection, TextDetection
from app.core.constants import DANGER_CLASSES, SAFETY_KEYWORDS


class PriorityEngine:
    """
    Assigns a risk level ("high" | "medium" | "low") to every object
    and every piece of text in a frame, using simple, explainable rules.

    Rules are deliberately readable if/then statements — not a learned
    model — because for a safety-adjacent feature you want to be able
    to say exactly WHY something was flagged, and to tune it by hand.
    """

    def __init__(self, near_labels=("very_near", "near"),
                 danger_classes=None, safety_keywords=None):
        self.near_labels = set(near_labels)
        self.danger_classes = set(danger_classes or DANGER_CLASSES)
        self.safety_keywords = set(safety_keywords or SAFETY_KEYWORDS)

    def score_detection(self, det: Detection) -> str:
        is_near = det.depth_label in self.near_labels
        is_very_near = det.depth_label == "very_near"
        is_center = det.position == "center"
        is_low_or_mid = det.vertical in ("lower", "middle")

        # Rule 1: anything very close AND directly ahead is high priority,
        # regardless of what it is.
        if is_very_near and is_center:
            return "high"

        # Rule 2: vehicles are dangerous even if not perfectly centered —
        # a car near your left shoulder still matters.
        if det.class_name in self.danger_classes and is_near:
            return "high"

        # Rule 3: anything near, ahead, and at floor/torso height is a
        # walking-into-it hazard even if it's not "very" near yet.
        if is_near and is_center and is_low_or_mid:
            return "high"

        # Rule 4: near or medium-distance objects are worth mentioning
        # but don't need to interrupt anything.
        if det.depth_label in ("near", "medium"):
            return "medium"

        return "low"

    def score_text(self, text: TextDetection) -> str:
        lowered = text.text.lower()
        if any(keyword in lowered for keyword in self.safety_keywords):
            return "high"
        return "medium"

    def annotate(self, objects: List[Detection],
                 texts: List[TextDetection]) -> Tuple[List[Detection], List[TextDetection]]:
        """Fills in .risk on every item, in place, and returns both lists."""
        for det in objects:
            det.risk = self.score_detection(det)
        for text in texts:
            text.risk = self.score_text(text)
        return objects, texts