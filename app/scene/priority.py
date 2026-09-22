from typing import List, Tuple
from app.core.types import Detection, TextDetection
from app.core.constants import DANGER_CLASSES, SAFETY_KEYWORDS


class PriorityEngine:
    """
    Assigns a risk level ("high" | "medium" | "low") to every object
    and every piece of text in a frame, using simple, explainable rules.
    Day 6 adds motion: something APPROACHING is worth flagging even
    before it's directly centered or "very near" yet.
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
        is_approaching = det.motion == "approaching"

        # Rule 1: very close AND directly ahead is high, regardless of class.
        if is_very_near and is_center:
            return "high"

        # Rule 2: vehicles that are near are dangerous even off-center.
        if det.class_name in self.danger_classes and is_near:
            return "high"

        # Rule 3: near, ahead, floor/torso height -> walking-into-it hazard.
        if is_near and is_center and is_low_or_mid:
            return "high"

        # Rule 4 (Day 6): something actively closing distance is worth a
        # warning even if it hasn't reached "very near / center" yet —
        # this is what lets the system say "approaching" BEFORE collision
        # range, not just at the moment it's already dangerously close.
        if is_approaching and is_near:
            return "high"

        # Rule 5: near or medium-distance objects worth mentioning, no rush.
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
        for det in objects:
            det.risk = self.score_detection(det)
        for text in texts:
            text.risk = self.score_text(text)
        return objects, texts