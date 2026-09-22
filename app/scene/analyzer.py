from collections import defaultdict
from typing import List
from app.core.types import Detection, TextDetection, Scene

SIDE_PHRASE = {"left": "on your left", "right": "on your right", "center": "ahead"}
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
DEPTH_ORDER = {"very_near": 0, "near": 1, "medium": 2, "far": 3}


class SceneAnalyzer:
    """
    Turns a Scene (already risk-scored) into a single spoken sentence:
    warnings first, routine narration after, signage last.
    """

    def __init__(self, max_objects: int = 3, max_texts: int = 2):
        self.max_objects = max_objects
        self.max_texts = max_texts

    def build_scene(self, objects: List[Detection],
                     texts: List[TextDetection]) -> Scene:
        return Scene(objects=objects, texts=texts)

    def _object_sort_key(self, d: Detection):
        return (RISK_ORDER.get(d.risk, 9), DEPTH_ORDER.get(d.depth_label, 9))

    def _text_sort_key(self, t: TextDetection):
        return RISK_ORDER.get(t.risk, 9)

    @staticmethod
    def _high_risk_phrase(det: Detection) -> str:
        side = SIDE_PHRASE.get(det.position, "nearby")

        # Most urgent framing first: already very close and dead ahead.
        if det.depth_label == "very_near" and det.position == "center":
            return f"Warning, {det.class_name} very close ahead."

        # Next: it's closing distance on you, even if not yet centered.
        if det.motion == "approaching":
            return f"Warning, {det.class_name} approaching {side}."

        depth_word = (det.depth_label or "near").replace("_", " ")
        return f"Caution, {det.class_name} {side}, {depth_word}."

    def describe(self, scene: Scene) -> str:
        objects = scene.objects
        texts = scene.texts

        if not objects and not texts:
            return "No objects detected."

        ranked_objects = sorted(objects, key=self._object_sort_key)

        warnings = [
            self._high_risk_phrase(det)
            for det in ranked_objects if det.risk == "high"
        ]

        groups = defaultdict(int)
        order = []
        for det in ranked_objects:
            if det.risk == "high":
                continue
            key = (det.class_name, det.position)
            if key not in groups:
                order.append(key)
            groups[key] += 1

        routine_parts = []
        for name, side in order[: self.max_objects]:
            count = groups[(name, side)]
            noun = name if count == 1 else f"{name}s"
            article = "a " if count == 1 else f"{count} "
            routine_parts.append(f"{article}{noun} {SIDE_PHRASE[side]}")

        ranked_texts = sorted(texts, key=self._text_sort_key)
        text_parts = []
        seen = set()
        for t in ranked_texts[: self.max_texts]:
            key = t.text.lower()
            if key in seen:
                continue
            seen.add(key)
            prefix = "Warning, sign reads" if t.risk == "high" else "Text detected"
            text_parts.append(f'{prefix} "{t.text}" {SIDE_PHRASE.get(t.position, "nearby")}')

        pieces = []
        if warnings:
            pieces.append(" ".join(warnings))
        if routine_parts:
            pieces.append("There is " + ", ".join(routine_parts) + ".")
        if text_parts:
            pieces.append(". ".join(text_parts) + ".")

        return " ".join(pieces) if pieces else "No objects detected."