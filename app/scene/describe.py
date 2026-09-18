from collections import defaultdict
from typing import List, Optional

from app.core.types import Detection, TextDetection


SIDE_PHRASE = {
    "left": "on your left",
    "right": "on your right",
    "center": "ahead",
}


DEPTH_ORDER = {
    "very_near": 0,
    "near": 1,
    "medium": 2,
    "far": 3,
}


def summarize(
    detections: List[Detection],
    texts: Optional[List[TextDetection]] = None,
    max_items: int = 4,
    max_texts: int = 2,
) -> str:
    """Create a spoken scene summary from objects and recognized text.

    Objects are prioritized by depth when available, otherwise by
    bounding-box area.

    OCR text is added separately and duplicate text fragments are
    removed case-insensitively.
    """

    texts = texts or []

    if not detections and not texts:
        return "No objects detected."

    sentence_parts = []

    # ---------------------------------------------------------
    # Object detections
    # ---------------------------------------------------------
    if detections:

        def sort_key(d: Detection):
            """Prioritize depth when available, otherwise box size."""
            if d.depth_label is not None:
                return DEPTH_ORDER.get(d.depth_label, 99)

            # Negative because larger boxes should come first.
            return -d.area_ratio

        ranked = sorted(detections, key=sort_key)

        # Group objects by class and spatial position.
        groups = defaultdict(int)
        order = []

        for det in ranked:
            # Fall back to center if position has not been annotated.
            side = det.position or "center"

            key = (det.class_name, side)

            if key not in groups:
                order.append(key)

            groups[key] += 1

        # Build object descriptions.
        for name, side in order[:max_items]:
            count = groups[(name, side)]

            noun = name if count == 1 else f"{name}s"
            article = "a " if count == 1 else f"{count} "

            sentence_parts.append(
                f"{article}{noun} {SIDE_PHRASE.get(side, 'ahead')}"
            )

    # ---------------------------------------------------------
    # Object sentence
    # ---------------------------------------------------------
    object_sentence = ""

    if sentence_parts:
        object_sentence = (
            "There is " + ", ".join(sentence_parts) + "."
        )

    # ---------------------------------------------------------
    # OCR text
    # ---------------------------------------------------------
    text_sentence = ""

    if texts:
        # Dedupe identical text seen in multiple OCR fragments.
        # The latest occurrence is kept.
        seen = {}

        for text_detection in texts[:max_texts]:
            if not text_detection.text.strip():
                continue

            key = text_detection.text.strip().lower()

            seen[key] = text_detection

        readable = []

        for text_detection in seen.values():
            text = text_detection.text.strip()

            side = text_detection.position or "center"

            readable.append(
                f'"{text}" {SIDE_PHRASE.get(side, "ahead")}'
            )

        if readable:
            text_sentence = (
                "Text detected: "
                + ", ".join(readable)
                + "."
            )

    # ---------------------------------------------------------
    # Final sentence
    # ---------------------------------------------------------
    return (
        " ".join(
            sentence
            for sentence in (object_sentence, text_sentence)
            if sentence
        )
        or "No objects detected."
    )