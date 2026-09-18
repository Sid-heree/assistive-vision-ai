from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """One detected object. Fields are filled in progressively:
    Day 1-2 perception, Day 3 depth, Day 6 tracking.
    """

    # --- Day 1: raw detection ---
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]                 # [x1, y1, x2, y2] in pixels
    frame_width: int
    frame_height: int

    # --- Day 2: spatial ---
    position: Optional[str] = None       # "left" | "center" | "right"
    vertical: Optional[str] = None       # "upper" | "middle" | "lower"

    # --- Day 3: depth ---
    distance: Optional[float] = None     # metres
    depth_label: Optional[str] = None    # "very_near" | "near" | ...

    # --- Day 6: tracking ---
    track_id: Optional[int] = None

    # ---------- geometry ----------

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def norm_center(self) -> Tuple[float, float]:
        """Center point normalized to 0-1 range."""
        cx, cy = self.center
        return (
            cx / self.frame_width,
            cy / self.frame_height
        )

    @property
    def ground_point(self) -> Tuple[float, float]:
        """Bottom-center of the bounding box.

        This approximates where the object meets the floor and is
        useful for depth sampling and obstacle positioning.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def area_ratio(self) -> float:
        """Fraction of the frame covered by the bounding box.

        This is only a crude proximity estimate and should not
        replace actual depth estimation.
        """
        x1, y1, x2, y2 = self.bbox

        return abs(
            (x2 - x1) * (y2 - y1)
        ) / (
            self.frame_width * self.frame_height
        )

    def to_dict(self) -> dict:
        """Convert to a JSON-ready dictionary.

        Fields that are still None are removed.
        """
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
        }

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        pos = self.position or "?"
        return f"<{self.class_name} {self.confidence:.2f} {pos}>"


@dataclass
class TextDetection:
    """One piece of recognized text.

    Separate from Detection because OCR confidence and geometry
    behave differently from object detection.
    """

    # --- OCR result ---
    text: str
    confidence: float
    bbox: List[int]              # [x1, y1, x2, y2] in pixels

    # --- Frame information ---
    frame_width: int
    frame_height: int

    # --- Spatial information ---
    position: Optional[str] = None       # "left" | "center" | "right"
    vertical: Optional[str] = None       # "upper" | "middle" | "lower"

    # ---------- geometry ----------

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the text bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def norm_center(self) -> Tuple[float, float]:
        """Center point normalized to 0-1 range."""
        cx, cy = self.center

        return (
            cx / self.frame_width,
            cy / self.frame_height
        )

    def to_dict(self) -> dict:
        """Convert to a JSON-ready dictionary.

        Fields that are still None are removed.
        """
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
        }

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        pos = self.position or "?"
        return f"<Text '{self.text}' {pos}>"