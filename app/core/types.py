from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """One detected object. Fields are filled in progressively:
    Day 1-2 perception, Day 3 depth, Day 6 tracking."""

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

    # --- Day 3: depth (placeholders) ---
    distance: Optional[float] = None     # metres
    depth_label: Optional[str] = None    # "very_near" | "near" | ...

    # --- Day 6: tracking (placeholders) ---
    track_id: Optional[int] = None

    # ---------- geometry ----------
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def norm_center(self) -> Tuple[float, float]:
        cx, cy = self.center
        return (cx / self.frame_width, cy / self.frame_height)

    @property
    def ground_point(self) -> Tuple[float, float]:
        """Bottom-centre of the box — where the object meets the floor.
        More meaningful than the centroid for obstacles, and the point
        we'll sample the depth map at on Day 3."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def area_ratio(self) -> float:
        """Fraction of the frame this box covers. A crude proximity
        proxy only — replaced properly by depth on Day 3."""
        x1, y1, x2, y2 = self.bbox
        return abs((x2 - x1) * (y2 - y1)) / (self.frame_width * self.frame_height)

    def to_dict(self) -> dict:
        """JSON-ready, with unfilled fields dropped."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self) -> str:
        pos = self.position or "?"
        return f"<{self.class_name} {self.confidence:.2f} {pos}>"