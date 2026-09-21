import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """One detected object. Fields are filled in progressively:
    Day 1-2 perception, Day 3 depth, Day 5 risk, Day 6 tracking."""

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
    distance: Optional[float] = None     # normalized nearness 0..1 (NOT meters)
    depth_label: Optional[str] = None    # "very_near" | "near" | "medium" | "far"

    # --- Day 5: risk ---
    risk: Optional[str] = None           # "high" | "medium" | "low"

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
        """Bottom-centre of the box — where the object meets the floor."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def area_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return abs((x2 - x1) * (y2 - y1)) / (self.frame_width * self.frame_height)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self) -> str:
        pos = self.position or "?"
        risk = f" [{self.risk}]" if self.risk else ""
        return f"<{self.class_name} {self.confidence:.2f} {pos}{risk}>"


@dataclass
class TextDetection:
    """One piece of recognized text."""

    text: str
    confidence: float
    bbox: List[int]
    frame_width: int
    frame_height: int

    position: Optional[str] = None
    vertical: Optional[str] = None
    risk: Optional[str] = None           # "high" | "medium" | "low"

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def norm_center(self) -> Tuple[float, float]:
        cx, cy = self.center
        return (cx / self.frame_width, cy / self.frame_height)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self):
        return f"<Text '{self.text}' {self.position} [{self.risk}]>"


@dataclass
class Scene:
    """A unified snapshot of everything perceived in one frame."""

    objects: List[Detection] = field(default_factory=list)
    texts: List[TextDetection] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "text": [t.to_dict() for t in self.texts],
            "timestamp": self.timestamp,
        }

    @property
    def is_empty(self) -> bool:
        return not self.objects and not self.texts

    @property
    def highest_risk(self) -> Optional[str]:
        risks = [o.risk for o in self.objects if o.risk] + \
                [t.risk for t in self.texts if t.risk]
        if "high" in risks:
            return "high"
        if "medium" in risks:
            return "medium"
        if risks:
            return "low"
        return None