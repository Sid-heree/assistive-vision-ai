from typing import List
from app.core.types import Detection


class SpatialReasoner:
    """Converts pixel geometry into user-relative direction words."""

    def __init__(self, left_boundary=0.33, right_boundary=0.66,
                 top_boundary=0.33, bottom_boundary=0.66):
        assert 0 < left_boundary < right_boundary < 1, "Invalid horizontal zones"
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary

    def horizontal_zone(self, ncx: float) -> str:
        if ncx < self.left_boundary:
            return "left"
        if ncx > self.right_boundary:
            return "right"
        return "center"

    def vertical_zone(self, ncy: float) -> str:
        if ncy < self.top_boundary:
            return "upper"
        if ncy > self.bottom_boundary:
            return "lower"
        return "middle"

    def annotate(self, detections: List[Detection]) -> List[Detection]:
        """Fills in .position and .vertical on every detection, in place."""
        for det in detections:
            ncx, ncy = det.norm_center
            det.position = self.horizontal_zone(ncx)
            det.vertical = self.vertical_zone(ncy)
        return detections