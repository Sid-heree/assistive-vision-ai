from typing import List, Union

from app.core.types import Detection, TextDetection


class SpatialReasoner:
    """Converts pixel geometry into user-relative direction words.

    Works with both object detections and OCR text detections.
    """

    def __init__(
        self,
        left_boundary=0.33,
        right_boundary=0.66,
        top_boundary=0.33,
        bottom_boundary=0.66,
    ):
        assert (
            0 < left_boundary < right_boundary < 1
        ), "Invalid horizontal zones"

        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary

    def horizontal_zone(self, ncx: float) -> str:
        """Convert normalized X coordinate into a horizontal zone."""
        if ncx < self.left_boundary:
            return "left"

        if ncx > self.right_boundary:
            return "right"

        return "center"

    def vertical_zone(self, ncy: float) -> str:
        """Convert normalized Y coordinate into a vertical zone."""
        if ncy < self.top_boundary:
            return "upper"

        if ncy > self.bottom_boundary:
            return "lower"

        return "middle"

    def annotate(
        self,
        items: List[Union[Detection, TextDetection]],
    ) -> List[Union[Detection, TextDetection]]:
        """Annotate objects or text with spatial position.

        Works on any item exposing:
            - .norm_center
            - .position
            - .vertical

        The objects are modified in place and the same list is returned.
        """
        for item in items:
            ncx, ncy = item.norm_center

            item.position = self.horizontal_zone(ncx)
            item.vertical = self.vertical_zone(ncy)

        return items