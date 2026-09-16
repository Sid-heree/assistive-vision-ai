import pytest
from app.core.types import Detection
from app.scene.spatial import SpatialReasoner


def make_det(x1, y1, x2, y2, w=640, h=480, name="chair"):
    return Detection(class_name=name, class_id=56, confidence=0.9,
                     bbox=[x1, y1, x2, y2], frame_width=w, frame_height=h)


@pytest.mark.parametrize("cx, expected", [
    (50,  "left"),
    (200, "left"),
    (320, "center"),
    (450, "right"),
    (600, "right"),
])
def test_horizontal_zones(cx, expected):
    det = make_det(cx - 20, 200, cx + 20, 300)
    SpatialReasoner().annotate([det])
    assert det.position == expected


def test_boundaries_are_exclusive_and_stable():
    sr = SpatialReasoner()
    assert sr.horizontal_zone(0.329) == "left"
    assert sr.horizontal_zone(0.330) == "center"   # not < boundary
    assert sr.horizontal_zone(0.660) == "center"   # not > boundary
    assert sr.horizontal_zone(0.661) == "right"


def test_resolution_independence():
    """Same relative position must give the same zone at any resolution."""
    small = make_det(60, 100, 100, 200, w=640, h=480)
    large = make_det(180, 300, 300, 600, w=1920, h=1440)
    SpatialReasoner().annotate([small, large])
    assert small.position == large.position == "left"


def test_ground_point_is_bottom_center():
    det = make_det(100, 100, 300, 400)
    assert det.ground_point == (200.0, 400)


def test_vertical_zones():
    low = make_det(300, 400, 340, 470)     # near floor
    high = make_det(300, 10, 340, 80)      # overhead
    SpatialReasoner().annotate([low, high])
    assert low.vertical == "lower"
    assert high.vertical == "upper"


def test_invalid_boundaries_rejected():
    with pytest.raises(AssertionError):
        SpatialReasoner(left_boundary=0.7, right_boundary=0.3)