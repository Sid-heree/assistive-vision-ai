import numpy as np
from app.core.types import Detection
from app.depth.sampler import sample_object_depth
from app.depth.depth_estimator import DepthEstimator


def make_det(x1, y1, x2, y2, w=100, h=100):
    return Detection(class_name="test", class_id=0, confidence=0.9,
                     bbox=[x1, y1, x2, y2], frame_width=w, frame_height=h)


def test_uniform_depth_returns_that_value():
    depth_map = np.full((100, 100), 0.5, dtype=np.float32)
    det = make_det(20, 20, 60, 80)
    result = sample_object_depth(det, depth_map)
    assert abs(result - 0.5) < 1e-6


def test_median_ignores_outlier_pixels():
    depth_map = np.full((100, 100), 0.4, dtype=np.float32)
    # Corrupt a couple of pixels near the ground point with a wild outlier
    depth_map[78:80, 38:40] = 50.0
    det = make_det(20, 20, 60, 80)  # ground_point = (40, 80)
    result = sample_object_depth(det, depth_map)
    # Median should stay near 0.4 despite the outlier patch
    assert result < 1.0


def test_ground_point_edge_of_frame_does_not_crash():
    depth_map = np.random.rand(100, 100).astype(np.float32)
    det = make_det(0, 90, 20, 99)  # ground point right at bottom edge
    result = sample_object_depth(det, depth_map)
    assert 0.0 <= result <= 1.0


def test_bucket_ordering_near_is_high_nearness():
    assert DepthEstimator.bucket(0.95) == "very_near"
    assert DepthEstimator.bucket(0.02) == "far"


def test_normalize_handles_flat_map():
    flat = np.full((10, 10), 3.0, dtype=np.float32)
    normed = DepthEstimator.normalize(flat)
    assert np.all(normed == 0.0)  # no crash on divide-by-zero


def test_normalize_scales_to_unit_range():
    depth_map = np.array([[1.0, 5.0], [3.0, 9.0]], dtype=np.float32)
    normed = DepthEstimator.normalize(depth_map)
    assert normed.min() == 0.0
    assert normed.max() == 1.0