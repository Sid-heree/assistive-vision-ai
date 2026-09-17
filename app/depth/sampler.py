import numpy as np
from app.core.types import Detection
from app.depth.depth_estimator import DepthEstimator


def sample_object_depth(detection: Detection, depth_map: np.ndarray,
                         patch_frac: float = 0.10) -> float:
    """
    Returns a normalized (0..1, higher = nearer) depth value for one
    detection, using a small patch around its ground point and the
    median of that patch — robust against single-pixel noise.
    """
    h, w = depth_map.shape
    gx, gy = detection.ground_point

    x1, y1, x2, y2 = detection.bbox
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    # Patch size scales with the object's own box, capped to stay sane
    # for very large or very small detections.
    patch_w = max(3, min(int(box_w * patch_frac), 40))
    patch_h = max(3, min(int(box_h * patch_frac), 40))

    # Clamp the patch to the frame boundary — ground_point can sit right
    # at the bottom edge of the image.
    px1 = int(np.clip(gx - patch_w / 2, 0, w - 1))
    px2 = int(np.clip(gx + patch_w / 2, px1 + 1, w))
    py1 = int(np.clip(gy - patch_h / 2, 0, h - 1))
    py2 = int(np.clip(gy + patch_h / 2, py1 + 1, h))

    patch = depth_map[py1:py2, px1:px2]
    if patch.size == 0:
        return float(depth_map[int(np.clip(gy, 0, h - 1)),
                                int(np.clip(gx, 0, w - 1))])

    return float(np.median(patch))


def annotate_depth(detections, depth_map: np.ndarray) -> list:
    """Fills in .distance (normalized 0..1 nearness) and .depth_label
    on each detection, in place."""
    normalized = DepthEstimator.normalize(depth_map)
    for det in detections:
        nearness = sample_object_depth(det, normalized)
        det.distance = round(nearness, 3)
        det.depth_label = DepthEstimator.bucket(nearness)
    return detections