import numpy as np
from transformers import pipeline
from PIL import Image

from app.core.constants import DEPTH_BUCKETS


class DepthEstimator:
    """
    Wraps a monocular depth model (Depth Anything V2).

    IMPORTANT: the output is RELATIVE inverse depth, not meters.
    - Higher raw value  = CLOSER to the camera
    - Lower raw value   = FARTHER from the camera
    - Values are only meaningful relative to OTHER pixels in the SAME frame.
      Do not compare raw values across different frames or different scenes.
    """

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 device="cpu"):
        # device: -1 = CPU, 0 = first GPU (transformers pipeline convention)
        hf_device = 0 if device in ("cuda", "gpu") else -1
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=hf_device,
        )
        print(f"[DepthEstimator] Loaded {model_name} on "
              f"{'GPU' if hf_device == 0 else 'CPU'}")

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Takes an OpenCV BGR frame, returns a depth map as a numpy array
        of shape (H, W), same size as the input frame, dtype float32.
        Higher value = nearer.
        """
        # transformers pipeline expects RGB PIL Image
        rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(rgb)

        result = self.pipe(image)
        depth_map = np.array(result["depth"], dtype=np.float32)

        # The model may return a different resolution than the input.
        # Resize back to the original frame size so pixel coords line up
        # with YOLO's bounding boxes.
        if depth_map.shape != (frame_bgr.shape[0], frame_bgr.shape[1]):
            depth_map = self._resize(depth_map, frame_bgr.shape[1], frame_bgr.shape[0])

        return depth_map

    @staticmethod
    def _resize(depth_map, target_w, target_h):
        import cv2
        return cv2.resize(depth_map, (target_w, target_h),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize(depth_map: np.ndarray) -> np.ndarray:
        """Scales a depth map to 0..1 for this frame only. Needed because
        raw magnitude varies frame to frame — only relative order is stable."""
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth_map)
        return (depth_map - d_min) / (d_max - d_min)

    @staticmethod
    def bucket(normalized_value: float) -> str:
        """0..1 normalized depth -> human label. Higher = nearer."""
        # We bucket by "nearness", so invert: near objects have HIGH values
        farness = 1.0 - normalized_value
        for threshold, label in DEPTH_BUCKETS:
            if farness <= threshold:
                return label
        return DEPTH_BUCKETS[-1][1]