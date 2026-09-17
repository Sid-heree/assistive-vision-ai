import time
from pathlib import Path

import cv2
import yaml

from app.camera.camera import Camera
from app.detection.detector import Detector
from app.scene.spatial import SpatialReasoner
from app.scene.describe import summarize
from app.depth.depth_estimator import DepthEstimator
from app.depth.sampler import annotate_depth
from app.utils.visualize import draw_detections, draw_fps, draw_zones

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def downscale(frame, target_width):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def main():
    cfg = load_config()
    sp_cfg = cfg["spatial"]
    d_cfg = cfg["depth"]

    detector = Detector(
        model_path=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
        device=cfg["detection"]["device"],
    )
    spatial = SpatialReasoner(
        left_boundary=sp_cfg["left_boundary"],
        right_boundary=sp_cfg["right_boundary"],
        top_boundary=sp_cfg["top_boundary"],
        bottom_boundary=sp_cfg["bottom_boundary"],
    )
    depth_estimator = DepthEstimator(
        model_name=d_cfg["model"],
        device=d_cfg["device"],
    )

    prev_time = time.time()
    frame_idx = 0
    last_depth_map = None
    depth_ms = 0.0

    with Camera(source=cfg["camera"]["source"],
                width=cfg["camera"]["width"],
                height=cfg["camera"]["height"]) as cam:

        for frame in cam.frames():
            frame_idx += 1

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000

            detections = spatial.annotate(detections)

            # Depth is expensive -> only recompute every N frames,
            # reuse the last map otherwise. Distance updates a bit
            # "laggy" but that's an acceptable tradeoff for now.
            if frame_idx % d_cfg["run_every_n_frames"] == 0 or last_depth_map is None:
                small = downscale(frame, d_cfg["downscale_width"])
                t2 = time.perf_counter()
                depth_small = depth_estimator.estimate(small)
                depth_ms = (time.perf_counter() - t2) * 1000
                # resize the small depth map back up to full frame size
                last_depth_map = DepthEstimator._resize(
                    depth_small, frame.shape[1], frame.shape[0]
                )

            detections = annotate_depth(detections, last_depth_map)

            if cfg["display"].get("show_zones", True):
                frame = draw_zones(frame, sp_cfg["left_boundary"],
                                   sp_cfg["right_boundary"])
            frame = draw_detections(frame, detections)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            if cfg["display"]["show_fps"]:
                frame = draw_fps(frame, fps)

            if frame_idx % cfg["display"].get("summary_every", 15) == 0:
                print(f"[yolo {detect_ms:5.1f}ms | depth {depth_ms:6.1f}ms] "
                      f"{summarize(detections)}")
                for det in detections:
                    print(f"    {det.class_name:10s} {det.position:6s} "
                          f"{det.depth_label:10s} (nearness={det.distance})")

            if cfg["display"]["show_window"]:
                cv2.imshow("Assistive Vision AI - Day 3", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()