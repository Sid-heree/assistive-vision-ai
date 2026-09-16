import time
from pathlib import Path

import cv2
import yaml

from app.camera.camera import Camera
from app.detection.detector import Detector
from app.scene.spatial import SpatialReasoner
from app.scene.describe import summarize
from app.utils.visualize import draw_detections, draw_fps, draw_zones

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    sp_cfg = cfg["spatial"]

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

    prev_time = time.time()
    frame_idx = 0

    with Camera(source=cfg["camera"]["source"],
                width=cfg["camera"]["width"],
                height=cfg["camera"]["height"]) as cam:

        for frame in cam.frames():
            frame_idx += 1

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            detections = spatial.annotate(detections)
            spatial_ms = (time.perf_counter() - t1) * 1000

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
                print(f"[yolo {detect_ms:5.1f}ms | spatial {spatial_ms:4.2f}ms] "
                      f"{summarize(detections)}")

            if cfg["display"]["show_window"]:
                cv2.imshow("Assistive Vision AI - Day 2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()