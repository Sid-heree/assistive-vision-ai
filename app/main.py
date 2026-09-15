import time
from pathlib import Path

import cv2
import yaml

from app.camera.camera import Camera
from app.detection.detector import Detector
from app.utils.visualize import draw_detections, draw_fps

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    detector = Detector(
        model_path=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
        device=cfg["detection"]["device"],
    )

    prev_time = time.time()

    with Camera(
        source=cfg["camera"]["source"],
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
    ) as cam:

        for frame in cam.frames():
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            infer_ms = (time.perf_counter() - t0) * 1000

            frame = draw_detections(frame, detections)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if cfg["display"]["show_fps"]:
                frame = draw_fps(frame, fps)

            if detections:
                summary = ", ".join(
                    f"{d['class']} {d['confidence']:.2f}" for d in detections
                )
                print(f"[{infer_ms:5.1f} ms] {summary}")

            if cfg["display"]["show_window"]:
                cv2.imshow("Assistive Vision AI - Day 1", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()