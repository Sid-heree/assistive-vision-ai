import time
from pathlib import Path

import cv2
import yaml

from app.camera.camera import Camera
from app.detection.detector import Detector
from app.scene.spatial import SpatialReasoner
from app.scene.priority import PriorityEngine
from app.scene.analyzer import SceneAnalyzer
from app.depth.depth_estimator import DepthEstimator
from app.depth.sampler import annotate_depth
from app.ocr.ocr_reader import OCRReader
from app.voice.speaker import Speaker
from app.tracking.history import TrackHistory
from app.utils.visualize import (
    draw_detections, draw_fps, draw_zones, draw_text_detections, draw_track_count
)

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
    ocr_cfg = cfg["ocr"]
    v_cfg = cfg["voice"]
    scene_cfg = cfg["scene"]
    t_cfg = cfg["tracking"]

    detector = Detector(
        model_path=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
        device=cfg["detection"]["device"],
        tracker=t_cfg["tracker"],
    )
    spatial = SpatialReasoner(
        left_boundary=sp_cfg["left_boundary"],
        right_boundary=sp_cfg["right_boundary"],
        top_boundary=sp_cfg["top_boundary"],
        bottom_boundary=sp_cfg["bottom_boundary"],
    )
    depth_estimator = DepthEstimator(
        model_name=d_cfg["model"], device=d_cfg["device"],
    )
    ocr_reader = OCRReader(
        languages=ocr_cfg["languages"],
        confidence=ocr_cfg["confidence"],
        min_text_length=ocr_cfg["min_text_length"],
        gpu=ocr_cfg["gpu"],
    )
    priority_engine = PriorityEngine(
        near_labels=scene_cfg["near_labels"],
    )
    analyzer = SceneAnalyzer(
        max_objects=scene_cfg["max_objects"],
        max_texts=scene_cfg["max_texts"],
    )
    history = TrackHistory(
        max_len=t_cfg["history_max_len"],
        stale_after=t_cfg["stale_after"],
        approach_threshold=t_cfg["approach_threshold"],
        min_samples=t_cfg["min_samples"],
    )
    speaker = Speaker(
        rate=v_cfg["rate"],
        volume=v_cfg["volume"],
        min_repeat_interval=v_cfg["min_repeat_interval"],
        urgent_repeat_interval=v_cfg.get("urgent_repeat_interval", 2.5),
        max_queue_size=v_cfg.get("max_queue_size", 3),
    ) if v_cfg["enabled"] else None

    prev_time = time.time()
    frame_idx = 0
    last_depth_map = None
    last_texts = []
    depth_ms = 0.0
    ocr_ms = 0.0

    try:
        with Camera(source=cfg["camera"]["source"],
                    width=cfg["camera"]["width"],
                    height=cfg["camera"]["height"]) as cam:

            for frame in cam.frames():
                frame_idx += 1

                t0 = time.perf_counter()
                detections = detector.track(frame)
                detect_ms = (time.perf_counter() - t0) * 1000

                detections = spatial.annotate(detections)

                if frame_idx % d_cfg["run_every_n_frames"] == 0 or last_depth_map is None:
                    small = downscale(frame, d_cfg["downscale_width"])
                    t2 = time.perf_counter()
                    depth_small = depth_estimator.estimate(small)
                    depth_ms = (time.perf_counter() - t2) * 1000
                    last_depth_map = DepthEstimator._resize(
                        depth_small, frame.shape[1], frame.shape[0]
                    )
                detections = annotate_depth(detections, last_depth_map)

                if frame_idx % ocr_cfg["run_every_n_frames"] == 0:
                    t3 = time.perf_counter()
                    last_texts = ocr_reader.read(frame)
                    ocr_ms = (time.perf_counter() - t3) * 1000
                    last_texts = spatial.annotate(last_texts)

                history.update(detections)
                detections = history.annotate(detections)

                detections, last_texts = priority_engine.annotate(detections, last_texts)
                scene = analyzer.build_scene(detections, last_texts)

                frame = draw_zones(frame, sp_cfg["left_boundary"], sp_cfg["right_boundary"])
                frame = draw_detections(frame, detections)
                frame = draw_text_detections(frame, last_texts)
                frame = draw_track_count(frame, history.active_track_count())

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                if cfg["display"]["show_fps"]:
                    frame = draw_fps(frame, fps)

                is_high_risk = scene.highest_risk == "high"
                should_report = (
                    is_high_risk or
                    frame_idx % cfg["display"].get("summary_every", 15) == 0
                )

                if should_report:
                    sentence = analyzer.describe(scene)
                    print(f"[yolo {detect_ms:5.1f}ms | depth {depth_ms:6.1f}ms | "
                          f"ocr {ocr_ms:6.1f}ms | tracks {history.active_track_count()} | "
                          f"risk {scene.highest_risk}] {sentence}")

                    if speaker:
                        speaker.say(sentence, force=is_high_risk)

                if cfg["display"]["show_window"]:
                    cv2.imshow("Assistive Vision AI - Day 7", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:
        if speaker:
            speaker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()