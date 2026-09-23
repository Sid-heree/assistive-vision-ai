import argparse
import statistics
import time
from datetime import datetime
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
from app.tracking.history import TrackHistory
from app.utils.timer import timer

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "benchmark_results.md"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def downscale(frame, target_width):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def run_benchmark(condition: str, frames: int, source):
    cfg = load_config()

    print(f"[Benchmark] Loading models...")
    detector = Detector(
        model_path=cfg["detection"]["model"], confidence=cfg["detection"]["confidence"],
        device=cfg["detection"]["device"], tracker=cfg["tracking"]["tracker"],
    )
    spatial = SpatialReasoner(**{k: cfg["spatial"][k] for k in
                                  ["left_boundary", "right_boundary",
                                   "top_boundary", "bottom_boundary"]})
    depth_estimator = DepthEstimator(model_name=cfg["depth"]["model"], device=cfg["depth"]["device"])
    ocr_reader = OCRReader(languages=cfg["ocr"]["languages"], confidence=cfg["ocr"]["confidence"],
                           min_text_length=cfg["ocr"]["min_text_length"], gpu=cfg["ocr"]["gpu"])
    priority_engine = PriorityEngine(near_labels=cfg["scene"]["near_labels"])
    analyzer = SceneAnalyzer(max_objects=cfg["scene"]["max_objects"], max_texts=cfg["scene"]["max_texts"])
    history = TrackHistory(**{k: cfg["tracking"][k] for k in
                               ["history_max_len", "stale_after",
                                "approach_threshold", "min_samples"]}) \
        if False else TrackHistory(
            max_len=cfg["tracking"]["history_max_len"],
            stale_after=cfg["tracking"]["stale_after"],
            approach_threshold=cfg["tracking"]["approach_threshold"],
            min_samples=cfg["tracking"]["min_samples"],
        )

    timings = {"yolo": [], "depth": [], "ocr": [], "scene": [], "total": []}

    print(f"[Benchmark] Warming up (first inference is always slower)...")
    with Camera(source=source, width=cfg["camera"]["width"],
                height=cfg["camera"]["height"]) as cam:
        frame_iter = cam.frames()
        warmup_frame = next(frame_iter)
        detector.track(warmup_frame)
        depth_estimator.estimate(downscale(warmup_frame, cfg["depth"]["downscale_width"]))
        ocr_reader.read(warmup_frame)

        print(f"[Benchmark] Running {frames} timed frames under condition: '{condition}'")
        count = 0
        for frame in frame_iter:
            if count >= frames:
                break
            count += 1

            with timer() as t_total:
                with timer() as t_yolo:
                    detections = detector.track(frame)
                detections = spatial.annotate(detections)

                with timer() as t_depth:
                    small = downscale(frame, cfg["depth"]["downscale_width"])
                    depth_map_small = depth_estimator.estimate(small)
                    depth_map = DepthEstimator._resize(depth_map_small, frame.shape[1], frame.shape[0])
                detections = annotate_depth(detections, depth_map)

                with timer() as t_ocr:
                    texts = ocr_reader.read(frame)
                texts = spatial.annotate(texts)

                with timer() as t_scene:
                    history.update(detections)
                    detections = history.annotate(detections)
                    detections, texts = priority_engine.annotate(detections, texts)
                    scene = analyzer.build_scene(detections, texts)
                    _ = analyzer.describe(scene)

            timings["yolo"].append(t_yolo["ms"])
            timings["depth"].append(t_depth["ms"])
            timings["ocr"].append(t_ocr["ms"])
            timings["scene"].append(t_scene["ms"])
            timings["total"].append(t_total["ms"])

            print(f"  frame {count}/{frames}  total={t_total['ms']:6.1f}ms", end="\r")

    print()
    return timings


def summarize(timings: dict) -> dict:
    summary = {}
    for stage, values in timings.items():
        if not values:
            continue
        summary[stage] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return summary


def print_table(condition: str, summary: dict):
    print(f"\n=== Results: {condition} ===")
    print(f"{'Stage':<10} {'Mean(ms)':>10} {'Median(ms)':>12} {'Max(ms)':>10}")
    for stage, s in summary.items():
        print(f"{stage:<10} {s['mean']:>10.1f} {s['median']:>12.1f} {s['max']:>10.1f}")
    if "total" in summary:
        fps = 1000.0 / summary["total"]["mean"]
        print(f"\nEffective FPS (all stages every frame, no skip-optimization): {fps:.1f}")


def append_to_markdown(condition: str, summary: dict, frame_count: int):
    RESULTS_PATH.touch(exist_ok=True)
    existing = RESULTS_PATH.read_text()

    header = "| Condition | Frames | YOLO ms | Depth ms | OCR ms | Scene ms | Total ms | FPS |\n"
    separator = "|---|---|---|---|---|---|---|---|\n"

    if "| Condition |" not in existing:
        existing = "# Benchmark Results\n\n" + header + separator + existing

    fps = 1000.0 / summary["total"]["mean"] if "total" in summary else 0.0
    row = (f"| {condition} | {frame_count} | "
           f"{summary.get('yolo', {}).get('mean', 0):.1f} | "
           f"{summary.get('depth', {}).get('mean', 0):.1f} | "
           f"{summary.get('ocr', {}).get('mean', 0):.1f} | "
           f"{summary.get('scene', {}).get('mean', 0):.1f} | "
           f"{summary.get('total', {}).get('mean', 0):.1f} | "
           f"{fps:.1f} |\n")

    RESULTS_PATH.write_text(existing + row)
    print(f"\n[Benchmark] Appended row to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the assistive vision pipeline")
    parser.add_argument("--condition", type=str, required=True,
                        help="Label for this test run, e.g. 'daylight_indoor_single_object'")
    parser.add_argument("--frames", type=int, default=60,
                        help="Number of frames to time (default: 60)")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index or video file path (default: 0)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    timings = run_benchmark(args.condition, args.frames, source)
    summary = summarize(timings)
    print_table(args.condition, summary)
    append_to_markdown(args.condition, summary, args.frames)


if __name__ == "__main__":
    main()