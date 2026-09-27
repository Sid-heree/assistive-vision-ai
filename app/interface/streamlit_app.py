import sys
from pathlib import Path

# streamlit run executes this file directly, so the project root is NOT
# automatically on sys.path the way `python -m app.main` puts it there.
# Add it manually, three levels up from this file (interface -> app -> root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import cv2
import streamlit as st
import yaml

from app.camera.camera import Camera
from app.detection.detector import Detector, CUSTOM_TRACK_ID_OFFSET
from app.scene.spatial import SpatialReasoner
from app.scene.priority import PriorityEngine
from app.scene.analyzer import SceneAnalyzer
from app.depth.depth_estimator import DepthEstimator
from app.depth.sampler import annotate_depth
from app.ocr.ocr_reader import OCRReader
from app.voice.speaker import Speaker
from app.tracking.history import TrackHistory

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

RISK_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟢"}
MOTION_EMOJI = {"approaching": "⬆️ approaching", "receding": "⬇️ receding",
                "stationary": "➡️ stationary"}


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading models (first run takes a minute)...")
def load_pipeline():
    """
    Loads every heavy component ONCE and keeps it alive for the life of
    the Streamlit server process. Without @st.cache_resource, Streamlit
    would reload YOLO, the depth model, and EasyOCR on every single
    widget interaction — since Streamlit reruns the whole script top to
    bottom on every rerun.
    """
    cfg = load_config()
    cd_cfg = cfg.get("custom_detection", {"enabled": False})

    detector = Detector(
        model_path=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
        device=cfg["detection"]["device"],
        tracker=cfg["tracking"]["tracker"],
        id_offset=0,
    )

    custom_detector = None
    if cd_cfg.get("enabled", False):
        custom_detector = Detector(
            model_path=cd_cfg["model"],
            confidence=cd_cfg["confidence"],
            device=cd_cfg["device"],
            tracker=cfg["tracking"]["tracker"],
            id_offset=CUSTOM_TRACK_ID_OFFSET,
        )

    spatial = SpatialReasoner(
        left_boundary=cfg["spatial"]["left_boundary"],
        right_boundary=cfg["spatial"]["right_boundary"],
        top_boundary=cfg["spatial"]["top_boundary"],
        bottom_boundary=cfg["spatial"]["bottom_boundary"],
    )
    depth_estimator = DepthEstimator(
        model_name=cfg["depth"]["model"], device=cfg["depth"]["device"],
    )
    ocr_reader = OCRReader(
        languages=cfg["ocr"]["languages"],
        confidence=cfg["ocr"]["confidence"],
        min_text_length=cfg["ocr"]["min_text_length"],
        gpu=cfg["ocr"]["gpu"],
    )
    priority_engine = PriorityEngine(near_labels=cfg["scene"]["near_labels"])
    analyzer = SceneAnalyzer(
        max_objects=cfg["scene"]["max_objects"],
        max_texts=cfg["scene"]["max_texts"],
    )
    history = TrackHistory(
        max_len=cfg["tracking"]["history_max_len"],
        stale_after=cfg["tracking"]["stale_after"],
        approach_threshold=cfg["tracking"]["approach_threshold"],
        min_samples=cfg["tracking"]["min_samples"],
    )
    speaker = Speaker(
        rate=cfg["voice"]["rate"],
        volume=cfg["voice"]["volume"],
        min_repeat_interval=cfg["voice"]["min_repeat_interval"],
        urgent_repeat_interval=cfg["voice"].get("urgent_repeat_interval", 2.5),
        max_queue_size=cfg["voice"].get("max_queue_size", 3),
    ) if cfg["voice"]["enabled"] else None

    return {
        "cfg": cfg, "detector": detector, "custom_detector": custom_detector,
        "spatial": spatial, "depth_estimator": depth_estimator,
        "ocr_reader": ocr_reader, "priority_engine": priority_engine,
        "analyzer": analyzer, "history": history, "speaker": speaker,
    }


def downscale(frame, target_width):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def format_scene_markdown(detections, texts, sentence, risk):
    lines = [f"**Overall risk:** {RISK_EMOJI.get(risk, '⚪')} {risk or 'none'}", ""]

    if detections:
        lines.append("**Objects:**")
        for d in detections:
            tid = f"#{d.track_id}" if d.track_id is not None else ""
            motion = f" — {MOTION_EMOJI.get(d.motion, '')}" if d.motion else ""
            lines.append(
                f"- {RISK_EMOJI.get(d.risk, '⚪')} {d.class_name}{tid} "
                f"({d.position}, {d.depth_label}){motion}"
            )
    else:
        lines.append("**Objects:** none")

    lines.append("")
    if texts:
        lines.append("**Text:**")
        for t in texts:
            lines.append(f'- {RISK_EMOJI.get(t.risk, "⚪")} "{t.text}" ({t.position})')
    else:
        lines.append("**Text:** none")

    lines.append("")
    lines.append(f"**Last spoken:** _{sentence}_")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Assistive Vision AI", page_icon="🧑‍🦯", layout="wide")
    st.title("🧑‍🦯 Assistive Vision AI — Live Demo")
    st.caption(
        "⚠️ Experimental prototype. Not a safety-critical navigation device. "
        "Distances are relative estimates, not meters."
    )

    pipeline = load_pipeline()
    cfg = pipeline["cfg"]

    if pipeline["custom_detector"] is not None:
        st.caption("🪜 Custom stairs detector: **active**")

    run = st.checkbox("▶️ Start camera")

    col_video, col_info = st.columns([2, 1])
    frame_placeholder = col_video.empty()
    fps_placeholder = col_video.empty()
    info_placeholder = col_info.empty()

    if not run:
        info_placeholder.info("Check the box above to start the camera.")
        return

    frame_idx = 0
    last_depth_map = None
    last_texts = []
    prev_time = time.time()

    try:
        with Camera(source=cfg["camera"]["source"],
                    width=cfg["camera"]["width"],
                    height=cfg["camera"]["height"]) as cam:

            for frame in cam.frames():
                frame_idx += 1

                detections = pipeline["detector"].track(frame)

                # Merge in custom-hazard detections (e.g. stairs) from the
                # second model on the SAME frame, exactly as app/main.py does.
                if pipeline["custom_detector"] is not None:
                    detections += pipeline["custom_detector"].track(frame)

                detections = pipeline["spatial"].annotate(detections)

                if frame_idx % cfg["depth"]["run_every_n_frames"] == 0 or last_depth_map is None:
                    small = downscale(frame, cfg["depth"]["downscale_width"])
                    depth_small = pipeline["depth_estimator"].estimate(small)
                    last_depth_map = DepthEstimator._resize(
                        depth_small, frame.shape[1], frame.shape[0]
                    )
                detections = annotate_depth(detections, last_depth_map)

                if frame_idx % cfg["ocr"]["run_every_n_frames"] == 0:
                    last_texts = pipeline["ocr_reader"].read(frame)
                    last_texts = pipeline["spatial"].annotate(last_texts)

                pipeline["history"].update(detections)
                detections = pipeline["history"].annotate(detections)

                detections, last_texts = pipeline["priority_engine"].annotate(
                    detections, last_texts
                )
                scene = pipeline["analyzer"].build_scene(detections, last_texts)
                sentence = pipeline["analyzer"].describe(scene)

                is_high_risk = scene.highest_risk == "high"
                if frame_idx % cfg["display"].get("summary_every", 15) == 0 or is_high_risk:
                    if pipeline["speaker"]:
                        pipeline["speaker"].say(sentence, force=is_high_risk)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                fps_placeholder.caption(f"FPS: {fps:.1f} | Active tracks: "
                                        f"{pipeline['history'].active_track_count()}")
                info_placeholder.markdown(
                    format_scene_markdown(detections, last_texts, sentence, scene.highest_risk)
                )

    except Exception as e:
        st.error(f"Camera loop stopped: {e}")


if __name__ == "__main__":
    main()