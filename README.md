commit? and readme file

README.md
markdown
# 🧑‍🦯 Assistive Vision AI

A real-time perception system that describes surroundings aloud for visually
impaired users — combining object detection, depth estimation, OCR, spatial
reasoning, tracking, priority-based risk assessment, and a custom-trained
hazard detector.

> ⚠️ **Experimental prototype. Not a safety-critical navigation device.**
> A missed obstacle or incorrect depth estimate can be dangerous. This
> system communicates relative estimates, not guarantees, and should not
> be relied on as a sole means of navigation.

## Status
- [x] Day 1 — Camera + YOLO object detection
- [x] Day 2 — Spatial position reasoning
- [x] Day 3 — Monocular depth estimation
- [x] Day 4 — OCR + text-to-speech
- [x] Day 5 — Scene fusion + priority engine
- [x] Day 6 — Tracking + risk analysis
- [x] Day 7 — Interface + benchmarking + voice rate-limiting fix
- [x] Phase 2 (partial) — Custom-trained stairs detector integrated
- [ ] Phase 2 (remaining) — door, curb, pothole, crosswalk

## Architecture

📷 CAMERA
│
▼
Frame Processing
│
┌───┼───┬──────┬─────────┐
▼ ▼ ▼ ▼ ▼
YOLO(COCO) YOLO(stairs) Depth OCR
│ │ │ │
└─────┬─────┘ │ │
└────────┬───────┴──────┘
▼
Spatial Reasoning
│
▼
Track History
(motion + smoothing)
│
▼
Priority Engine
│
▼
Scene Analyzer
│
▼
🔊 TTS


Two detectors run per frame — the stock YOLO11n model (80 COCO classes)
and a custom-trained single-class model for stairs — and their outputs are
merged into one list of `Detection` objects before anything downstream
(spatial reasoning, tracking, priority, voice) runs. Neither the priority
engine nor the scene analyzer has any notion of which model produced a
given detection.

## Setup

```bash
python -m venv ai-vision
source ai-vision/bin/activate      # Windows: .\ai-vision\Scripts\activate
pip install -r requirements.txt
```

Custom model weights go in `models/custom/` (not tracked in git — see
`.gitignore`). Currently expected: `models/custom/stairs_v1.pt`.

## Running

**Command-line pipeline (OpenCV window):**
```bash
python -m app.main
```

**Streamlit demo interface:**
```bash
streamlit run app/interface/streamlit_app.py
```

**Benchmarking:**
```bash
python -m app.benchmark --condition "daylight_indoor" --frames 60
```
Results append to `benchmark_results.md`.

## Testing

```bash
python -m pytest -q
```

## Benchmark results

See [`benchmark_results.md`](./benchmark_results.md) for measured latency
across conditions (daylight, low light, indoor, outdoor, single/multiple
objects).

## Known issues / limitations

- Depth is RELATIVE per-frame, not metric — "near/medium/far" only, not
  meters.
- Depth and OCR run on a frame-skip interval in the live pipeline (every
  3rd / 10th frame) for performance — distance/text can lag a few frames
  behind reality on fast movement.
- Risk rules (`app/scene/priority.py`) are hand-written heuristics, not
  learned — tune `DANGER_CLASSES` and thresholds in `app/core/constants.py`
  / `config.yaml` as needed.
- Motion detection needs `min_samples` (default 4) frames of history
  before reporting approaching/receding.
- Running two YOLO models per frame roughly doubles detection-stage
  latency on CPU. If FPS becomes uncomfortably low, the custom detector
  can be gated to a skip-interval the same way depth/OCR are.
- **Custom stairs detector (v1):** trained on a single-class Roboflow
  Universe dataset (CC BY 4.0), fine-tuned from YOLO11n via transfer
  learning, 100 epochs with early stopping, on Colab T4.
  - Validation metrics: **mAP50 = 0.897**, **mAP50-95 = 0.625**,
    **Precision = 0.804**, **Recall = 0.840**
  - ~16% of real staircases may go undetected (recall = 0.84),
    particularly at unusual angles or partial occlusion. ~20% of
    "stairs" detections may be false positives (precision = 0.804).
    Not yet reliable enough to be a sole safety signal — treat as
    assistive, not authoritative.
- YOLO's 80 COCO classes still do not include door, curb, pothole, or
  crosswalk — these remain structurally invisible to the current
  pipeline until their own custom models are trained (Phase 2, ongoing).

## Resolved issues (historical)
- **Day 2** — zone-boundary position flicker: fixed on Day 6 via
  `TrackHistory.smoothed_position()`.
- **Day 6** — `pyttsx3` on Windows silently stopping after the first
  utterance: fixed by creating a fresh engine per utterance in
  `app/voice/speaker.py` (SAPI5/COM engine-reuse limitation on Windows).
- **Day 7** — unrated high-risk voice warnings caused a repeat-spam
  backlog. Fixed via `urgent_repeat_interval` (separate, shorter cooldown
  for forced/urgent speech) and clearing stale queued utterances when a
  new urgent message arrives.

## Phase 2 — Custom model plan

**Stairs — v1 trained and integrated.** See metrics above.

**Pipeline used (repeatable for remaining classes):**

Roboflow Universe dataset (or own collected images)
↓
Verify annotation quality + class balance
↓
Upload to Google Colab (T4 GPU), fix data.yaml paths
↓
YOLO11n pretrained weights -> transfer learning fine-tune
↓
Evaluate (precision, recall, mAP50, mAP50-95)
↓
Download best.pt -> models/custom/<class>_v1.pt
↓
Add class to DANGER_CLASSES if it's a hazard
↓
Integrate as a second Detector in app/main.py / streamlit_app.py,
merged into the existing Detection / PriorityEngine / SceneAnalyzer
pipeline (no fusion-logic changes needed per new class)


**Remaining classes:** door, curb, pothole, crosswalk — not yet started.