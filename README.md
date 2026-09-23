Full replacement: README.md
markdown
# 🧑‍🦯 Assistive Vision AI

A real-time perception system that describes surroundings aloud for visually
impaired users — combining object detection, depth estimation, OCR, spatial
reasoning, tracking, and priority-based risk assessment.

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
- [ ] Phase 2 — Custom model training (classes TBD)

## Architecture

📷 CAMERA
│
▼
Frame Processing
│
┌───┼───┬────────┐
▼ ▼ ▼ ▼
YOLO+Track Depth OCR
│ │ │
└────────┼──────┘
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


## Setup

```bash
python -m venv ai-vision
source ai-vision/bin/activate      # Windows: .\ai-vision\Scripts\activate
pip install -r requirements.txt
```

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
objects). Filled in by running `app/benchmark.py` under each real condition
— not simulated numbers.

## Known issues / limitations

- Depth is RELATIVE per-frame, not metric — "near/medium/far" only, not
  meters. True metric depth needs camera calibration + a known reference
  object, or a stereo/LiDAR sensor.
- Depth and OCR run on a frame-skip interval in the live pipeline (every
  3rd / 10th frame) for performance — distance/text can lag by a few
  frames behind reality on fast movement. The benchmark script measures
  true per-call cost with no skipping, for an honest latency picture.
- Risk rules (`app/scene/priority.py`) are hand-written heuristics, not
  learned — tune `DANGER_CLASSES` and thresholds in
  `app/core/constants.py` / `config.yaml` if false positives/negatives
  show up during testing.
- Motion detection needs `min_samples` (default 4) frames of history
  before it reports approaching/receding — very fast-approaching objects
  in their first few frames in view won't be flagged immediately.
- YOLO's 80 COCO classes do not include navigation-critical hazards like
  stairs, curbs, potholes, or crosswalks at all — these are structurally
  invisible to the current model regardless of lighting or confidence
  threshold. This motivates the custom model phase (below, TBD).

## Resolved issues (historical)
- **Day 2** — zone-boundary position flicker: fixed on Day 6 via
  `TrackHistory.smoothed_position()` (majority vote over recent frames).
- **Day 6** — `pyttsx3` on Windows silently stopping after the first
  utterance: fixed by creating a fresh engine per utterance in
  `app/voice/speaker.py` (documented SAPI5/COM engine-reuse limitation
  on Windows).
- **Day 7** — high-risk warnings (`force=True`) had no rate limit at
  all, so a person staying close to the camera for several seconds
  queued dozens of identical "very close ahead" utterances per second.
  This built up a large audio backlog that kept repeating the same
  warning long after the person had moved or left the frame entirely.
  Fixed in `app/voice/speaker.py`:
  - forced/urgent calls now use their own short cooldown
    (`urgent_repeat_interval`, default 2.5s) instead of firing on every
    frame
  - a new urgent message clears any stale queued utterances first, so
    it's heard promptly instead of waiting behind an outdated backlog
  - a `max_queue_size` cap was added to routine speech as a safety net
    against unbounded queue growth

## Phase 2 — Custom model plan

*(To be filled in — target hazard classes not yet finalized. Planned
approach: collect + annotate images for chosen classes, fine-tune from
YOLO pretrained weights via transfer learning, evaluate with
precision/recall/mAP, then integrate as a second detector feeding the
same `Detection` / `PriorityEngine` / `SceneAnalyzer` pipeline already
built in Days 5–6, so no rewrite of the fusion logic is needed.)*