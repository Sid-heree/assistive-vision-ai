# 🧑‍🦯 Assistive Vision AI

A real-time perception system that describes surroundings aloud for visually
impaired users — combining object detection, depth estimation, OCR, and
spatial reasoning.

> ⚠️ Experimental prototype. Not a safety-critical navigation device.

## Status
- [x] Day 1 — Camera + YOLO object detection
- [ ] Day 2 — Spatial position reasoning
- [ ] Day 3 — Monocular depth estimation
- [ ] Day 4 — OCR + text-to-speech
- [ ] Day 5 — Scene fusion + priority engine
- [ ] Day 6 — Tracking + risk analysis
- [ ] Day 7 — Interface + benchmarking

## Setup
\`\`\`bash
python -m venv ai-vision
source ai-vision/bin/activate      # Windows: .\ai-vision\Scripts\activate
pip install -r requirements.txt
python -m app.main
\`\`\`

## Architecture (current)
\`\`\`
Camera → Frame → YOLO → Detections → Visualisation
\`\`\`
## Known issues
- Zone labels flicker for objects sitting on a boundary (fix: temporal
  smoothing, planned Day 6 with tracking)
- Position is angular, not metric — "left" means a direction, not a
  displacement, until depth lands on Day 3
- [x] Day 3 — Monocular depth estimation

## Known issues (cont.)
- Depth is RELATIVE per-frame, not metric — "near/medium/far" only,
  not meters. True metric depth needs either camera calibration + a
  known reference object, or a stereo/LiDAR sensor.
- Depth runs every 3rd frame for performance; distance can lag by up
  to ~2 frames behind the object's actual position on fast movement.

- [x] Day 4 — OCR + text-to-speech

## Known issues (cont.)
- OCR runs every 10th frame; brief signage in view for < ~0.3s may be missed
- TTS de-duplication means the SAME sentence won't repeat within 4s even
  if genuinely still true — will be replaced by proper priority/change
  detection in Day 5
- No urgency/interrupt behavior yet — a new "warning" and a routine
  "chair ahead" are spoken with equal weight (Day 5/6 fix this)