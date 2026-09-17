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