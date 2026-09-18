import easyocr
from app.core.types import TextDetection


class OCRReader:
    """Wraps EasyOCR and returns clean TextDetection objects,
    filtered for confidence and junk results."""

    def __init__(self, languages=None, confidence=0.5, min_text_length=2,
                 gpu=False):
        languages = languages or ["en"]
        self.confidence_threshold = confidence
        self.min_text_length = min_text_length
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print(f"[OCRReader] Loaded EasyOCR ({languages}), "
              f"gpu={gpu}, conf_threshold={confidence}")

    def read(self, frame) -> list:
        h, w = frame.shape[:2]
        raw_results = self.reader.readtext(frame)
        # Each raw result: (points, text, confidence)
        # points = 4 corner coordinates (not a simple x1,y1,x2,y2 box,
        # since text can be at an angle) -> we take the bounding rect.

        detections = []
        for points, text, conf in raw_results:
            cleaned = text.strip()

            if conf < self.confidence_threshold:
                continue
            if len(cleaned) < self.min_text_length:
                continue
            if not any(c.isalnum() for c in cleaned):
                continue  # skip pure-punctuation noise like "--" or "..."

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

            detections.append(TextDetection(
                text=cleaned,
                confidence=float(conf),
                bbox=bbox,
                frame_width=w,
                frame_height=h,
            ))
        return detections