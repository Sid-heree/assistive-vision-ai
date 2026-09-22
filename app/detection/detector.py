from ultralytics import YOLO
from app.core.types import Detection


class Detector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.4, device="cpu",
                 tracker="bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.tracker = tracker
        self.class_names = self.model.names
        print(f"[Detector] Loaded {model_path} on {device} "
              f"({len(self.class_names)} classes)")

    def detect(self, frame) -> list:
        """Plain, un-tracked detection. Kept for Day 1-5 style single-frame
        use and for tests that don't need track IDs."""
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.confidence,
                             device=self.device, verbose=False)
        return self._to_detections(results, w, h, with_tracking=False)

    def track(self, frame) -> list:
        """
        Detection + tracking: same as detect(), but every Detection also
        carries a persistent .track_id, assigned by ByteTrack (or whatever
        tracker config is passed in). `persist=True` tells Ultralytics to
        remember track state between calls on this same model instance —
        without it, every call would start tracking from scratch and IDs
        would never persist across frames.
        """
        h, w = frame.shape[:2]
        results = self.model.track(
            frame,
            conf=self.confidence,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        return self._to_detections(results, w, h, with_tracking=True)

    def _to_detections(self, results, w, h, with_tracking: bool) -> list:
        detections = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])

            track_id = None
            if with_tracking and box.id is not None:
                track_id = int(box.id[0])

            detections.append(Detection(
                class_name=self.class_names[class_id],
                class_id=class_id,
                confidence=float(box.conf[0]),
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                frame_width=w,
                frame_height=h,
                track_id=track_id,
            ))
        return detections