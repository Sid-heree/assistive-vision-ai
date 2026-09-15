from ultralytics import YOLO


class Detector:
    """Runs YOLO on a frame and returns a clean list of detections."""

    def __init__(self, model_path="yolo11n.pt", confidence=0.4, device="cpu"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.class_names = self.model.names
        print(f"[Detector] Loaded {model_path} on {device} "
              f"({len(self.class_names)} classes)")

    def detect(self, frame):
        """
        Returns a list of dicts:
        {"class": "chair", "class_id": 56, "confidence": 0.88,
         "bbox": [x1, y1, x2, y2]}
        """
        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            detections.append({
                "class": self.class_names[class_id],
                "class_id": class_id,
                "confidence": float(box.conf[0]),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })
        return detections