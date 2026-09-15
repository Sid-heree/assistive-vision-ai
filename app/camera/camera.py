import cv2


class Camera:
    """Wraps an OpenCV video source and yields frames one at a time."""

    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened source {source} at {self.width}x{self.height}")

    def frames(self):
        """Generator that yields frames until the stream ends or is stopped."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[Camera] Failed to read frame. Stopping.")
                break
            yield frame

    def release(self):
        self.cap.release()

    # Lets you use: with Camera() as cam:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()