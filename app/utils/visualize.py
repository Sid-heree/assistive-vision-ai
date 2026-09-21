import cv2

ZONE_COLORS = {
    "left":   (255, 180, 0),
    "center": (0, 255, 0),
    "right":  (0, 180, 255),
}

RISK_COLORS = {
    "high": (0, 0, 255),       # red, BGR
    "medium": (0, 165, 255),   # orange
}


def draw_zones(frame, left=0.33, right=0.66):
    h, w = frame.shape[:2]
    for b in (left, right):
        x = int(b * w)
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    cv2.putText(frame, "LEFT", (int(0.10 * w), h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(frame, "CENTER", (int(0.43 * w), h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(frame, "RIGHT", (int(0.78 * w), h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    return frame


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det.bbox

        color = RISK_COLORS.get(det.risk) or ZONE_COLORS.get(det.position, (0, 255, 0))
        thickness = 3 if det.risk == "high" else 2

        depth_str = f" | {det.depth_label}" if det.depth_label else ""
        risk_str = f" | {det.risk.upper()}" if det.risk else ""
        label = f"{det.class_name} {det.confidence:.2f} | {det.position}{depth_str}{risk_str}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        gx, gy = det.ground_point
        cv2.circle(frame, (int(gx), int(gy)), 4, color, -1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def draw_text_detections(frame, text_detections):
    for t in text_detections:
        x1, y1, x2, y2 = t.bbox
        color = (0, 0, 255) if t.risk == "high" else (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'"{t.text}" {t.confidence:.2f}'
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return frame