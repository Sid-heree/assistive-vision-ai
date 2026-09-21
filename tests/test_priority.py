from app.core.types import Detection, TextDetection
from app.scene.priority import PriorityEngine


def make_det(class_name="chair", position="center", vertical="middle",
             depth_label="medium"):
    return Detection(
        class_name=class_name, class_id=0, confidence=0.9,
        bbox=[100, 100, 200, 300], frame_width=640, frame_height=480,
        position=position, vertical=vertical, depth_label=depth_label,
    )


def make_text(text="hello", position="center"):
    return TextDetection(
        text=text, confidence=0.9, bbox=[10, 10, 100, 40],
        frame_width=640, frame_height=480, position=position,
    )


def test_very_near_center_is_high():
    engine = PriorityEngine()
    det = make_det(depth_label="very_near", position="center")
    assert engine.score_detection(det) == "high"


def test_far_object_is_low():
    engine = PriorityEngine()
    det = make_det(depth_label="far", position="left")
    assert engine.score_detection(det) == "low"


def test_danger_class_near_is_high_even_off_center():
    engine = PriorityEngine()
    det = make_det(class_name="car", depth_label="near", position="left")
    assert engine.score_detection(det) == "high"


def test_medium_distance_is_medium():
    engine = PriorityEngine()
    det = make_det(depth_label="medium", position="right")
    assert engine.score_detection(det) == "medium"


def test_near_center_low_vertical_is_high():
    engine = PriorityEngine()
    det = make_det(depth_label="near", position="center", vertical="lower")
    assert engine.score_detection(det) == "high"


def test_safety_keyword_text_is_high():
    engine = PriorityEngine()
    text = make_text(text="EXIT")
    assert engine.score_text(text) == "high"


def test_normal_text_is_medium():
    engine = PriorityEngine()
    text = make_text(text="Cafe Menu")
    assert engine.score_text(text) == "medium"


def test_annotate_fills_risk_on_all_items():
    engine = PriorityEngine()
    dets = [make_det(depth_label="very_near", position="center"),
            make_det(depth_label="far", position="left")]
    texts = [make_text(text="STOP")]
    engine.annotate(dets, texts)
    assert dets[0].risk == "high"
    assert dets[1].risk == "low"
    assert texts[0].risk == "high"