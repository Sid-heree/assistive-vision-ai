from app.core.types import Detection
from app.scene.priority import PriorityEngine
from app.scene.analyzer import SceneAnalyzer


def make_det(class_name="chair", position="center", vertical="middle",
             depth_label="medium"):
    return Detection(
        class_name=class_name, class_id=0, confidence=0.9,
        bbox=[100, 100, 200, 300], frame_width=640, frame_height=480,
        position=position, vertical=vertical, depth_label=depth_label,
    )


def test_empty_scene_reports_nothing():
    analyzer = SceneAnalyzer()
    scene = analyzer.build_scene([], [])
    assert analyzer.describe(scene) == "No objects detected."


def test_high_risk_object_produces_warning_phrase():
    engine = PriorityEngine()
    analyzer = SceneAnalyzer()
    det = make_det(depth_label="very_near", position="center")
    engine.annotate([det], [])
    scene = analyzer.build_scene([det], [])
    sentence = analyzer.describe(scene)
    assert "warning" in sentence.lower()


def test_routine_objects_are_grouped():
    engine = PriorityEngine()
    analyzer = SceneAnalyzer()
    dets = [make_det(position="left", depth_label="far"),
            make_det(position="left", depth_label="far")]
    engine.annotate(dets, [])
    scene = analyzer.build_scene(dets, [])
    sentence = analyzer.describe(scene)
    assert "2 chairs" in sentence


def test_scene_highest_risk_property():
    engine = PriorityEngine()
    dets = [make_det(depth_label="far"),
            make_det(depth_label="very_near", position="center")]
    engine.annotate(dets, [])
    analyzer = SceneAnalyzer()
    scene = analyzer.build_scene(dets, [])
    assert scene.highest_risk == "high"


def test_scene_to_dict_roundtrip_shape():
    engine = PriorityEngine()
    dets = [make_det()]
    engine.annotate(dets, [])
    analyzer = SceneAnalyzer()
    scene = analyzer.build_scene(dets, [])
    d = scene.to_dict()
    assert "objects" in d and "text" in d and "timestamp" in d