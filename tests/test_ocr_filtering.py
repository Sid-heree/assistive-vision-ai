from app.core.types import TextDetection


def passes_filters(text, conf, min_conf=0.5, min_len=2):
    """Mirrors the filtering logic inside OCRReader.read() so we can
    test the RULES without loading the actual EasyOCR model."""
    cleaned = text.strip()
    if conf < min_conf:
        return False
    if len(cleaned) < min_len:
        return False
    if not any(c.isalnum() for c in cleaned):
        return False
    return True


def test_low_confidence_rejected():
    assert not passes_filters("EXIT", 0.3)


def test_high_confidence_accepted():
    assert passes_filters("EXIT", 0.9)


def test_single_char_rejected():
    assert not passes_filters("E", 0.9)


def test_pure_punctuation_rejected():
    assert not passes_filters("---", 0.9)
    assert not passes_filters("....", 0.95)


def test_alphanumeric_mixed_accepted():
    assert passes_filters("ROOM 203", 0.8)