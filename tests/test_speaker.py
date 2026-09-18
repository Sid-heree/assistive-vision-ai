import time
from unittest.mock import patch, MagicMock
from app.voice.speaker import Speaker


def make_fake_speaker(**kwargs):
    with patch("app.voice.speaker.pyttsx3.init") as mock_init:
        mock_init.return_value = MagicMock()
        speaker = Speaker(**kwargs)
    time.sleep(0.1)  # let the worker thread spin up
    return speaker


def test_first_utterance_is_spoken():
    speaker = make_fake_speaker(min_repeat_interval=5.0)
    assert speaker.say("hello") is True
    speaker.stop()


def test_immediate_repeat_is_suppressed():
    speaker = make_fake_speaker(min_repeat_interval=5.0)
    speaker.say("hello")
    result = speaker.say("hello")
    assert result is False
    speaker.stop()


def test_different_text_is_not_suppressed():
    speaker = make_fake_speaker(min_repeat_interval=5.0)
    speaker.say("hello")
    result = speaker.say("goodbye")
    assert result is True
    speaker.stop()


def test_repeat_after_interval_elapses():
    speaker = make_fake_speaker(min_repeat_interval=0.2)
    speaker.say("hello")
    time.sleep(0.3)
    result = speaker.say("hello")
    assert result is True
    speaker.stop()


def test_force_bypasses_dedup():
    speaker = make_fake_speaker(min_repeat_interval=10.0)
    speaker.say("hello")
    result = speaker.say("hello", force=True)
    assert result is True
    speaker.stop()