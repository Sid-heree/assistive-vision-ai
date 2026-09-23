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
def test_forced_calls_are_rate_limited_not_unlimited():
    """This is the Day 7 bug: force=True must NOT mean 'speak every
    single call' -- it must still respect a (short) cooldown."""
    speaker = make_fake_speaker(urgent_repeat_interval=1.0)
    results = [speaker.say("warning", force=True) for _ in range(10)]
    # Only the FIRST of 10 rapid-fire forced calls should actually queue
    assert results.count(True) == 1
    speaker.stop()


def test_forced_call_clears_stale_queue():
    speaker = make_fake_speaker(min_repeat_interval=100.0, urgent_repeat_interval=0.01)
    speaker.say("routine message one")
    speaker.say("routine message two", force=False)
    import time as _time
    _time.sleep(0.02)
    speaker.say("urgent warning", force=True)
    # queue should now contain at most the urgent message, not a backlog
    assert speaker._queue.qsize() <= 1
    speaker.stop()