import pytest

from dentabridge_voice.segment import Segment, Segmenter

RATE, FRAME_MS = 16_000, 30
FRAME = b"\x01\x00" * (RATE * FRAME_MS // 1000)


def feed(segmenter: Segmenter, pattern: list[tuple[bool, float]]) -> list[Segment]:
    """pattern: (is_speech, seconds) runs, as 30 ms frames."""
    out: list[Segment] = []
    for speech, seconds in pattern:
        for _ in range(round(seconds * 1000 / FRAME_MS)):
            out += segmenter.push(FRAME, speech)
    return out + segmenter.flush()


def test_a_pause_cuts_the_utterance_and_silence_is_trimmed() -> None:
    segments = feed(Segmenter(RATE), [(False, 1), (True, 2), (False, 0.9), (True, 1.5), (False, 1)])
    assert [(s.start_ms, s.end_ms) for s in segments] == [(990, 3000), (3900, 5400)]


def test_a_short_pause_does_not_cut() -> None:
    segments = feed(Segmenter(RATE), [(True, 1), (False, 0.3), (True, 1), (False, 1)])
    assert len(segments) == 1


def test_segments_capped_at_10s() -> None:
    segments = feed(Segmenter(RATE), [(True, 25)])
    assert [(s.start_ms, s.end_ms) for s in segments] == [(0, 9990), (9990, 19980), (19980, 24990)]
    assert all(s.end_ms - s.start_ms <= 10_000 for s in segments)
    assert all(len(s.pcm) == int((s.end_ms - s.start_ms) / 1000 * RATE) * 2 for s in segments)


def test_a_click_is_not_an_utterance() -> None:
    assert feed(Segmenter(RATE), [(False, 1), (True, 0.1), (False, 1)]) == []


def test_flush_returns_what_is_still_being_spoken() -> None:
    segmenter = Segmenter(RATE)
    for _ in range(50):
        assert segmenter.push(FRAME, True) == []
    assert len(segmenter.flush()) == 1
    assert segmenter.flush() == []


def test_a_frame_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(ValueError, match="frame"):
        Segmenter(RATE).push(FRAME[:-2], True)
