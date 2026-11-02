"""Cutting a stream of microphone frames into utterances: at a pause, or after 10 seconds.

The voice-activity decision per frame comes from outside (Silero in the worker, a list in tests).
Silence before and after the speech is left out, a pause shorter than `pause_ms` stays inside it,
and a blip shorter than `min_ms` is not an utterance.
"""

from dataclasses import dataclass

MAX_MS = 10_000


@dataclass(frozen=True)
class Segment:
    pcm: bytes
    start_ms: int
    end_ms: int


class Segmenter:
    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        pause_ms: int = 600,
        min_ms: int = 300,
        max_ms: int = MAX_MS,
    ) -> None:
        self._frame_ms = frame_ms
        self._frame_bytes = sample_rate * frame_ms // 1000 * 2  # 16-bit mono
        self._pause_frames = pause_ms // frame_ms
        self._min_frames = -(-min_ms // frame_ms)
        self._max_frames = max_ms // frame_ms
        self._index = 0  # frames seen so far
        self._speech: list[bytes] = []  # from the first speech frame to the last one
        self._gap: list[bytes] = []  # the silence since the last speech frame
        self._start = 0

    def push(self, frame: bytes, is_speech: bool) -> list[Segment]:
        """Take the next frame; return the utterances it completes (one at most, or one and the
        start of the next when a pause would carry the utterance past the cap)."""
        if len(frame) != self._frame_bytes:
            raise ValueError(f"a frame is {self._frame_bytes} bytes, got {len(frame)}")
        done: list[Segment] = []
        if is_speech:
            if self._speech and len(self._speech) + len(self._gap) + 1 > self._max_frames:
                done += self._close()
            if self._speech:
                self._speech += self._gap  # a short pause stays inside the utterance
            else:
                self._start = self._index
            self._gap = []
            self._speech.append(frame)
            if len(self._speech) >= self._max_frames:
                done += self._close()
        elif self._speech:
            self._gap.append(frame)
            if len(self._gap) >= self._pause_frames:
                done += self._close()
        self._index += 1
        return done

    def flush(self) -> list[Segment]:
        """End of the stream: whatever is still being said."""
        return self._close()

    def _close(self) -> list[Segment]:
        speech, start = self._speech, self._start
        self._speech, self._gap = [], []
        if len(speech) < self._min_frames:
            return []
        return [
            Segment(
                pcm=b"".join(speech),
                start_ms=start * self._frame_ms,
                end_ms=(start + len(speech)) * self._frame_ms,
            )
        ]
