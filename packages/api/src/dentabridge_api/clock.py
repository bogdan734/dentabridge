"""Time as a dependency: tests freeze it, so holds, quotes and TTLs are deterministic."""

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    def __init__(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("a clock needs an aware datetime")
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, by: timedelta) -> None:
        self._at += by
