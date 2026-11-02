"""Identifiers come from here, so a recorded scenario can run on a counter instead of chance."""

import itertools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

_source: Callable[[], UUID] = uuid4


def new_id() -> UUID:
    return _source()


@contextmanager
def sequential() -> Iterator[None]:
    """Ids 00000000-0000-4000-8000-000000000001, ...2, ... while the block runs (fixtures)."""
    global _source
    counter = itertools.count(1)
    previous = _source
    _source = lambda: UUID(f"00000000-0000-4000-8000-{next(counter):012d}")  # noqa: E731
    try:
        yield
    finally:
        _source = previous
