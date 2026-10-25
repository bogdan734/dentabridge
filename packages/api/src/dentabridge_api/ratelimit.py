"""In-memory fixed-window rate limit per client and route class.

The client key is a hash of the address and a daily salt: the raw address is never kept, and
nothing here is written to disk, so yesterday's keys cannot be linked to today's.
"""

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from fastapi import Request, Response

from dentabridge_api.clock import Clock
from dentabridge_api.errors import ApiError
from dentabridge_core import ErrorCode

RouteClass = Literal["read", "write", "analysis"]
DEFAULT_BUDGETS: dict[RouteClass, int] = {"analysis": 10, "write": 30, "read": 120}  # per minute
WINDOW_S = 60


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_s: int  # seconds until the window ends


def client_key(address: str, clock: Clock, secret: str) -> str:
    day = clock.now().date().isoformat()
    return hashlib.sha256(f"{address}|{day}|{secret}".encode()).hexdigest()[:32]


class RateLimiter:
    def __init__(self, budgets: Mapping[RouteClass, int], clock: Clock) -> None:
        self._budgets = dict(budgets)
        self._clock = clock
        self._counts: dict[tuple[str, str], tuple[int, int]] = {}  # (key, class) -> (window, n)

    def check(self, client: str, cls: RouteClass) -> RateDecision:
        now = self._clock.now().timestamp()
        window = int(now // WINDOW_S)
        reset_s = math.ceil((window + 1) * WINDOW_S - now)
        if len(self._counts) > 10_000:  # old windows are dead weight
            self._counts = {k: v for k, v in self._counts.items() if v[0] == window}
        limit = self._budgets[cls]
        seen_window, used = self._counts.get((client, cls), (window, 0))
        used = used if seen_window == window else 0
        if used >= limit:
            return RateDecision(False, limit, 0, reset_s)
        self._counts[(client, cls)] = (window, used + 1)
        return RateDecision(True, limit, limit - used - 1, reset_s)

    def state_keys(self) -> list[str]:
        return [key for key, _ in self._counts]


def rate_limited(cls: RouteClass) -> Callable[[Request, Response], None]:
    """A route dependency: counts the call, sets RateLimit-* headers, answers 429 over budget."""

    def check(request: Request, response: Response) -> None:
        state = request.app.state
        address = request.client.host if request.client else "unknown"
        key = client_key(address, state.clock, state.settings.call_secret)
        decision = state.limiter.check(key, cls)
        headers = {
            "RateLimit-Limit": str(decision.limit),
            "RateLimit-Remaining": str(decision.remaining),
            "RateLimit-Reset": str(decision.reset_s),
        }
        if not decision.allowed:
            raise ApiError(
                429, ErrorCode.RATE_LIMITED, retry_after_s=decision.reset_s, headers=headers
            )
        response.headers.update(headers)

    return check
