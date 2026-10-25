from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from dentabridge_api.app import create_app
from dentabridge_api.clock import FrozenClock
from dentabridge_api.errors import install_error_handlers
from dentabridge_api.ratelimit import DEFAULT_BUDGETS, RateLimiter, client_key, rate_limited
from dentabridge_api.settings import Settings
from dentabridge_core import ErrorCode, ErrorResponse
from dentabridge_core.testing import FakeFareEstimator, FakeVisionBackend

START = datetime(2026, 10, 25, 12, 0, 20, tzinfo=UTC)  # 20 s into a minute


def _app(budgets: dict[str, int], clock: FrozenClock) -> TestClient:
    app = FastAPI()
    app.state.limiter = RateLimiter(budgets, clock)  # type: ignore[arg-type]
    app.state.settings = Settings(db_url="sqlite://", call_secret="s")
    app.state.clock = clock
    router = APIRouter()

    @router.get("/ping", dependencies=[Depends(rate_limited("read"))])
    def ping() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(router)
    install_error_handlers(app)

    @app.middleware("http")
    async def request_id(request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = "rid"
        return await call_next(request)

    return TestClient(app)


def test_429_after_budget_with_retry_after() -> None:
    clock = FrozenClock(START)
    client = _app({"read": 3}, clock)
    assert [client.get("/ping").status_code for _ in range(3)] == [200, 200, 200]
    blocked = client.get("/ping")
    assert blocked.status_code == 429
    body = ErrorResponse.model_validate(blocked.json())
    assert body.code == ErrorCode.RATE_LIMITED
    assert blocked.headers["retry-after"] == "40"  # the minute ends 40 s from now
    assert body.retry_after_s == 40
    clock.advance(timedelta(seconds=40))
    assert client.get("/ping").status_code == 200


def test_ratelimit_headers() -> None:
    client = _app({"read": 3}, FrozenClock(START))
    first = client.get("/ping")
    assert (first.headers["ratelimit-limit"], first.headers["ratelimit-remaining"]) == ("3", "2")
    assert first.headers["ratelimit-reset"] == "40"
    third = [client.get("/ping") for _ in range(2)][-1]
    assert third.headers["ratelimit-remaining"] == "0"


def test_budgets_are_per_client_and_per_class() -> None:
    limiter = RateLimiter({"read": 1, "write": 1}, FrozenClock(START))
    assert limiter.check("a", "read").allowed
    assert not limiter.check("a", "read").allowed
    assert limiter.check("b", "read").allowed
    assert limiter.check("a", "write").allowed


def test_the_documented_budgets() -> None:
    assert DEFAULT_BUDGETS == {"analysis": 10, "write": 30, "read": 120}


def test_raw_ip_never_stored(caplog: object) -> None:
    clock = FrozenClock(START)
    key = client_key("203.0.113.9", clock, "secret")
    assert "203.0.113.9" not in key
    assert len(key) == 32
    assert key == client_key("203.0.113.9", clock, "secret")
    clock.advance(timedelta(days=1))
    assert key != client_key("203.0.113.9", clock, "secret"), "the salt changes every day"
    assert key != client_key("203.0.113.10", clock, "secret")

    settings = Settings(db_url="sqlite://")
    app = create_app(settings, FrozenClock(START), FakeVisionBackend({}), FakeFareEstimator())
    TestClient(app).get("/v1/health")
    limiter: RateLimiter = app.state.limiter
    assert not any("testclient" in str(k) for k in limiter.state_keys())
