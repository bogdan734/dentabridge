from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.testclient import TestClient
from pydantic import BaseModel

from dentabridge_api.clock import FrozenClock
from dentabridge_api.db.session import init_schema, make_engine, session_factory
from dentabridge_api.errors import ApiError, install_error_handlers
from dentabridge_api.idempotency import IdempotencyStore, IdempotentRoute
from dentabridge_core import ErrorCode, ErrorResponse

START = datetime(2026, 10, 25, 12, tzinfo=UTC)
KEY = "0b1c2d3e-aaaa"


class Order(BaseModel):
    item: str


def _client() -> tuple[TestClient, FrozenClock, IdempotencyStore, list[str]]:
    clock = FrozenClock(START)
    engine = make_engine("sqlite://")
    init_schema(engine)
    store = IdempotencyStore(session_factory(engine), clock, timedelta(hours=24))
    app = FastAPI()
    app.state.idempotency = store
    ran: list[str] = []
    router = APIRouter(route_class=IdempotentRoute)

    @router.post("/orders", status_code=201)
    def make(order: Order) -> dict[str, object]:
        ran.append(order.item)
        if order.item == "taken":
            raise ApiError(409, ErrorCode.SLOT_UNAVAILABLE)
        return {"item": order.item, "n": len(ran)}

    @router.post("/files", status_code=201)
    async def upload(file: UploadFile, request: Request) -> dict[str, object]:
        ran.append(file.filename or "")
        return {"bytes": len(await file.read()), "n": len(ran)}

    app.include_router(router)
    install_error_handlers(app)

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable) -> object:  # type: ignore[type-arg]
        request.state.request_id = "rid"
        return await call_next(request)

    return TestClient(app), clock, store, ran


def test_missing_or_malformed_key_400() -> None:
    client, *_ = _client()
    for headers in ({}, {"Idempotency-Key": "short"}, {"Idempotency-Key": "has space and !" * 2}):
        response = client.post("/orders", json={"item": "a"}, headers=headers)
        assert response.status_code == 400
        assert (
            ErrorResponse.model_validate(response.json()).code == ErrorCode.IDEMPOTENCY_KEY_REQUIRED
        )


def test_replay_same_status_and_body() -> None:
    client, _, _, ran = _client()
    first = client.post("/orders", json={"item": "a"}, headers={"Idempotency-Key": KEY})
    again = client.post("/orders", json={"item": "a"}, headers={"Idempotency-Key": KEY})
    assert (first.status_code, again.status_code) == (201, 201)
    assert again.json() == first.json()
    assert "idempotent-replayed" not in first.headers
    assert again.headers["idempotent-replayed"] == "true"
    assert ran == ["a"], "the handler ran once"


def test_reuse_with_other_body_422() -> None:
    client, *_ = _client()
    client.post("/orders", json={"item": "a"}, headers={"Idempotency-Key": KEY})
    other = client.post("/orders", json={"item": "b"}, headers={"Idempotency-Key": KEY})
    assert other.status_code == 422
    assert ErrorResponse.model_validate(other.json()).code == ErrorCode.IDEMPOTENCY_KEY_REUSE


def test_in_progress_409() -> None:
    client, _, store, ran = _client()
    import hashlib

    digest = hashlib.sha256(b'{"item":"a"}').hexdigest()
    assert store.begin(KEY, "POST /orders", digest) is None  # another request holds the key
    response = client.post(
        "/orders",
        content=b'{"item":"a"}',
        headers={"Idempotency-Key": KEY, "Content-Type": "application/json"},
    )
    assert response.status_code == 409
    assert ErrorResponse.model_validate(response.json()).code == ErrorCode.IDEMPOTENCY_IN_PROGRESS
    assert ran == []


def test_a_failed_attempt_does_not_keep_the_key() -> None:
    client, _, _, ran = _client()
    headers = {"Idempotency-Key": KEY}
    assert client.post("/orders", json={"item": "taken"}, headers=headers).status_code == 409
    assert client.post("/orders", json={"item": "taken"}, headers=headers).status_code == 409
    assert ran == ["taken", "taken"], "an error is retried, not replayed"


def test_keys_expire_after_the_ttl() -> None:
    client, clock, _, ran = _client()
    headers = {"Idempotency-Key": KEY}
    client.post("/orders", json={"item": "a"}, headers=headers)
    clock.advance(timedelta(hours=24, seconds=1))
    again = client.post("/orders", json={"item": "a"}, headers=headers)
    assert "idempotent-replayed" not in again.headers
    assert ran == ["a", "a"]


def test_a_multipart_retry_replays_although_the_boundary_differs() -> None:
    client, _, _, ran = _client()
    headers = {"Idempotency-Key": KEY}
    files = {"file": ("photo.jpg", b"jpeg bytes", "image/jpeg")}
    first = client.post("/files", files=files, headers=headers)
    again = client.post("/files", files=files, headers=headers)
    assert again.headers["idempotent-replayed"] == "true"
    assert again.json() == first.json()
    assert ran == ["photo.jpg"]
    different = client.post(
        "/files", files={"file": ("photo.jpg", b"other", "image/jpeg")}, headers=headers
    )
    assert different.status_code == 422
