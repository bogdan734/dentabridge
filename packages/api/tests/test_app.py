from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dentabridge_api.app import create_app
from dentabridge_api.clock import FrozenClock
from dentabridge_api.settings import Settings
from dentabridge_core import CONTRACT_VERSION, ErrorCode, ErrorResponse
from dentabridge_core.testing import FakeFareEstimator, FakeVisionBackend

ORIGIN = "http://localhost:3000"


def make_client(**overrides: object) -> TestClient:
    settings = Settings(db_url="sqlite://", **overrides)  # type: ignore[arg-type]
    clock = FrozenClock(datetime(2026, 10, 25, 12, tzinfo=UTC))
    return TestClient(create_app(settings, clock, FakeVisionBackend({}), FakeFareEstimator()))


def test_health_and_meta_200() -> None:
    client = make_client()
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
        "mode": "demo",
        "vision_backend": "fake-vision",
    }
    meta = client.get("/v1/meta", params={"locale": "ru"}).json()
    assert meta["uploads_enabled"] is True
    assert meta["assessable"]["intraoral_photo"]
    assert meta["verified_pairs"] == [["en", "ru"], ["ru", "en"]]
    assert meta["texts"]["see_dentist"]
    assert "RUB" not in meta["currencies"]


def test_meta_texts_follow_the_locale() -> None:
    client = make_client()
    english = client.get("/v1/meta", params={"locale": "en"}).json()["texts"]
    russian = client.get("/v1/meta", params={"locale": "ru"}).json()["texts"]
    assert english.keys() == russian.keys()
    assert english["banner"] != russian["banner"]


def test_uploads_flag_reaches_meta() -> None:
    assert make_client(uploads="off").get("/v1/meta").json()["uploads_enabled"] is False


def test_errors_use_error_response_and_request_id() -> None:
    client = make_client()
    unknown = client.get("/v1/nothing-here", headers={"X-Request-Id": "abc-123"})
    assert unknown.status_code == 404
    body = ErrorResponse.model_validate(unknown.json())
    assert (body.code, body.request_id) == (ErrorCode.NOT_FOUND, "abc-123")
    assert unknown.headers["x-request-id"] == "abc-123"

    invalid = client.get("/v1/meta", params={"locale": "de"})
    assert invalid.status_code == 422
    assert ErrorResponse.model_validate(invalid.json()).code == ErrorCode.VALIDATION_ERROR
    assert invalid.headers["x-request-id"], "generated when the client sent none"


def test_cors_preflight_only_allowlisted() -> None:
    client = make_client(cors_origins=[ORIGIN])
    asked = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Idempotency-Key",
    }
    allowed = client.options("/v1/health", headers={"Origin": ORIGIN, **asked})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in allowed.headers
    assert allowed.headers["access-control-max-age"] == "600"
    refused = client.options("/v1/health", headers={"Origin": "https://evil.example", **asked})
    assert refused.status_code == 400
    assert "access-control-allow-origin" not in refused.headers


def test_cors_exposes_the_headers_the_site_reads() -> None:
    client = make_client(cors_origins=[ORIGIN])
    exposed = client.get("/v1/health", headers={"Origin": ORIGIN}).headers[
        "access-control-expose-headers"
    ]
    for name in ("X-Request-Id", "Retry-After", "Idempotent-Replayed"):
        assert name.lower() in exposed.lower()


def test_wildcard_origin_refused() -> None:
    with pytest.raises(ValidationError, match="wildcard"):
        Settings(cors_origins=["*"])
