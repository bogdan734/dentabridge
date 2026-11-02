import json
from pathlib import Path
from typing import Any

import pytest

from dentabridge_api.export import build_demo_app, export_openapi, record_fixtures

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
ERROR_REF = {"$ref": "#/components/schemas/ErrorResponse"}
IDEMPOTENT = {
    ("post", "/v1/analyses"),
    ("post", "/v1/analyses/sample"),
    ("post", "/v1/holds"),
    ("post", "/v1/bookings"),
    ("post", "/v1/bookings/{booking_id}/call"),
}


def _tree(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): p.read_text() for p in sorted(folder.rglob("*")) if p.is_file()
    }


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    document: dict[str, Any] = build_demo_app().openapi()
    return document


def test_openapi_matches_committed(tmp_path: Path) -> None:
    export_openapi(tmp_path / "openapi.json")
    committed = (CONTRACTS / "openapi.json").read_text()
    assert (tmp_path / "openapi.json").read_text() == committed, "run `make contracts`"


def test_every_operation_has_operation_id_and_error_schema(document: dict[str, Any]) -> None:
    ids = []
    for path, item in document["paths"].items():
        for method, operation in item.items():
            assert operation.get("operationId"), (method, path)
            ids.append(operation["operationId"])
            responses = operation["responses"]
            assert responses["default"]["content"]["application/json"]["schema"] == ERROR_REF
            if "422" in responses:
                assert responses["422"]["content"]["application/json"]["schema"] == ERROR_REF
    assert len(ids) == len(set(ids)), "operation ids name the generated client's functions"
    assert "HTTPValidationError" not in document["components"]["schemas"], "we never send it"


def test_idempotent_operations_declare_the_key_header_and_only_they_do(
    document: dict[str, Any],
) -> None:
    declared = set()
    for path, item in document["paths"].items():
        for method, operation in item.items():
            headers = [p for p in operation.get("parameters", []) if p["name"] == "Idempotency-Key"]
            if headers:
                declared.add((method, path))
                assert headers[0]["in"] == "header"
                assert headers[0]["required"] is True
    assert declared == IDEMPOTENT


def test_fixtures_regenerate_identically_and_match_the_committed_ones(tmp_path: Path) -> None:
    for scenario in ("happy_path", "errors"):
        first, second = tmp_path / "a" / scenario, tmp_path / "b" / scenario
        record_fixtures(scenario, first)
        record_fixtures(scenario, second)
        assert _tree(first), scenario
        assert _tree(first) == _tree(second), "recording twice gives the same bytes"
        assert _tree(first) == _tree(CONTRACTS / "fixtures" / scenario), "run `make contracts`"


def test_the_happy_path_walks_from_a_photo_to_the_dentists_panel(tmp_path: Path) -> None:
    record_fixtures("happy_path", tmp_path)
    steps = {
        p.name.split("-", 1)[1].removesuffix(".json"): json.loads(p.read_text())
        for p in sorted(tmp_path.glob("*.json"))
    }
    order = list(steps)
    for earlier, later in [
        ("analysis", "quote"),
        ("quote", "booking"),
        ("booking", "call-session"),
        ("call-session", "dentist-context"),
    ]:
        assert order.index(earlier) < order.index(later), (earlier, later)
    assert all(s["response"]["status"] < 300 for s in steps.values())
    context = steps["dentist-context"]["response"]["body"]
    assert context["analysis"] is not None, "the patient shared the results"
    assert steps["booking"]["response"]["body"]["simulated"] is True


def test_the_errors_scenario_shows_the_site_every_state_it_must_render(tmp_path: Path) -> None:
    record_fixtures("errors", tmp_path)
    codes = {json.loads(p.read_text())["response"]["body"]["code"] for p in tmp_path.glob("*.json")}
    assert codes >= {
        "consent_required",
        "validation_error",
        "not_found",
        "slot_unavailable",
        "share_consent_required",
        "idempotency_key_required",
        "idempotency_key_reuse",
        "trip_too_short",
        "currency_unsupported",
        "unknown_airport",
        "language_pair_unsupported",
        "disclosure_not_acknowledged",
        "invalid_call_token",
        "wrong_role",
        "booking_cancelled",
        "rate_limited",
    }
