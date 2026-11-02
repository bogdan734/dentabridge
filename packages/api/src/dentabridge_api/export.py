"""What the site builds against, generated and never edited: the OpenAPI document and fixtures.

`make contracts` rewrites contracts/ and CI fails on a diff. A fixture is one request and the answer
the API gave, recorded in demo mode against the app itself: frozen clock, counter ids, fixed secret.
"""

import argparse
import hashlib
import json
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dentabridge_api import ids
from dentabridge_api.app import create_app
from dentabridge_api.clock import FrozenClock
from dentabridge_api.settings import Settings
from dentabridge_api.testing import (
    CLINIC,
    ScriptedVision,
    an_analysis,
    book,
    consent,
    hold,
    jpeg_bytes,
    quote,
)
from dentabridge_core import Finding, FindingCode, Region
from dentabridge_pricing.airports import load_airports
from dentabridge_pricing.fares import OfflineFareEstimator, load_fare_params

Scenario = Literal["happy_path", "errors"]
NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)
SAMPLE_ID = "demo-photo"
FIXTURE_SECRET = "fixture-call-secret-not-for-use-0000"
SHOWN_HEADERS = (
    "content-type",
    "idempotent-replayed",
    "retry-after",
    "ratelimit-limit",
    "ratelimit-remaining",
    "ratelimit-reset",
)
NOBODY = "00000000-0000-4000-8000-000000000000"


def build_demo_app(
    samples_dir: Path | None = None, clock: FrozenClock | None = None, **settings: Any
) -> FastAPI:
    """The app the fixtures are recorded against (and whose document is exported)."""
    values: dict[str, Any] = {"db_url": "sqlite://", "call_secret": FIXTURE_SECRET, **settings}
    if samples_dir is not None:
        values["samples_dir"] = samples_dir
    config = Settings.model_validate(values)  # not Settings(): the environment must not leak in
    findings = (
        Finding(
            code=FindingCode.CARIES,
            status="possible",
            score=0.71,
            regions=(Region(x0=0.42, y0=0.38, x1=0.58, y1=0.55),),
        ),
        Finding(code=FindingCode.FILLING, status="not_seen", score=0.08),
    )
    fares = OfflineFareEstimator(load_fare_params(), load_airports())
    return create_app(config, clock or FrozenClock(NOW), ScriptedVision(findings), fares)


def export_openapi(out: Path) -> None:
    document = build_demo_app().openapi()
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class Recorder:
    """Calls the app and writes `NN-step.json` for each answer."""

    def __init__(self, client: TestClient, out: Path) -> None:
        self.client = client
        self.out = out
        self.count = 0

    def call(
        self,
        step: str,
        method: str,
        path: str,
        *,
        body: Any = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        files: Any = None,
        form: dict[str, str] | None = None,
    ) -> Any:
        self.count += 1
        request_id = f"fixture-{self.count:02d}"
        response = self.client.request(
            method,
            path,
            json=body,
            headers={"X-Request-Id": request_id, **(headers or {})},
            params=params,
            files=files,
            data=form,
        )
        request: dict[str, Any] = {"method": method, "path": path}
        if params:
            request["query"] = params
        if headers:
            request["headers"] = headers
        if body is not None:
            request["body"] = body
        if form is not None:
            request["multipart"] = {**form, "file": "<jpeg>"}
        answer: dict[str, Any] = {
            "status": response.status_code,
            "headers": {
                name: response.headers[name] for name in SHOWN_HEADERS if name in response.headers
            },
        }
        if response.headers["content-type"].startswith("application/json"):
            answer["body"] = response.json()
        else:
            answer["body"] = {
                "bytes": len(response.content),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            }
        document = {"step": step, "request": request, "response": answer}
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        (self.out / f"{self.count:02d}-{step}.json").write_text(text, encoding="utf-8")
        return response


def _key(step: str) -> dict[str, str]:
    return {"Idempotency-Key": f"fixture-{step}"}


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _trip(**changes: object) -> dict[str, object]:
    return {
        "clinic_id": CLINIC,
        "treatment_codes": ["crown"],
        "origin_iata": "LHR",
        "depart_date": "2026-11-10",
        "return_date": "2026-11-20",
        "travellers": 1,
        "currency": "EUR",
        **changes,
    }


def _consent_body(purpose: str) -> dict[str, object]:
    return {"purpose": purpose, "health_data_explicit": True, "text_version": "v1", "locale": "en"}


def _happy_path(r: Recorder) -> None:
    r.call("health", "GET", "/v1/health")
    r.call("meta", "GET", "/v1/meta", params={"locale": "en"})
    r.call("samples", "GET", "/v1/samples")
    r.call("sample-image", "GET", f"/v1/samples/{SAMPLE_ID}/image")
    r.call("consent-photo", "POST", "/v1/consents", body=_consent_body("photo_analysis"))
    analysis = r.call(
        "analysis",
        "POST",
        "/v1/analyses/sample",
        body={"sample_id": SAMPLE_ID, "locale": "en"},
        headers=_key("analysis"),
    ).json()
    r.call("analysis-again", "GET", f"/v1/analyses/{analysis['analysis_id']}")
    r.call("treatments", "GET", "/v1/treatments")
    r.call("cities", "GET", "/v1/cities", params={"q": "lon", "limit": 5})
    r.call("clinics", "GET", "/v1/clinics", params={"treatment": "crown"})
    trip = r.call("quote", "POST", "/v1/quotes", body=_trip()).json()
    slots = r.call(
        "slots",
        "GET",
        f"/v1/clinics/{CLINIC}/slots",
        params={"from": "2026-11-11", "to": "2026-11-14"},
    ).json()
    held = r.call(
        "hold", "POST", "/v1/holds", body={"slot_id": slots[0]["slot_id"]}, headers=_key("hold")
    ).json()
    share = r.call(
        "consent-share", "POST", "/v1/consents", body=_consent_body("share_with_clinic")
    ).json()
    booking = r.call(
        "booking",
        "POST",
        "/v1/bookings",
        body={
            "hold_id": held["hold_id"],
            "quote_id": trip["quote_id"],
            "analysis_id": analysis["analysis_id"],
            "share_consent_id": share["consent_id"],
        },
        headers=_key("booking"),
    ).json()
    session = r.call(
        "call-session",
        "POST",
        f"/v1/bookings/{booking['booking_id']}/call",
        body={"patient_language": "ru", "dentist_language": "en"},
        headers=_key("call-session"),
    ).json()
    sid = session["session_id"]
    acknowledged = {"ai_disclosure_acknowledged": True}
    r.call(
        "token-patient", "POST", f"/v1/calls/{sid}/tokens", body={"role": "patient", **acknowledged}
    )
    dentist = r.call(
        "token-dentist", "POST", f"/v1/calls/{sid}/tokens", body={"role": "dentist", **acknowledged}
    ).json()
    r.call("dentist-context", "GET", f"/v1/calls/{sid}/context", headers=_bearer(dentist["token"]))


def _errors(r: Recorder) -> None:
    client = r.client  # silent set-up goes through the client; only the failures are recorded
    r.call(
        "consent-required",
        "POST",
        "/v1/analyses",
        headers=_key("upload"),
        files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")},
        form={"consent_id": NOBODY, "modality": "intraoral_photo", "locale": "en"},
    )
    r.call("validation-error", "POST", "/v1/consents", body={})
    r.call("not-found", "GET", f"/v1/analyses/{NOBODY}")
    r.call("idempotency-key-required", "POST", "/v1/holds", body={"slot_id": NOBODY})
    slot_a, slot_b = (
        client.get(
            f"/v1/clinics/{CLINIC}/slots", params={"from": "2026-11-11", "to": "2026-11-14"}
        ).json()[i]["slot_id"]
        for i in (0, 1)
    )
    client.post("/v1/holds", json={"slot_id": slot_a}, headers=_key("setup-hold"))
    r.call(
        "slot-unavailable",
        "POST",
        "/v1/holds",
        body={"slot_id": slot_a},
        headers=_key("hold-again"),
    )
    r.call(
        "idempotency-key-reuse",
        "POST",
        "/v1/holds",
        body={"slot_id": slot_b},
        headers=_key("setup-hold"),
    )

    analysis, photo_consent = an_analysis(client, SAMPLE_ID), consent(client, "photo_analysis")
    trip = quote(client)
    held = hold(client, slot_b).json()
    wrong = book(
        client,
        held["hold_id"],
        trip["quote_id"],
        analysis_id=analysis,
        share_consent_id=photo_consent,
    )
    assert wrong.status_code == 403
    r.call(
        "share-consent-required",
        "POST",
        "/v1/bookings",
        body={
            "hold_id": held["hold_id"],
            "quote_id": trip["quote_id"],
            "analysis_id": analysis,
            "share_consent_id": photo_consent,
        },
        headers=_key("booking-wrong-consent"),
    )
    r.call("trip-too-short", "POST", "/v1/quotes", body=_trip(return_date="2026-11-11"))
    r.call("currency-unsupported", "POST", "/v1/quotes", body=_trip(currency="RUB"))
    r.call("unknown-airport", "POST", "/v1/quotes", body=_trip(origin_iata="QQQ"))

    booking = book(client, held["hold_id"], trip["quote_id"]).json()
    call = f"/v1/bookings/{booking['booking_id']}/call"
    r.call(
        "language-pair-unsupported",
        "POST",
        call,
        body={"patient_language": "en", "dentist_language": "en"},
        headers=_key("call-same"),
    )
    session = client.post(
        call, json={"patient_language": "ru", "dentist_language": "en"}, headers=_key("call-ok")
    ).json()
    sid = session["session_id"]
    tokens = f"/v1/calls/{sid}/tokens"
    r.call(
        "disclosure-not-acknowledged",
        "POST",
        tokens,
        body={"role": "patient", "ai_disclosure_acknowledged": False},
    )
    patient = client.post(
        tokens, json={"role": "patient", "ai_disclosure_acknowledged": True}
    ).json()
    r.call("invalid-call-token", "GET", f"/v1/calls/{sid}/context", headers=_bearer("not-a-token"))
    r.call("wrong-role", "GET", f"/v1/calls/{sid}/context", headers=_bearer(patient["token"]))
    client.post(f"/v1/bookings/{booking['booking_id']}/cancel")
    r.call(
        "booking-cancelled",
        "POST",
        call,
        body={"patient_language": "ru", "dentist_language": "en"},
        headers=_key("call-cancelled"),
    )
    while (
        r.call(
            "rate-limited", "POST", "/v1/consents", body=_consent_body("photo_analysis")
        ).status_code
        != 429
    ):
        r.count -= 1  # only the 429 is kept: the next attempt overwrites this file


def record_fixtures(scenario: Scenario, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.json"):
        stale.unlink()
    with tempfile.TemporaryDirectory() as tmp, ids.sequential():
        samples = Path(tmp)
        (samples / "demo.jpg").write_bytes(jpeg_bytes((320, 240)))
        (samples / "samples.json").write_text(
            json.dumps(
                [
                    {
                        "id": SAMPLE_ID,
                        "file": "demo.jpg",
                        "modality": "intraoral_photo",
                        "attribution": "Synthetic placeholder, no person",
                        "licence": "CC0",
                    }
                ]
            ),
            encoding="utf-8",
        )
        client = TestClient(build_demo_app(samples), raise_server_exceptions=False)
        {"happy_path": _happy_path, "errors": _errors}[scenario](Recorder(client, out))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dentabridge_api.export")
    parser.add_argument("contracts", type=Path, help="the contracts/ folder to rewrite")
    folder = parser.parse_args(argv).contracts
    folder.mkdir(parents=True, exist_ok=True)
    export_openapi(folder / "openapi.json")
    for scenario in ("happy_path", "errors"):
        record_fixtures(scenario, folder / "fixtures" / scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
