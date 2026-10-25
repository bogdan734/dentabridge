import itertools
import logging
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from dentabridge_api.clock import FrozenClock
from dentabridge_api.testing import ScriptedVision, jpeg_bytes
from dentabridge_core import AnalysisResult, ErrorCode, ErrorResponse, Finding, FindingCode
from dentabridge_core.policy import lint

KEY = {"Idempotency-Key": "analysis-0001"}
_keys = itertools.count(100)
PHOTO = {"file": ("secret-name.jpg", jpeg_bytes(), "image/jpeg")}


def consent(client: Any, purpose: str = "photo_analysis", locale: str = "en") -> str:
    response = client.post(
        "/v1/consents",
        json={
            "purpose": purpose,
            "health_data_explicit": True,
            "text_version": "v1",
            "locale": locale,
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["consent_id"])


def analyse(client: Any, consent_id: str, **fields: str) -> httpx.Response:
    data = {"consent_id": consent_id, "modality": "intraoral_photo", "locale": "en", **fields}
    headers = {"Idempotency-Key": f"analysis-{next(_keys):04d}"}
    response: httpx.Response = client.post("/v1/analyses", data=data, files=PHOTO, headers=headers)
    return response


def code_of(response: httpx.Response) -> ErrorCode:
    return ErrorResponse.model_validate(response.json()).code


def test_consent_roundtrip_and_text_version(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    record = client.post(
        "/v1/consents",
        json={
            "purpose": "photo_analysis",
            "health_data_explicit": True,
            "text_version": "v1",
            "locale": "ru",
        },
    ).json()
    assert record["expires_at"] > record["given_at"]
    assert (
        client.post(
            "/v1/consents",
            json={
                "purpose": "photo_analysis",
                "health_data_explicit": True,
                "text_version": "v0",
                "locale": "en",
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/consents",
            json={
                "purpose": "photo_analysis",
                "health_data_explicit": False,
                "text_version": "v1",
                "locale": "en",
            },
        ).status_code
        == 422
    ), "consent has to be explicit"


def test_analysis_with_consent_returns_the_contract(make_client: Any) -> None:
    findings = (Finding(code=FindingCode.CARIES, status="possible"),)
    client = make_client(ScriptedVision(findings))
    response = analyse(client, consent(client))
    assert response.status_code == 201, response.text
    result = AnalysisResult.model_validate(response.json())
    assert result.urgency.level == "soon"
    assert {f.code for f in result.findings} == set(FindingCode), "every code exactly once"
    assert next(f for f in result.findings if f.code == FindingCode.CARIES).status == "possible"
    assert next(f for f in result.findings if f.code == FindingCode.CROWN).status == "not_assessed"


def test_patient_text_passes_the_runtime_lint(make_client: Any) -> None:
    client = make_client(ScriptedVision((Finding(code=FindingCode.GINGIVITIS, status="possible"),)))
    for locale in ("en", "ru"):
        result = AnalysisResult.model_validate(
            analyse(client, consent(client, locale=locale), locale=locale).json()
        )
        text = result.patient_text
        for line in (*text.observations, text.urgency_line, text.see_dentist_line, text.disclaimer):
            assert not lint(line, locale, "patient")


def test_consent_required_403(make_client: Any) -> None:
    client = make_client()
    unknown = analyse(client, "00000000-0000-4000-8000-000000000000")
    assert (unknown.status_code, code_of(unknown)) == (403, ErrorCode.CONSENT_REQUIRED)
    wrong = analyse(client, consent(client, purpose="share_with_clinic"))
    assert (wrong.status_code, code_of(wrong)) == (403, ErrorCode.CONSENT_REQUIRED)


def test_expired_consent_410(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    consent_id = consent(client)
    clock.advance(timedelta(hours=24, seconds=1))
    response = analyse(client, consent_id)
    assert (response.status_code, code_of(response)) == (410, ErrorCode.CONSENT_EXPIRED)


def test_uploads_off_403(make_client: Any) -> None:
    client = make_client(uploads="off")
    response = analyse(client, consent(client))
    assert (response.status_code, code_of(response)) == (403, ErrorCode.UPLOADS_DISABLED)


def test_400_413_415_through_the_route(make_client: Any) -> None:
    client = make_client()
    consent_id = consent(client)
    data = {"consent_id": consent_id, "modality": "intraoral_photo", "locale": "en"}
    bad = client.post(
        "/v1/analyses", data=data, files={"file": ("x.jpg", b"nope", "image/jpeg")}, headers=KEY
    )
    assert (bad.status_code, code_of(bad)) == (400, ErrorCode.IMAGE_UNREADABLE)
    wrong_type = client.post(
        "/v1/analyses",
        data=data,
        files={"file": ("x.txt", b"hello", "text/plain")},
        headers={"Idempotency-Key": "analysis-0002"},
    )
    assert (wrong_type.status_code, code_of(wrong_type)) == (415, ErrorCode.UNSUPPORTED_MEDIA_TYPE)
    huge = jpeg_bytes((3200, 3200), noise=True)
    assert len(huge) > 8 * 1024 * 1024
    too_big = client.post(
        "/v1/analyses",
        data=data,
        files={"file": ("x.jpg", huge, "image/jpeg")},
        headers={"Idempotency-Key": "analysis-0003"},
    )
    assert (too_big.status_code, code_of(too_big)) == (413, ErrorCode.PAYLOAD_TOO_LARGE)


def test_a_repeated_key_replays_without_a_second_analysis(make_client: Any) -> None:
    vision = ScriptedVision()
    client = make_client(vision)
    consent_id = consent(client)
    data = {"consent_id": consent_id, "modality": "intraoral_photo", "locale": "en"}
    first = client.post("/v1/analyses", data=data, files=PHOTO, headers=KEY)
    again = client.post("/v1/analyses", data=data, files=PHOTO, headers=KEY)
    assert again.headers["idempotent-replayed"] == "true"
    assert again.json() == first.json()
    assert vision.calls == 1


def test_backend_unavailable_503(make_client: Any) -> None:
    class Down:
        model = ScriptedVision.model

        async def analyse(self, image: Any, modality: Any) -> Any:
            raise httpx.ConnectError("no server")

    client = make_client(Down())
    response = analyse(client, consent(client))
    assert (response.status_code, code_of(response)) == (503, ErrorCode.BACKEND_UNAVAILABLE)


def test_no_file_written(make_client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def spooled_to_disk(self: object) -> None:
        raise AssertionError("an upload was rolled over to a temporary file")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", spooled_to_disk)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    client = make_client()
    consent_id = consent(client)
    big = jpeg_bytes((1600, 1200), noise=True)  # a few MB: over Starlette's 1 MB default spool
    assert len(big) > 1024 * 1024
    response = client.post(
        "/v1/analyses",
        data={"consent_id": consent_id, "modality": "intraoral_photo", "locale": "en"},
        files={"file": ("big.jpg", big, "image/jpeg")},
        headers=KEY,
    )
    assert response.status_code == 201, response.text
    assert list(workdir.rglob("*")) == []


def test_no_image_bytes_in_logs(make_client: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = make_client()
    analyse(client, consent(client))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret-name" not in logged
    assert "testclient" not in logged
    assert "POST /v1/analyses" in logged, "the access line itself is there"


def test_result_can_be_read_deleted_and_expires(make_client: Any, clock: FrozenClock) -> None:
    client = make_client()
    created = analyse(client, consent(client)).json()
    url = f"/v1/analyses/{created['analysis_id']}"
    assert client.get(url).json() == created
    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404

    other = analyse(client, consent(client)).json()
    clock.advance(timedelta(hours=24, seconds=1))
    gone = client.get(f"/v1/analyses/{other['analysis_id']}")
    assert (gone.status_code, code_of(gone)) == (410, ErrorCode.GONE)


def test_samples_listing_image_and_analysis(make_client: Any) -> None:
    client = make_client()
    listing = client.get("/v1/samples").json()
    assert listing == [
        {
            "id": "one",
            "attribution": "A. Author, Sample one, https://example.test, CC BY 4.0",
            "licence": "CC BY 4.0",
        }
    ]
    image = client.get("/v1/samples/one/image")
    assert (image.status_code, image.headers["content-type"]) == (200, "image/jpeg")
    assert client.get("/v1/samples/missing/image").status_code == 404
    analysed = client.post(
        "/v1/analyses/sample", json={"sample_id": "one", "locale": "ru"}, headers=KEY
    )
    assert analysed.status_code == 201
    assert analysed.json()["patient_text"]["locale"] == "ru"
    missing = client.post(
        "/v1/analyses/sample",
        json={"sample_id": "nope", "locale": "en"},
        headers={"Idempotency-Key": "analysis-0004"},
    )
    assert (missing.status_code, code_of(missing)) == (404, ErrorCode.NOT_FOUND)


def test_meta_texts_pass_patient_lint(make_client: Any) -> None:
    client = make_client()
    for locale in ("en", "ru"):
        texts = client.get("/v1/meta", params={"locale": locale}).json()["texts"]
        assert not any(lint(text, locale, "patient") for text in texts.values())
