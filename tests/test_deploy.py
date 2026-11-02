"""The one-command path (D1): compose profiles, the make targets, and the keys in .env."""

import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "deploy" / "compose.yaml"
SECRETS = ("DENTABRIDGE_LIVEKIT_KEY", "DENTABRIDGE_LIVEKIT_SECRET", "DENTABRIDGE_CALL_SECRET")
PROFILES = {
    "demo": {"api-demo"},
    "mac": {"api", "livekit", "piper"},
    "cpu": {"api", "livekit", "piper", "ollama", "voice"},
    "cuda": {"api", "livekit", "piper", "ollama-cuda", "voice-cuda"},
}


def compose(profile: str, *args: str) -> str:
    if shutil.which("docker") is None:
        pytest.skip("needs docker compose")
    values = {name: f"test-{name.lower()}" for name in SECRETS}
    env = {**os.environ, **values}
    command = ["docker", "compose", "-f", str(COMPOSE), "--profile", profile, *args]
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def env_values(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs = (line.partition("=") for line in lines if "=" in line and not line.startswith("#"))
    return {name: value for name, _, value in pairs}


def tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(example: Path, out: Path) -> None:
    script = ROOT / "tools" / "make_env.py"
    command = [sys.executable, str(script), "--example", str(example), "--out", str(out)]
    subprocess.run(command, check=True, capture_output=True)


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_compose_config_valid_per_profile(profile: str) -> None:
    compose(profile, "config", "-q")
    assert set(compose(profile, "config", "--services").split()) == PROFILES[profile]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_port_is_published_on_localhost_only(profile: str) -> None:
    services: dict[str, Any] = json.loads(compose(profile, "config", "--format", "json"))[
        "services"
    ]
    ports = [port for service in services.values() for port in service.get("ports", [])]
    assert ports
    assert {port.get("host_ip") for port in ports} == {"127.0.0.1"}


def test_no_committed_secrets() -> None:
    """Keys reach the containers only through ${...} from the .env that make writes."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert ".env" not in tracked
    example = env_values(ROOT / ".env.example")
    assert {name: example[name] for name in SECRETS} == dict.fromkeys(SECRETS, "")
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    for service in services.values():
        for name, value in (service.get("environment") or {}).items():
            if "KEY" in name or "SECRET" in name:
                assert re.sub(r"\$\{[^}]+\}", "", str(value)).strip(" :") == "", (name, value)


def test_make_writes_fresh_keys_into_env_and_never_overwrites_it(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
    first, second = tmp_path / "first.env", tmp_path / "second.env"
    make_env(example, first)
    make_env(example, second)
    written = env_values(first)
    assert all(written[name] for name in SECRETS)
    assert len(written["DENTABRIDGE_LIVEKIT_SECRET"]) >= 32  # LiveKit refuses shorter secrets
    assert written["DENTABRIDGE_LIVEKIT_SECRET"] != env_values(second)["DENTABRIDGE_LIVEKIT_SECRET"]
    assert first.stat().st_mode & 0o777 == 0o600
    first.write_text("DENTABRIDGE_LIVEKIT_KEY=in-use\n", encoding="utf-8")
    make_env(example, first)
    assert first.read_text(encoding="utf-8") == "DENTABRIDGE_LIVEKIT_KEY=in-use\n"


def test_the_make_path_names_every_profile() -> None:
    text = (ROOT / "mk" / "deploy.mk").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z][a-z0-9-]*):", text, flags=re.MULTILINE))
    assert {"demo", "mac", "up-cpu", "up-cuda", "down", "models"} <= targets


def test_every_download_shows_its_dated_licence_row() -> None:
    models = tool("models")
    for download in models.DOWNLOADS:
        row = models.licence_row(download.component)
        assert row["status"] == "default"
        assert date.fromisoformat(row["verified_on"])


def test_nothing_is_downloaded_without_a_yes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tool("models")
    ran: list[tuple[str, ...]] = []
    monkeypatch.setattr(models, "VOICES", tmp_path)
    monkeypatch.setattr(models.subprocess, "run", lambda command, check: ran.append(command))
    answers = iter(["", "n", "y", "no"])
    assert models.main([], ask=lambda prompt: next(answers)) == 0
    assert ran == [models.DOWNLOADS[2].command]
    shown = capsys.readouterr().out
    assert "Apache-2.0" in shown
    assert "public domain" in shown


@pytest.mark.slow
def test_demo_smoke() -> None:
    """`make demo` from this checkout: the API in Docker answers its health check."""
    if shutil.which("docker") is None:
        pytest.skip("needs Docker")
    subprocess.run(["make", "demo"], cwd=ROOT, check=True, timeout=1800)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                health = httpx.get("http://127.0.0.1:8000/v1/health", timeout=2)
                if health.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, "no healthy API within 60 s"
            time.sleep(1)
        assert health.json()["mode"] == "demo"
    finally:
        subprocess.run(["make", "down"], cwd=ROOT, check=False, timeout=300)


def book_a_call(client: httpx.Client) -> dict[str, Any]:
    """Quote, hold, book and open the call, as the site does; the dates move with today."""
    from dentabridge_api.testing import CLINIC

    def key() -> dict[str, str]:
        return {"Idempotency-Key": f"deploy-test-{uuid4()}"}

    def ok(response: httpx.Response) -> Any:
        assert response.status_code == 201, response.text
        return response.json()

    today = date.today()
    trip = {
        "depart_date": str(today + timedelta(days=8)),
        "return_date": str(today + timedelta(days=18)),
    }
    quote = ok(
        client.post(
            "/v1/quotes",
            json={
                "clinic_id": CLINIC,
                "treatment_codes": ["crown"],
                "origin_iata": "LHR",
                "travellers": 1,
                "currency": "EUR",
                **trip,
            },
        )
    )
    window = {"from": str(today + timedelta(days=9)), "to": str(today + timedelta(days=12))}
    slots = client.get(f"/v1/clinics/{CLINIC}/slots", params=window).json()
    free = next(s for s in slots if s["status"] == "available")  # earlier runs booked some
    held = ok(client.post("/v1/holds", json={"slot_id": free["slot_id"]}, headers=key()))
    booking = ok(
        client.post(
            "/v1/bookings",
            json={"hold_id": held["hold_id"], "quote_id": quote["quote_id"]},
            headers=key(),
        )
    )
    languages = {"patient_language": "ru", "dentist_language": "en"}
    session = ok(
        client.post(f"/v1/bookings/{booking['booking_id']}/call", json=languages, headers=key())
    )
    tokens = {
        role: ok(
            client.post(
                f"/v1/calls/{session['session_id']}/tokens",
                json={"role": role, "ai_disclosure_acknowledged": True},
            )
        )
        for role in ("patient", "dentist")
    }
    return {"session": session, **tokens}


@pytest.mark.slow
def test_a_call_through_the_mac_profile() -> None:
    """The mac profile with the interpreter run here instead of by `make mac`: the API, LiveKit
    and Piper in Docker, Ollama and whisper native. A call opened through the API carries the
    patient's Russian to the dentist in English."""
    for module in ("livekit.rtc", "faster_whisper", "soundfile"):
        pytest.importorskip(module)
    from dentabridge_voice.adapters.faster_whisper import FasterWhisperSTT
    from dentabridge_voice.adapters.ollama_translator import OllamaTranslator
    from dentabridge_voice.adapters.piper_http import VOICES, PiperHttpTTS
    from dentabridge_voice.rooms import LiveKitServer, watch_rooms
    from dentabridge_voice.testing import Dentist, assert_interpreted, call
    from dentabridge_voice.worker import Services, UtcClock

    try:
        health = httpx.get("http://127.0.0.1:8000/v1/health", timeout=2).json()
        httpx.get("http://localhost:11434/api/version", timeout=2)
    except httpx.HTTPError:
        pytest.skip("needs the mac profile up (make mac) and Ollama")
    if health["mode"] != "live":
        pytest.skip("the API on :8000 runs the demo profile")
    env = env_values(ROOT / ".env")
    server = LiveKitServer(
        "ws://127.0.0.1:7880", env["DENTABRIDGE_LIVEKIT_KEY"], env["DENTABRIDGE_LIVEKIT_SECRET"]
    )
    services = Services(
        stt=FasterWhisperSTT("small", "cpu", "int8"),
        translator=OllamaTranslator("http://localhost:11434", "qwen3:4b-instruct"),
        tts=PiperHttpTTS("http://127.0.0.1:5100", VOICES),
        wall=UtcClock(),
    )
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
        opened = book_a_call(client)
    patient, dentist = opened["patient"], opened["dentist"]
    assert patient["livekit_url"] == env["DENTABRIDGE_LIVEKIT_URL"]

    async def interpreted() -> Dentist:
        worker = asyncio.create_task(watch_rooms(server, services))
        try:
            interpreter = f"interpreter-{opened['session']['session_id']}"
            url = patient["livekit_url"]
            return await call(url, patient["token"], dentist["token"], interpreter, "Болит зуб 36")
        finally:
            worker.cancel()

    assert_interpreted(asyncio.run(interpreted()), "Болит зуб 36")
