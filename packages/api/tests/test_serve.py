import asyncio
import json
from pathlib import Path

import pytest

from dentabridge_api.__main__ import DEMO_NOW, build_app, main
from dentabridge_api.clock import SystemClock
from dentabridge_api.demo.fixture_vision import PRECOMPUTED, FixtureVisionBackend
from dentabridge_api.samples import SampleCatalogue
from dentabridge_api.settings import Settings
from dentabridge_ml.serving.ollama import OllamaVisionBackend


def no_samples(folder: Path, precomputed: bool = True) -> Path:
    folder.mkdir(exist_ok=True)
    (folder / "samples.json").write_text("[]")
    if precomputed:
        model = {"model_id": "none", "version": "0", "backend": "fixture"}
        (folder / "precomputed.json").write_text(json.dumps({"model": model, "answers": {}}))
    return folder


def test_demo_mode_replays_the_samples_on_the_frozen_demo_clock(tmp_path: Path) -> None:
    app = build_app(Settings(db_url="sqlite://", samples_dir=no_samples(tmp_path / "samples")))
    assert isinstance(app.state.vision, FixtureVisionBackend)
    assert app.state.clock.now() == DEMO_NOW


def test_demo_mode_without_precomputed_answers_fails_at_startup(tmp_path: Path) -> None:
    samples = no_samples(tmp_path / "samples", precomputed=False)
    with pytest.raises(FileNotFoundError, match="precomputed"):
        build_app(Settings(db_url="sqlite://", samples_dir=samples))


def test_live_mode_runs_the_named_model_under_the_release_config(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    release.write_text('{"assessed": []}')
    settings = Settings(
        mode="live",
        db_url="sqlite://",
        release_config=release,
        vision_backend="ollama:dentabridge-tuned",
    )
    app = build_app(settings)
    assert isinstance(app.state.vision, OllamaVisionBackend)
    assert app.state.vision.model.model_id == "dentabridge-tuned"
    assert isinstance(app.state.clock, SystemClock)


def test_live_mode_without_a_release_config_fails_at_startup(tmp_path: Path) -> None:
    settings = Settings(mode="live", db_url="sqlite://", release_config=tmp_path / "none.json")
    with pytest.raises(FileNotFoundError, match=r"none\.json"):
        build_app(settings)


def test_serve_runs_uvicorn_on_the_given_address(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    served: list[tuple[str, int]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: served.append((host, port)))
    monkeypatch.setenv("DENTABRIDGE_DB_URL", "sqlite://")
    monkeypatch.setenv("DENTABRIDGE_SAMPLES_DIR", str(no_samples(tmp_path / "samples")))
    assert main(["serve", "--host", "0.0.0.0", "--port", "8080"]) == 0
    assert served == [("0.0.0.0", 8080)]


def test_a_developers_env_file_does_not_reach_the_tests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("DENTABRIDGE_MODE=live\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().mode == "demo"


def test_every_bundled_sample_has_a_precomputed_answer_and_nothing_else_does() -> None:
    """Until samples are bundled the answers are empty and no model is named; once they are,
    `make precompute-samples` must have run with the shipped model."""
    folder = Settings().samples_dir
    catalogue = SampleCatalogue(folder).all()
    backend = FixtureVisionBackend(folder)
    for sample in catalogue:
        asyncio.run(backend.analyse(sample.sanitised(), sample.modality))
    answers = json.loads((folder / PRECOMPUTED).read_text(encoding="utf-8"))["answers"]
    assert len(answers) == len(catalogue)
    assert (backend.model.model_id == "none") == (not catalogue)
