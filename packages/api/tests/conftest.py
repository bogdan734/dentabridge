import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dentabridge_api.app import create_app
from dentabridge_api.clock import FrozenClock
from dentabridge_api.settings import Settings
from dentabridge_api.testing import ScriptedVision, jpeg_bytes
from dentabridge_core import VisionBackend
from dentabridge_core.testing import FakeFareEstimator

NOW = datetime(2026, 10, 25, 12, tzinfo=UTC)


@pytest.fixture
def samples_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "samples"
    folder.mkdir()
    (folder / "one.jpg").write_bytes(jpeg_bytes((200, 120)))
    (folder / "samples.json").write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "file": "one.jpg",
                    "modality": "intraoral_photo",
                    "attribution": "A. Author, Sample one, https://example.test, CC BY 4.0",
                    "licence": "CC BY 4.0",
                }
            ]
        )
    )
    return folder


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(NOW)


@pytest.fixture
def make_client(samples_dir: Path, clock: FrozenClock):  # type: ignore[no-untyped-def]
    def build(vision: VisionBackend | None = None, **settings: object) -> TestClient:
        config = Settings(db_url="sqlite://", samples_dir=samples_dir, **settings)  # type: ignore[arg-type]
        app = create_app(config, clock, vision or ScriptedVision(), FakeFareEstimator())
        return TestClient(app, raise_server_exceptions=False)

    return build
