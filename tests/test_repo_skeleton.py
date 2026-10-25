import importlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_ignore(path: str) -> int:
    # git exits 0 when the path is ignored, 1 when it is not, 128 on a git error
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False)
    return result.returncode


@pytest.mark.parametrize(
    "name",
    [
        "dentabridge_core",
        "dentabridge_ml",
        "dentabridge_api",
        "dentabridge_pricing",
        "dentabridge_voice",
    ],
)
def test_every_package_imports(name: str) -> None:
    assert importlib.import_module(name).__name__ == name


@pytest.mark.parametrize("path", ["data/x", "private/x", "var/x", "a.gguf", ".env"])
def test_data_dirs_ignored(path: str) -> None:
    assert _check_ignore(path) == 0


# data/ is also a code package (ML2) and a snapshot dir (P1, P2): only the root one is local
@pytest.mark.parametrize(
    "path",
    ["packages/ml/src/dentabridge_ml/data/sources.py", "packages/pricing/data/fx.json"],
)
def test_source_data_dirs_not_ignored(path: str) -> None:
    assert _check_ignore(path) == 1
