import importlib
import re
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


@pytest.mark.parametrize(
    "path",
    [
        "data/x",
        "private/x",
        "var/x",
        "a.gguf",
        ".env",
        ".env.local",
        "a.bin",
        "b.ckpt",
        "checkpoint-100/x",
    ],
)
def test_data_dirs_ignored(path: str) -> None:
    assert _check_ignore(path) == 0


# data/ is also a code package (ML2) and a snapshot dir (P1, P2): only the root one is local.
# .env.example is committed by WS04.
@pytest.mark.parametrize(
    "path",
    [
        "packages/ml/src/dentabridge_ml/data/sources.py",
        "packages/pricing/data/fx.json",
        ".env.example",
    ],
)
def test_source_data_dirs_not_ignored(path: str) -> None:
    assert _check_ignore(path) == 1


def test_make_targets_unique_across_mk_files() -> None:
    seen: dict[str, str] = {}
    for mk in sorted((REPO_ROOT / "mk").glob("*.mk")):
        for line in mk.read_text().splitlines():
            match = re.match(r"^([A-Za-z0-9_-]+):(?!=)", line)
            if match is None:
                continue
            target = match.group(1)
            assert target not in seen, f"{target} is defined in {seen[target]} and {mk.name}"
            seen[target] = mk.name
