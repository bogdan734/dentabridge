import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOTEBOOK = REPO / "packages" / "ml" / "notebooks" / "kaggle_lean.ipynb"
SCRIPT = REPO / "scripts" / "kaggle" / "run.sh"


def _stages() -> set[str]:
    return set(re.findall(r"^stage_(\w+)\(\)", SCRIPT.read_text(), flags=re.MULTILINE))


def test_every_stage_the_notebook_calls_exists_in_the_script() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    called = {
        stage
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for stage in re.findall(r"kaggle/run\.sh (\w+)", "".join(cell["source"]))
    }
    assert called, "the notebook calls no stage"
    assert called <= _stages()


def test_the_script_dispatches_exactly_its_stages() -> None:
    text = SCRIPT.read_text()
    dispatch = re.search(r"case \"\$stage\" in\n\s+([\w|]+\)) \"stage_", text)
    assert dispatch is not None
    assert set(dispatch.group(1).rstrip(")").split("|")) == _stages()


def test_scripts_are_executable_and_the_bundle_builds_from_a_clean_commit_only() -> None:
    for name in ("run.sh", "bundle.sh"):
        assert (REPO / "scripts" / "kaggle" / name).stat().st_mode & 0o111
    assert "commit first" in (REPO / "scripts" / "kaggle" / "bundle.sh").read_text()
