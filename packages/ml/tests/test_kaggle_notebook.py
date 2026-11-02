import ast
import json
import re
import types
from pathlib import Path

import pytest

from dentabridge_ml.train.config import CONFIG_DIR

REPO = Path(__file__).resolve().parents[3]
NOTEBOOK = REPO / "packages" / "ml" / "notebooks" / "kaggle_lean.ipynb"
SCRIPT = REPO / "scripts" / "kaggle" / "run.sh"


def _stages() -> set[str]:
    return set(re.findall(r"^stage_(\w+)\(\)", SCRIPT.read_text(), flags=re.MULTILINE))


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _lanes() -> dict[str, dict[str, object]]:
    cell = next(src for src in _code_cells() if src.startswith("LANES = "))
    lanes: dict[str, dict[str, object]] = ast.literal_eval(cell.removeprefix("LANES = "))
    return lanes


def test_every_stage_the_notebook_calls_exists_in_the_script() -> None:
    called = {s for src in _code_cells() for s in re.findall(r'run_stage\("(\w+)"\)', src)}
    called |= {stage for lane in _lanes().values() for stage in lane["stages"]}  # type: ignore[attr-defined]
    assert called, "the notebook calls no stage"
    assert called <= _stages()


def test_each_lane_has_its_own_gpu_and_a_shipped_config() -> None:
    lanes = _lanes()
    assert sorted(int(lane["gpu"]) for lane in lanes.values()) == [0, 1]  # type: ignore[call-overload]
    for lane in lanes.values():
        assert (CONFIG_DIR / f"{lane['config']}.toml").exists()
    assert len({lane["config"] for lane in lanes.values()}) == len(lanes), (
        "lanes must not share a run"
    )


def test_the_script_dispatches_exactly_its_stages() -> None:
    text = SCRIPT.read_text()
    dispatch = re.search(r"case \"\$stage\" in\n\s+([\w|]+\)) \"stage_", text)
    assert dispatch is not None
    assert set(dispatch.group(1).rstrip(")").split("|")) == _stages()


def test_scripts_are_executable_and_the_bundle_builds_from_a_clean_commit_only() -> None:
    for name in ("run.sh", "bundle.sh", "push.sh"):
        assert (REPO / "scripts" / "kaggle" / name).stat().st_mode & 0o111
    assert "commit first" in (REPO / "scripts" / "kaggle" / "bundle.sh").read_text()


def test_the_default_stages_only_smoke_test() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    line = next(src for src in sources if src.lstrip().startswith("STAGES"))
    assert re.search(r'STAGES = \["setup", "data", "smoke"\]', line), "a first push must not train"


Call = tuple[str, str, str, str]  # lane tag, stage, GPU, config


def _run_lane_cell(
    nvidia_smi: str, calls: list[Call], failing: tuple[str, str] | None = None
) -> None:
    """Execute the notebook's lane cell against stubs, appending one entry per stage run."""

    def run_stage(name: str, tag: str = "", **env: str) -> None:
        calls.append((tag.strip(" []"), name, env["GPU"], env["CONFIG"]))
        if failing == (tag.strip(" []"), name):
            raise RuntimeError(f"stage {name} failed")

    fake_subprocess = types.SimpleNamespace(
        run=lambda *args, **kwargs: types.SimpleNamespace(stdout=nvidia_smi)
    )
    cells = _code_cells()
    namespace = {"run_stage": run_stage, "subprocess": fake_subprocess}
    for src in (
        next(c for c in cells if c.startswith("LANES = ")),
        next(c for c in cells if c.startswith("import threading")),
    ):
        exec(src, namespace)  # the notebook cell under test, with stubs


TWO_T4 = "GPU 0: Tesla T4 (UUID: a)\nGPU 1: Tesla T4 (UUID: b)\n"


def test_two_gpus_run_both_lanes_each_on_its_own_gpu_and_config() -> None:
    calls: list[Call] = []
    _run_lane_cell(TWO_T4, calls)
    assert [c[1:] for c in calls if c[0] == "A"] == [
        ("train", "0", "lean-t4"),
        ("merge", "0", "lean-t4"),
        ("eval", "0", "lean-t4"),
        ("zeroshot", "0", "lean-t4"),
    ]
    assert [c[1:] for c in calls if c[0] == "B"] == [
        ("train", "1", "lean-t4-vision"),
        ("merge", "1", "lean-t4-vision"),
        ("eval", "1", "lean-t4-vision"),
    ]


def test_one_gpu_runs_lane_a_only() -> None:
    calls: list[Call] = []
    _run_lane_cell("GPU 0: Tesla T4 (UUID: a)\n", calls)
    assert {c[0] for c in calls} == {"A"}


def test_a_failing_lane_stops_itself_not_the_other_and_fails_the_notebook() -> None:
    calls: list[Call] = []
    with pytest.raises(RuntimeError, match="A: stage train failed"):
        _run_lane_cell(TWO_T4, calls, failing=("A", "train"))
    assert [c[1] for c in calls if c[0] == "A"] == ["train"]
    assert [c[1] for c in calls if c[0] == "B"] == ["train", "merge", "eval"]
