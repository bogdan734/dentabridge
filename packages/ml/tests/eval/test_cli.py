import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dentabridge_core import Finding, FindingCode, ModelInfo, VisionOutput
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.eval import __main__ as cli
from dentabridge_ml.eval.results import CodeResult, SplitResult, render_markdown
from dentabridge_ml.splits.manifest import SplitEntry, SplitManifest, write_manifests

CARIES = FindingCode.CARIES


class FixedBackend:
    model = ModelInfo(model_id="fixed", version="0", backend="fixture")

    async def analyse(self, image: Any, modality: Any) -> VisionOutput:
        finding = Finding(code=CARIES, status="not_seen")
        return VisionOutput(image_quality="ok", findings=(finding,), latency_ms=1.0)


def test_markdown_shows_intervals_and_dashes_for_what_cannot_be_said() -> None:
    result = SplitResult(
        model_id="m",
        split="val",
        n_images=40,
        json_valid_rate=0.975,
        codes=(
            CodeResult(
                code=CARIES,
                n_pos=10,
                n_neg=30,
                sens=0.7,
                sens_ci=(0.4, 0.9),
                spec=None,
                ece=0.05,
            ),
            CodeResult(code=FindingCode.FILLING, n_pos=0, n_neg=0),
        ),
    )
    lines = render_markdown(result).splitlines()
    assert lines[0] == "# m on val"
    assert "40 images, 97.5 % of answers were valid JSON" in lines[2]
    assert "| caries | 10 | 30 | 0.70 (0.40-0.90) | — | 0.05 | — |" in lines
    assert "| filling | 0 | 0 | — | — | — | — |" in lines


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    records = []
    for i in range(6):
        Image.new("RGB", (64, 48), (i * 30, 20, 20)).save(tmp_path / f"{i}.jpg")
        records.append(
            ImageRecord(
                record_id=f"aku_caries/{i}",
                source="aku_caries",
                path=f"{i}.jpg",
                width=64,
                height=48,
                boxes=(LabelBox(code=CARIES, x0=0.1, y0=0.1, x1=0.4, y1=0.4),) if i < 3 else (),
            )
        )
    (tmp_path / "aku_caries").symlink_to(tmp_path)
    entries = tuple(
        SplitEntry(
            record_id=r.record_id,
            source="aku_caries",
            split="val" if i < 4 else "test",
            group=f"g{i}",
            group_method="cluster",
        )
        for i, r in enumerate(records)
    )
    write_manifests(SplitManifest(seed=0, ratios=(0.4, 0.3, 0.3), entries=entries), tmp_path / "m")
    monkeypatch.setattr(cli, "load_records", lambda source, root: records)
    monkeypatch.setattr(cli, "load_vision_backend", lambda spec, release, px: FixedBackend())
    return tmp_path


def _run(world: Path, *extra: str, split: str = "val") -> int:
    args = ["run", "--backend", "ollama:x", "--split", split, "--out-dir", str(world / "out")]
    args += ["--manifests", str(world / "m"), "--root", str(world), *extra]
    return cli.main(args)


def test_run_writes_predictions_result_and_table(world: Path) -> None:
    assert _run(world) == 0
    out = world / "out"
    rows = [json.loads(line) for line in (out / "val.predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert SplitResult.read(out / "val.json").n_images == 4
    assert (out / "val.md").read_text().startswith("# fixed on val")
    assert "# fixed on val [source=aku_caries]" in (out / "val.by-source.md").read_text()
    assert "[view=unknown]" in (out / "val.by-view.md").read_text()


def test_limit_keeps_a_smoke_run_small_and_spread_over_the_split(world: Path) -> None:
    _run(world, "--limit", "2")
    assert SplitResult.read(world / "out" / "val.json").n_images == 2
    lines = (world / "out" / "val.predictions.jsonl").read_text().splitlines()
    ids = [json.loads(line)["record_id"] for line in lines]
    assert ids == ["aku_caries/0", "aku_caries/2"], "every other val image, not the first two"


def test_the_test_split_needs_final_and_a_log_and_never_a_limit(world: Path) -> None:
    with pytest.raises(PermissionError, match="final=True"):
        _run(world, split="test")
    with pytest.raises(SystemExit):
        _run(world, "--final", "--limit", "1", split="test")
    log = world / "access.log"
    assert _run(world, "--final", "--access-log", str(log), split="test") == 0
    assert "split=test" in log.read_text()
