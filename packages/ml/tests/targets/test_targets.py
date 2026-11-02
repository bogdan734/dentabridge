import json

import numpy as np
import pytest
from PIL import Image

from dentabridge_core import FindingCode, VisionOutput
from dentabridge_ml.data.records import ImageRecord, LabelBox
from dentabridge_ml.serving.parse import ReleaseConfig, parse_output
from dentabridge_ml.targets.degrade import DEGRADE_KINDS, REASON, degrade
from dentabridge_ml.targets.prompt import build_prompt
from dentabridge_ml.targets.sft import MAX_REGIONS, target_json, unusable_json

CARIES, FILLING = FindingCode.CARIES, FindingCode.FILLING
RELEASE = ReleaseConfig(assessed=frozenset({CARIES, FILLING}))


def _record(*boxes: LabelBox, source: str = "alphadent") -> ImageRecord:
    return ImageRecord.model_validate(
        {
            "record_id": f"{source}/a",
            "source": source,
            "path": "a.jpg",
            "width": 100,
            "height": 80,
            "boxes": boxes,
        }
    )


def _box(code: FindingCode, x0: float = 0.1234567) -> LabelBox:
    return LabelBox(code=code, x0=x0, y0=0.1, x1=0.5, y1=0.4, subtype="caries_3")


def _picture() -> Image.Image:
    pixels = np.random.default_rng(7).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def _parse(answer: str) -> VisionOutput:
    return parse_output(answer, None, RELEASE)  # the strict parser serving uses, not a lenient one


def test_target_limited_to_source_coverage() -> None:
    aku = _record(source="aku_caries")
    assert target_json(aku, frozenset({CARIES}))
    with pytest.raises(ValueError, match="no labels for"):
        target_json(aku, frozenset({CARIES, FILLING}))


def test_target_parses_as_vision_output() -> None:
    answer = target_json(_record(_box(CARIES)), frozenset({CARIES, FILLING}))
    output = _parse(answer)
    assert output.image_quality == "ok"
    assert {f.code: f.status for f in output.findings} == {CARIES: "possible", FILLING: "not_seen"}
    (region,) = output.findings[0].regions
    assert (region.x0, region.y0, region.x1, region.y1) == (0.123, 0.1, 0.5, 0.4)


def test_possible_carries_rounded_boxes_and_not_seen_none() -> None:
    answer = json.loads(target_json(_record(_box(CARIES)), frozenset({CARIES, FILLING})))
    caries, filling = answer["findings"]
    assert caries["regions"] == [[123, 100, 500, 400]], "integers 0-1000, the scale Qwen grounds in"
    assert filling == {"code": "filling", "status": "not_seen", "regions": []}


def test_a_box_thinner_than_a_thousandth_is_dropped_not_written_degenerate() -> None:
    sliver = LabelBox(code=CARIES, x0=0.3001, y0=0.2, x1=0.3004, y1=0.4)
    answer = json.loads(target_json(_record(sliver, _box(CARIES)), frozenset({CARIES})))
    assert answer["findings"][0]["regions"] == [[123, 100, 500, 400]]


def test_regions_are_capped_to_the_largest_boxes() -> None:
    # box i is i * 0.05 wide, so the three smallest (i = 1..3) must be the ones dropped
    boxes = [
        LabelBox(code=CARIES, x0=0.0, y0=0.05 * (10 - i), x1=0.05 * i, y1=0.05 * (10 - i) + 0.04)
        for i in range(1, MAX_REGIONS + 4)
    ]
    answer = json.loads(target_json(_record(*boxes), frozenset({CARIES})))
    regions = answer["findings"][0]["regions"]
    assert len(regions) == MAX_REGIONS
    assert min(r[2] for r in regions) == 200  # the narrowest kept box is i = 4
    assert [r[1] for r in regions] == sorted(r[1] for r in regions)


def test_findings_follow_catalogue_order_whatever_the_set_order() -> None:
    record = _record()
    first = target_json(record, frozenset({FindingCode.CROWN, CARIES, FILLING}))
    second = target_json(record, frozenset([FILLING, CARIES, FindingCode.CROWN]))
    assert first == second
    assert [f["code"] for f in json.loads(first)["findings"]] == ["caries", "filling", "crown"]


def test_image_level_label_counts_as_present() -> None:
    gums = ImageRecord(
        record_id="mendeley_gingivitis/g",
        source="mendeley_gingivitis",
        path="g.jpg",
        width=10,
        height=10,
        image_labels=frozenset({FindingCode.GINGIVITIS}),
    )
    findings = json.loads(target_json(gums, frozenset({FindingCode.GINGIVITIS})))["findings"]
    assert findings == [{"code": "gingivitis", "status": "possible", "regions": []}]


@pytest.mark.parametrize("reason", ["blurry", "dark", "no_teeth", "not_dental"])
def test_unusable_answer_parses(reason: str) -> None:
    output = _parse(unusable_json(reason))  # type: ignore[arg-type]
    assert (output.image_quality, output.quality_reason, output.findings) == (
        "unusable",
        reason,
        (),
    )


def test_prompt_lists_codes_in_catalogue_order_and_needs_one() -> None:
    prompt = build_prompt(frozenset({FindingCode.GINGIVITIS, CARIES}), "intraoral_photo")
    assert "Judge only these findings: caries, gingivitis." in prompt
    assert prompt == build_prompt(frozenset({CARIES, FindingCode.GINGIVITIS}), "intraoral_photo")
    assert "X-ray" in build_prompt(frozenset({CARIES}), "panoramic_xray")
    with pytest.raises(ValueError, match="at least one"):
        build_prompt(frozenset(), "intraoral_photo")


def _mean(img: Image.Image) -> float:
    return float(np.asarray(img).mean())


def _edge_energy(img: Image.Image) -> float:
    return float(np.abs(np.diff(np.asarray(img.convert("L"), dtype=float), axis=1)).mean())


@pytest.mark.parametrize("kind", DEGRADE_KINDS)
def test_degrade_deterministic_and_keeps_size(kind: str) -> None:
    picture = _picture()
    first = degrade(picture, kind, seed=3)  # type: ignore[arg-type]
    second = degrade(picture, kind, seed=3)  # type: ignore[arg-type]
    assert first.size == picture.size
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != degrade(picture, kind, seed=4).tobytes()  # type: ignore[arg-type]


def test_degrade_kinds_do_what_their_reason_says() -> None:
    picture = _picture()
    assert _edge_energy(degrade(picture, "blur", 1)) < _edge_energy(picture) / 3
    assert _mean(degrade(picture, "dark", 1)) < _mean(picture) / 5
    assert _mean(degrade(picture, "overexposed", 1)) > _mean(picture) * 0.9
    assert _edge_energy(degrade(picture, "crop_no_teeth", 1)) < _edge_energy(picture) / 2
    assert set(REASON) == set(DEGRADE_KINDS)


def test_the_prompt_asks_for_integer_boxes_the_parser_reads() -> None:
    prompt = build_prompt(frozenset({CARIES}), "intraoral_photo")
    assert '"regions": [[x0, y0, x1, y1]]' in prompt
    assert "integers from 0 to 1000" in prompt
    assert "fractions" not in prompt
