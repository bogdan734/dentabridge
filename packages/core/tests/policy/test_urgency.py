import hashlib
import json
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from dentabridge_core import ASSESSABLE, Finding, FindingCode
from dentabridge_core.analysis import ImageQuality
from dentabridge_core.findings import FindingStatus
from dentabridge_core.policy import DEFAULT_TABLE_PATH, evaluate, load_table, review_status

STATUSES: tuple[FindingStatus, ...] = ("possible", "not_seen", "not_assessed")
QUALITIES: tuple[ImageQuality, ...] = ("ok", "unusable")
PHOTO_CODES = sorted(ASSESSABLE["intraoral_photo"])
LEVELS = {"routine": 0, "soon": 1, "prompt": 2}
ENABLED_RULES = ("R0", "R2", "R3", "R9")


def _findings(combo: tuple[FindingStatus, ...]) -> list[Finding]:
    pairs = zip(PHOTO_CODES, combo, strict=True)
    return [Finding(code=code, status=status) for code, status in pairs]


def _record(sha: str, verdict: str = "agree", rules: tuple[str, ...] = ENABLED_RULES) -> str:
    return json.dumps(
        {
            "reviewer_name": "Dr Example",
            "qualification": "BDS",
            "registration": "000000",
            "reviewed_on": "2026-11-20",
            "table_sha256": sha,
            "verdicts": {rule: {"verdict": verdict} for rule in rules},
            "statement": "I reviewed this table.",
        }
    )


def test_every_status_combination_hits_one_rule() -> None:
    table = load_table(DEFAULT_TABLE_PATH)
    enabled = {r.id for r in table.rules if r.enabled}
    count = 0
    for quality in QUALITIES:
        for combo in product(STATUSES, repeat=5):
            hint = evaluate(_findings(combo), quality, table)
            count += 1
            assert hint.rule_id in enabled
            assert hint.table_version == "v1"
            if quality == "unusable":
                assert (hint.rule_id, hint.level) == ("R0", "retake_photo")
                continue
            by_code = dict(zip(PHOTO_CODES, combo, strict=True))
            caries = by_code[FindingCode.CARIES] == "possible"
            gums = by_code[FindingCode.GINGIVITIS] == "possible"
            assert hint.rule_id == ("R2" if caries else "R3" if gums else "R9")
    assert count == 486


def test_adding_possible_never_lowers_level() -> None:
    table = load_table(DEFAULT_TABLE_PATH)
    for combo in product(STATUSES, repeat=5):
        before = LEVELS[evaluate(_findings(combo), "ok", table).level]
        for i, status in enumerate(combo):
            if status == "possible":
                continue
            raised: tuple[FindingStatus, ...] = (*combo[:i], "possible", *combo[i + 1 :])
            assert LEVELS[evaluate(_findings(raised), "ok", table).level] >= before


def test_triggered_by_names_the_matching_code() -> None:
    table = load_table(DEFAULT_TABLE_PATH)
    hint = evaluate([Finding(code=FindingCode.CARIES, status="possible")], "ok", table)
    assert hint.triggered_by == (FindingCode.CARIES,)
    assert evaluate([], "ok", table).triggered_by == ()


def test_disabled_rule_never_fires() -> None:
    table = load_table(DEFAULT_TABLE_PATH)
    assert not next(r for r in table.rules if r.id == "R1").enabled
    caries = Finding(code=FindingCode.CARIES, status="possible", subtype="caries_5")
    assert evaluate([caries], "ok", table).rule_id == "R2"


def test_illustrative_by_default(tmp_path: Path) -> None:
    table = load_table(DEFAULT_TABLE_PATH)
    assert table.review_status == "illustrative_not_clinically_reviewed"
    assert evaluate([], "ok", table).review_status == "illustrative_not_clinically_reviewed"
    assert review_status(DEFAULT_TABLE_PATH, tmp_path / "missing") == table.review_status
    assert review_status(DEFAULT_TABLE_PATH, tmp_path) == table.review_status


def test_reviewed_only_when_sha_matches(tmp_path: Path) -> None:
    table = tmp_path / "table.toml"
    table.write_bytes(DEFAULT_TABLE_PATH.read_bytes())
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    (reviews / "dr-example.json").write_text(_record(sha))
    assert review_status(table, reviews) == "reviewed"
    assert load_table(table, reviews).review_status == "reviewed"
    assert evaluate([], "ok", load_table(table, reviews)).review_status == "reviewed"

    table.write_text(table.read_text() + "\n# edited after the review\n")
    assert review_status(table, reviews) == "illustrative_not_clinically_reviewed"


@pytest.mark.parametrize(
    ("verdict", "rules"),
    [("change", ENABLED_RULES), ("remove", ENABLED_RULES), ("agree", ("R0", "R2"))],
)
def test_partial_or_critical_review_stays_illustrative(
    tmp_path: Path, verdict: str, rules: tuple[str, ...]
) -> None:
    sha = hashlib.sha256(DEFAULT_TABLE_PATH.read_bytes()).hexdigest()
    (tmp_path / "r.json").write_text(_record(sha, verdict, rules))
    assert review_status(DEFAULT_TABLE_PATH, tmp_path) == "illustrative_not_clinically_reviewed"


def test_malformed_review_record_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text('{"reviewer_name": "x"}')
    with pytest.raises(ValidationError):
        review_status(DEFAULT_TABLE_PATH, tmp_path)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (("[[rule]]", 'id = "A"', 'level = "soon"', 'code = "caries"'), "match everything"),
        (
            ("[[rule]]", 'id = "A"', 'level = "soon"', "[[rule]]", 'id = "A"', 'level = "routine"'),
            "unique",
        ),
        (("[[rule]]", 'id = "A"', 'level = "soon"', 'status = "possible"'), "need a code"),
    ],
)
def test_invalid_tables_are_rejected(tmp_path: Path, lines: tuple[str, ...], message: str) -> None:
    path = tmp_path / "t.toml"
    path.write_text("\n".join(['version = "v"', *lines]) + "\n")
    with pytest.raises(ValueError, match=message):
        load_table(path)
