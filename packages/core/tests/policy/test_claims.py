from pathlib import Path

import pytest

from dentabridge_core import Locale
from dentabridge_core.policy import disclaimers, lint
from dentabridge_core.policy.lint_cli import main

CORPUS = Path(__file__).parent / "corpus"


def _lines(name: str) -> list[str]:
    return [ln for ln in (CORPUS / name).read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.parametrize("locale", ["en", "ru"])
def test_fail_corpus_every_line_flagged(locale: Locale) -> None:
    lines = _lines(f"{locale}_fail.txt")
    assert len(lines) >= 12
    assert [ln for ln in lines if not lint(ln, locale, "patient")] == []


@pytest.mark.parametrize("locale", ["en", "ru"])
def test_pass_corpus_clean(locale: Locale) -> None:
    lines = _lines(f"{locale}_pass.txt")
    assert len(lines) >= 10
    assert {ln: lint(ln, locale, "patient") for ln in lines if lint(ln, locale, "patient")} == {}


@pytest.mark.parametrize("locale", ["en", "ru"])
def test_disclaimer_allowed_only_in_docs_profile(locale: Locale) -> None:
    sentences = disclaimers(locale)
    assert len(sentences) >= 5
    for sentence in sentences:
        assert lint(sentence, locale, "patient"), sentence
        assert lint(sentence, locale, "docs") == (), sentence


def test_docs_profile_still_flags_everything_else() -> None:
    allowed = disclaimers("en")[0]
    text = f"{allowed} It guarantees results. You have caries."
    assert {v.rule_id for v in lint(text, "en", "docs")} == {"claim.guarantee", "claim.verdict"}


def test_hedge_decides_verdicts_sentence_by_sentence() -> None:
    assert lint("You have a cavity.", "en", "patient")[0].rule_id == "claim.verdict"
    assert lint("You may have a cavity.", "en", "patient") == ()
    text = "You may have a cavity. You have gingivitis."
    flagged = lint(text, "en", "patient")
    assert [text[v.start : v.end] for v in flagged] == ["You have gingivitis"]


def test_offsets_point_at_the_excerpt() -> None:
    text = "Intro. We guarantee it, and we diagnose."
    violations = lint(text, "en", "patient")
    assert [v.rule_id for v in violations] == ["claim.guarantee", "claim.diagnosis"]
    assert all(text[v.start : v.end] == v.excerpt for v in violations)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = tmp_path / "clean.json"
    clean.write_text('{"a": "Please see a dentist.", "b": ["Prices are illustrative."]}')
    dirty = tmp_path / "dirty.json"
    dirty.write_text('{"a": {"b": "We guarantee results."}}')
    markdown = tmp_path / "doc.md"
    markdown.write_text("Fine line.\n```\nwe diagnose here\n```\nWe diagnose disease.\n")

    assert main(["--profile", "patient", "--locale", "en", str(clean)]) == 0
    assert capsys.readouterr().out == ""
    assert main(["--profile", "patient", "--locale", "en", str(clean), str(dirty)]) == 1
    assert "dirty.json:$.a.b: claim.guarantee" in capsys.readouterr().out
    assert main(["--profile", "patient", "--locale", "en", str(markdown)]) == 1
    out = capsys.readouterr().out
    assert "doc.md:5: claim.diagnosis" in out
    assert "doc.md:3" not in out
