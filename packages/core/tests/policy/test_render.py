from itertools import product

import pytest

from dentabridge_core import ASSESSABLE, Finding, FindingCode, Locale
from dentabridge_core.findings import FindingStatus
from dentabridge_core.policy import (
    DEFAULT_TABLE_PATH,
    evaluate,
    guard_note,
    lint,
    load_table,
    render_patient_text,
    texts,
)
from dentabridge_core.testing import sample_findings

STATUSES: tuple[FindingStatus, ...] = ("possible", "not_seen", "not_assessed")
PHOTO_CODES = sorted(ASSESSABLE["intraoral_photo"])
TABLE = load_table(DEFAULT_TABLE_PATH)


def _all_findings(combo: tuple[FindingStatus, ...]) -> list[Finding]:
    chosen = dict(zip(PHOTO_CODES, combo, strict=True))
    return [Finding(code=code, status=chosen.get(code, "not_assessed")) for code in FindingCode]


def test_keys_identical_across_locales() -> None:
    assert texts("en").keys() == texts("ru").keys()


@pytest.mark.parametrize("locale", ["en", "ru"])
def test_every_text_passes_patient_lint(locale: Locale) -> None:
    violations = {key: lint(text, locale, "patient") for key, text in texts(locale).items()}
    assert {key: found for key, found in violations.items() if found} == {}


@pytest.mark.parametrize("locale", ["en", "ru"])
def test_every_rendering_passes_patient_lint_and_names_a_dentist(locale: Locale) -> None:
    see_dentist = texts(locale)["see_dentist"]
    for combo in product(STATUSES, repeat=5):
        findings = _all_findings(combo)
        urgency = evaluate(findings, "ok", TABLE)
        text = render_patient_text(findings, urgency, locale)
        assert text.locale == locale
        assert text.see_dentist_line == see_dentist
        strings = [*text.observations, text.urgency_line, text.not_assessed_line, text.disclaimer]
        assert all(s and not lint(s, locale, "patient") for s in strings)


@pytest.mark.parametrize("locale", ["en", "ru"])
def test_possible_and_not_seen_become_observations(locale: Locale) -> None:
    findings = list(sample_findings({FindingCode.CARIES}))
    text = render_patient_text(findings, evaluate(findings, "ok", TABLE), locale)
    assert len(text.observations) == 5
    assert text.observations[0] == texts(locale)["finding.caries.possible"]
    assert text.urgency_line == texts(locale)["urgency.soon"]


def test_unusable_photo_text() -> None:
    nothing = [Finding(code=code, status="not_assessed") for code in FindingCode]
    text = render_patient_text(nothing, evaluate(nothing, "unusable", TABLE), "en")
    assert text.observations == ()
    assert text.urgency_line == texts("en")["urgency.retake_photo"]
    assert text.not_assessed_line == texts("en")["not_assessed.unusable"]


def test_not_assessed_line_lists_only_codes_relevant_to_the_modality() -> None:
    nothing_seen = _all_findings(("not_seen",) * 5)
    urgency = evaluate(nothing_seen, "ok", TABLE)
    photo = render_patient_text(nothing_seen, urgency, "en", "intraoral_photo")
    xray = render_patient_text(nothing_seen, urgency, "en", "panoramic_xray")
    assert "tartar" in photo.not_assessed_line
    assert "root-tip" not in photo.not_assessed_line
    assert "root-tip" in xray.not_assessed_line


def test_note_with_verdict_rejected() -> None:
    assert guard_note("You have caries on the left side.") == (None, "rejected")
    assert guard_note("This tool diagnoses disease.") == (None, "rejected")
    assert guard_note("x" * 601) == (None, "rejected")
    assert guard_note("A darker area near the upper molar may be decay.") == (
        "A darker area near the upper molar may be decay.",
        "passed",
    )
    assert guard_note(None) == (None, "absent")
    assert guard_note("   ") == (None, "absent")
