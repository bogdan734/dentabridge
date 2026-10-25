"""Patient-facing text: i18n templates only, and every string is linted before it leaves."""

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from types import MappingProxyType

from pydantic import TypeAdapter

from dentabridge_core._base import Locale
from dentabridge_core.analysis import NoteStatus, PatientText
from dentabridge_core.findings import ASSESSABLE, Finding, Modality
from dentabridge_core.policy.claims import lint
from dentabridge_core.urgency import UrgencyHint

I18N_DIR = Path(__file__).resolve().parents[1] / "i18n"
MAX_NOTE_CHARS = 600


class ClaimsViolation(ValueError):
    """A string that is about to reach a patient breaks the claims rules."""


def _checked(text: str, locale: Locale) -> str:
    violations = lint(text, locale, "patient")
    if violations:
        first = violations[0]
        raise ClaimsViolation(f"{first.rule_id}: {first.excerpt!r} in {text!r}")
    return text


@cache
def _load(locale: Locale) -> Mapping[str, str]:
    raw = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
    loaded = TypeAdapter(dict[str, str]).validate_python(raw)
    for text in loaded.values():
        _checked(text, locale)
    return MappingProxyType(loaded)


def texts(locale: Locale) -> Mapping[str, str]:
    """Every safety-relevant string for a locale, served by GET /v1/meta."""
    return _load(locale)


def render_patient_text(
    findings: Sequence[Finding],
    urgency: UrgencyHint,
    locale: Locale,
    modality: Modality = "intraoral_photo",
) -> PatientText:
    t = _load(locale)
    observations = tuple(
        t[f"finding.{f.code.value}.{f.status}"] for f in findings if f.status != "not_assessed"
    )
    # Conditions only the other modality can see are noise in this modality's "not assessed" list.
    other_only = {c for m, codes in ASSESSABLE.items() if m != modality for c in codes}
    other_only -= ASSESSABLE[modality]
    names = [
        t[f"finding_name.{f.code.value}"]
        for f in findings
        if f.status == "not_assessed" and f.code not in other_only
    ]
    if urgency.level == "retake_photo":
        not_assessed = t["not_assessed.unusable"]
    elif names:
        not_assessed = f"{t['not_assessed.prefix']} {', '.join(names)}."
    else:
        not_assessed = t["not_assessed.none"]
    return PatientText(
        locale=locale,
        observations=tuple(_checked(o, locale) for o in observations),
        urgency_line=_checked(t[f"urgency.{urgency.level}"], locale),
        see_dentist_line=_checked(t["see_dentist"], locale),
        not_assessed_line=_checked(not_assessed, locale),
        disclaimer=_checked(t["disclaimer"], locale),
    )


def guard_note(note: str | None) -> tuple[str | None, NoteStatus]:
    """Decide whether the model's optional English note may be shown."""
    if note is None or not note.strip():
        return None, "absent"
    if len(note) > MAX_NOTE_CHARS or lint(note, "en", "patient"):
        return None, "rejected"
    return note, "passed"
