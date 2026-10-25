"""Claims linter: no diagnostic, regulatory or verdict wording in user-facing text."""

import re
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dentabridge_core._base import Locale

Profile = Literal["patient", "docs"]

RULES_PATH = Path(__file__).with_name("claims_rules.toml")
_SENTENCE = re.compile(r"[^.!?…\n]+[.!?…]?")


@dataclass(frozen=True)
class Violation:
    rule_id: str
    start: int
    end: int
    excerpt: str


class _Banned(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    pattern: str


class _LocaleRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hedge: str
    verdict: tuple[str, ...]
    banned: tuple[_Banned, ...]
    docs_allowlist: tuple[str, ...]


class _RulesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    en: _LocaleRules
    ru: _LocaleRules


@dataclass(frozen=True)
class _Compiled:
    banned: tuple[tuple[str, re.Pattern[str]], ...]
    verdict: tuple[re.Pattern[str], ...]
    hedge: re.Pattern[str]
    allowlist: tuple[str, ...]


@cache
def _compiled(locale: Locale) -> _Compiled:
    raw = _RulesFile.model_validate(tomllib.loads(RULES_PATH.read_text(encoding="utf-8")))
    rules = raw.en if locale == "en" else raw.ru
    flags = re.IGNORECASE
    return _Compiled(
        banned=tuple((b.id, re.compile(b.pattern, flags)) for b in rules.banned),
        verdict=tuple(re.compile(p, flags) for p in rules.verdict),
        hedge=re.compile(rules.hedge, flags),
        allowlist=rules.docs_allowlist,
    )


def disclaimers(locale: Locale) -> tuple[str, ...]:
    """The exact sentences the docs profile accepts. README and model cards copy them verbatim."""
    return _compiled(locale).allowlist


def lint(text: str, locale: Locale, profile: Profile) -> tuple[Violation, ...]:
    rules = _compiled(locale)
    checked = text
    if profile == "docs":
        # blank out the exact disclaimer sentences; offsets stay valid
        for allowed in rules.allowlist:
            checked = checked.replace(allowed, " " * len(allowed))
    found = [
        Violation(rule_id, m.start(), m.end(), text[m.start() : m.end()])
        for rule_id, pattern in rules.banned
        for m in pattern.finditer(checked)
    ]
    for sentence in _SENTENCE.finditer(checked):
        if rules.hedge.search(sentence.group()):
            continue
        for pattern in rules.verdict:
            if m := pattern.search(sentence.group()):
                start = sentence.start() + m.start()
                end = sentence.start() + m.end()
                found.append(Violation("claim.verdict", start, end, text[start:end]))
    return tuple(sorted(found, key=lambda v: (v.start, v.rule_id)))
