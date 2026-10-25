"""`dentabridge-lint`: run the claims linter over JSON values and Markdown text."""

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

from dentabridge_core._base import Locale
from dentabridge_core.policy.claims import Profile, lint

_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _json_strings(value: object, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _json_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _json_strings(item, f"{path}[{index}]")


def _lint_file(path: Path, locale: Locale, profile: Profile) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if path.suffix == ".json":
        for where, text in _json_strings(json.loads(raw)):
            problems += [
                f'{path}:{where}: {v.rule_id}: "{v.excerpt}"' for v in lint(text, locale, profile)
            ]
        return problems
    # code fences are not prose: blank them but keep line numbers
    text = _FENCE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), raw)
    for v in lint(text, locale, profile):
        line = text.count("\n", 0, v.start) + 1
        problems.append(f'{path}:{line}: {v.rule_id}: "{v.excerpt}"')
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dentabridge-lint", description=__doc__)
    parser.add_argument("--profile", choices=["patient", "docs"], required=True)
    parser.add_argument("--locale", choices=["en", "ru"], required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    problems = [
        problem for path in args.paths for problem in _lint_file(path, args.locale, args.profile)
    ]
    for problem in problems:
        print(problem)
    return 1 if problems else 0
