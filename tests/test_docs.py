"""The README and the licences page are generated, linted and free of hand-typed results."""

import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import ModuleType

from dentabridge_core.policy.claims import disclaimers
from dentabridge_core.policy.lint_cli import main as lint

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LICENCES = ROOT / "docs" / "licences.md"
LINK = re.compile(r"(?:\]\(|src=\")(?!https?://|#|mailto:)([^)\s\"]+)")


def test_readme_tables_are_rendered_from_the_results(tool: Callable[[str], ModuleType]) -> None:
    render_readme = tool("render_readme")
    text = README.read_text(encoding="utf-8")
    assert render_readme.render(text) == text, "run `make docs`"
    blocks = {match["name"] for match in render_readme.BLOCK.finditer(text)}
    assert blocks == set(render_readme.BLOCKS), "every renderer has its block in the README"


def test_no_measured_number_is_typed_outside_the_generated_blocks(
    tool: Callable[[str], ModuleType],
) -> None:
    prose = tool("render_readme").BLOCK.sub("", README.read_text(encoding="utf-8"))
    prose = re.sub(r"<[^>]*>|\]\(https?://[^)]*\)", "", prose)  # tags and URLs hold no results
    typed = re.findall(r"\d[\d.,]*\s?(?:%|ms\b|s\b|chrF|WER)", prose)
    assert typed == [], "results come from docs/results through tools/render_readme.py"


def test_licences_page_is_rendered_from_the_tables(tool: Callable[[str], ModuleType]) -> None:
    assert LICENCES.read_text(encoding="utf-8") == tool("render_licences").render(), (
        "run `make docs`"
    )


def test_every_licence_row_says_what_it_is_and_when_it_was_read(
    tool: Callable[[str], ModuleType],
) -> None:
    render_licences = tool("render_licences")
    for package, _ in render_licences.SECTIONS:
        found = render_licences.rows(package)
        assert found, package
        for row in found:
            assert date.fromisoformat(row["verified_on"]), row
            assert row["licence"], row
            assert row["status"], row


def test_the_readme_says_what_it_is_not() -> None:
    text = README.read_text(encoding="utf-8")
    assert "not a medical device" in text.lower()
    for sentence in disclaimers("en")[:2]:
        assert sentence in text, sentence


def test_a_dataset_with_no_licence_never_reaches_the_readme() -> None:
    text = README.read_text(encoding="utf-8").lower()
    assert "oral diseases" not in text
    assert "salmansajid" not in text


def test_the_docs_pass_the_claims_linter() -> None:
    assert lint(["--profile", "docs", "--locale", "en", str(README), str(LICENCES)]) == 0


def test_relative_links_in_the_readme_resolve() -> None:
    targets = LINK.findall(README.read_text(encoding="utf-8"))
    assert targets
    missing = [t for t in targets if not (ROOT / t.split("#")[0]).exists()]
    assert missing == []
