"""The dental glossary: one file per language pair, read in either direction."""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dentabridge_core import GlossaryEntry

GLOSSARY_DIR = Path(__file__).resolve().parents[2] / "glossary"
_CALL_ID = re.compile(r"dental-([a-z]{2,3}-[a-z]{2,3})")


class Glossary(BaseModel):
    model_config = ConfigDict(frozen=True)

    langs: tuple[str, str]  # the order of the file's columns
    entries: tuple[GlossaryEntry, ...]

    def for_direction(self, src: str, tgt: str) -> tuple[GlossaryEntry, ...]:
        """The entries oriented src -> tgt; none for a pair this file does not cover."""
        if (src, tgt) == self.langs:
            return self.entries
        if (tgt, src) == self.langs:
            return tuple(GlossaryEntry(src=e.tgt, tgt=e.src) for e in self.entries)
        return ()


def load_glossary(path: Path) -> Glossary:
    """`en-ru.tsv` holds english in the first column and russian in the second. `#` lines are
    comments; the first other line names the columns."""
    langs = path.stem.split("-")
    if len(langs) != 2:
        raise ValueError(f"{path.name}: expected a name like en-ru.tsv")
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header, body = rows[0], rows[1:]
    if header != langs:
        raise ValueError(f"{path.name}: the header {header} does not match the file name")
    bad = [row for row in body if len(row) != 2]
    if bad:
        raise ValueError(f"{path.name}: rows need exactly two columns: {bad[0]}")
    return Glossary(
        langs=(langs[0], langs[1]),
        entries=tuple(GlossaryEntry(src=src, tgt=tgt) for src, tgt in body),
    )


def glossary_for(glossary_id: str) -> Glossary:
    """The glossary a call's room metadata names: "dental-en-ru" is glossary/en-ru.tsv (the API
    names it after the call's two languages, sorted)."""
    match = _CALL_ID.fullmatch(glossary_id)
    path = GLOSSARY_DIR / f"{match[1]}.tsv" if match else None
    if path is None or not path.is_file():
        raise ValueError(f"no glossary {glossary_id!r} in {GLOSSARY_DIR}")
    return load_glossary(path)
