import re

from dentabridge_core import GlossaryEntry
from dentabridge_voice.glossary import GLOSSARY_DIR, Glossary, load_glossary


def test_the_shipped_glossary_is_en_ru_with_about_two_hundred_terms() -> None:
    glossary = load_glossary(GLOSSARY_DIR / "en-ru.tsv")
    assert glossary.langs == ("en", "ru")
    assert 150 <= len(glossary.entries) <= 260
    sources = [e.src for e in glossary.entries]
    assert len(sources) == len(set(sources)), "one row per source term"
    assert all("\t" not in e.src + e.tgt and e.src == e.src.strip() for e in glossary.entries)
    assert all(e.src == e.src.casefold() for e in glossary.entries), "terms are written lower case"


def test_each_side_starts_with_a_full_form_the_translator_prompt_can_quote() -> None:
    for entry in load_glossary(GLOSSARY_DIR / "en-ru.tsv").entries:
        assert "*" not in entry.src.split("|")[0], entry
        assert "*" not in entry.tgt.split("|")[0], entry


def test_no_word_mixes_latin_and_cyrillic_letters() -> None:
    """A Latin e typed inside a Russian word looks right and never matches."""
    mixed = re.compile(r"(?=\w*[a-z])(?=\w*[а-яё])\w+", re.IGNORECASE)
    for entry in load_glossary(GLOSSARY_DIR / "en-ru.tsv").entries:
        assert not mixed.search(entry.src + " " + entry.tgt), entry


def test_the_glossary_reads_in_both_directions() -> None:
    glossary = Glossary(langs=("en", "ru"), entries=(GlossaryEntry(src="tooth|teeth", tgt="зуб*"),))
    assert glossary.for_direction("en", "ru") == glossary.entries
    assert glossary.for_direction("ru", "en") == (GlossaryEntry(src="зуб*", tgt="tooth|teeth"),)
    assert glossary.for_direction("en", "de") == (), "an unverified pair has no glossary"


def test_licences_file_has_a_dated_row_for_every_default_model_and_never_imports_piper() -> None:
    import tomllib
    from datetime import date
    from pathlib import Path

    root = GLOSSARY_DIR.parent
    document = tomllib.loads((root / "LICENCES.toml").read_text(encoding="utf-8"))
    rows = [row for kind in document.values() for row in kind]
    assert all(date.fromisoformat(row["verified_on"]) for row in rows)
    assert all(row["licence"] and row["origin"] for row in rows)
    defaults = {row["component"] for row in rows if row["status"] == "default"}
    assert any("Qwen3" in c for c in defaults)
    assert any("faster-whisper" in c for c in defaults)
    assert any("ljspeech" in c for c in defaults)
    assert any("denis" in c for c in defaults)
    source = "".join(p.read_text(encoding="utf-8") for p in Path(root / "src").rglob("*.py"))
    assert "import piper" not in source
    assert "from piper" not in source
