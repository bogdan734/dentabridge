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
    assert all("ё" not in e.tgt for e in glossary.entries), "ye and yo are one letter to the checks"


def test_the_glossary_reads_in_both_directions() -> None:
    glossary = Glossary(langs=("en", "ru"), entries=(GlossaryEntry(src="tooth|teeth", tgt="зуб*"),))
    assert glossary.for_direction("en", "ru") == glossary.entries
    assert glossary.for_direction("ru", "en") == (GlossaryEntry(src="зуб*", tgt="tooth|teeth"),)
    assert glossary.for_direction("en", "de") == (), "an unverified pair has no glossary"
