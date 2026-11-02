import pytest

from dentabridge_core import GlossaryEntry
from dentabridge_voice.checks import digits_preserved, matched_entries, missing_terms

TOOTH = GlossaryEntry(src="tooth|teeth", tgt="зуб*")
ROOT_CANAL = GlossaryEntry(src="root canal", tgt="канал*|эндодонт*")
PAIN = GlossaryEntry(src="pain|ache", tgt="бол*")
ENTRIES = (TOOTH, ROOT_CANAL, PAIN)


def test_a_term_present_in_the_source_must_appear_in_the_translation() -> None:
    assert missing_terms("my tooth hurts", "мой зуб болит", ENTRIES) == ()
    assert missing_terms("my tooth hurts", "у меня болит", ENTRIES) == ("tooth",)


def test_missing_glossary_term_reported_in_the_order_of_the_glossary() -> None:
    missing = missing_terms(
        "I have pain in a tooth after the root canal", "это было давно", ENTRIES
    )
    assert missing == ("tooth", "root canal", "pain")


def test_terms_absent_from_the_source_are_never_demanded() -> None:
    assert missing_terms("good morning", "доброе утро", ENTRIES) == ()


@pytest.mark.parametrize("translation", ["зубы", "зубов", "Зуба", "ЗУБНАЯ"])
def test_a_stem_matches_every_ending_and_case(translation: str) -> None:
    assert missing_terms("tooth", translation, (TOOTH,)) == ()


def test_alternatives_and_plurals_are_accepted_on_both_sides() -> None:
    assert missing_terms("both teeth", "оба зуба", (TOOTH,)) == ()
    assert missing_terms("root canal treatment", "лечение каналов", (ROOT_CANAL,)) == ()
    assert missing_terms("root canal treatment", "эндодонтия", (ROOT_CANAL,)) == ()


def test_whole_words_only() -> None:
    assert missing_terms("painful", "ничего", (PAIN,)) == (), "painful is not pain"
    assert missing_terms("a toothpaste", "паста", (TOOTH,)) == (), "toothpaste is not tooth"


def test_yo_and_ye_are_one_letter() -> None:
    swelling = GlossaryEntry(src="swelling", tgt="отек*")
    assert missing_terms("swelling", "отёк щеки", (swelling,)) == ()


def test_multiword_patterns_match_consecutive_words_only() -> None:
    wisdom = GlossaryEntry(src="wisdom tooth|wisdom teeth", tgt="зуб* мудрост*")
    assert missing_terms("a wisdom tooth", "зуб мудрости", (wisdom,)) == ()
    assert missing_terms("a wisdom tooth", "мудрости не хватает, а зуб болит", (wisdom,)) == (
        "wisdom tooth",
    )


def test_matched_entries_are_those_whose_source_side_is_in_the_text() -> None:
    assert matched_entries("my tooth hurts", ENTRIES) == (TOOTH,)
    assert matched_entries("зуб", ENTRIES) == ()


@pytest.mark.parametrize(
    ("src", "tgt", "ok"),
    [
        ("tooth 36", "зуб 36", True),
        ("tooth 36", "зуб 63", False),
        ("take 2 tablets at 10:30", "примите 2 таблетки в 10:30", True),
        ("take 2 tablets", "примите две таблетки", False),
        ("2.5 mg", "2,5 мг", True),
        ("two teeth", "два зуба", True),
        ("36 and 46", "46 и 36", True),
        ("36 and 36", "36", False),
        ("", "", True),
    ],
)
def test_digit_swap_flagged(src: str, tgt: str, ok: bool) -> None:
    assert digits_preserved(src, tgt) is ok
