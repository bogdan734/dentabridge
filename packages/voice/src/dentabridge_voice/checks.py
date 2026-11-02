"""Safety checks on a translated sentence: did the dental terms and the numbers survive?

Both are cheap string tests on text the pipeline already has, run on every utterance and every chat
message. A failed check does not block anything; it tells the listener to double-check.

Glossary terms: "a|b" are alternatives, a trailing * in a word is a stem, words match whole and in
order, and ё equals е. A term is demanded in the translation only when its source side is in the
sentence.
"""

import re
from collections import Counter
from collections.abc import Sequence

from dentabridge_core import GlossaryEntry

_WORD = re.compile(r"\w+")
_PATTERN_WORD = re.compile(r"\w+\*?")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _fold(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _words(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


def _matches(pattern: Sequence[str], words: Sequence[str]) -> bool:
    def same(want: str, got: str) -> bool:
        return got.startswith(want[:-1]) if want.endswith("*") else got == want

    n = len(pattern)
    return n > 0 and any(
        all(same(want, got) for want, got in zip(pattern, words[i : i + n], strict=True))
        for i in range(len(words) - n + 1)
    )


def _present(term: str, words: Sequence[str]) -> bool:
    return any(_matches(_PATTERN_WORD.findall(_fold(alt)), words) for alt in term.split("|"))


def _display(term: str) -> str:
    return term.split("|")[0].replace("*", "")


def matched_entries(text: str, entries: Sequence[GlossaryEntry]) -> tuple[GlossaryEntry, ...]:
    """The entries whose source side occurs in `text`: what a translator needs to be told."""
    words = _words(text)
    return tuple(e for e in entries if _present(e.src, words))


def missing_terms(src: str, tgt: str, entries: Sequence[GlossaryEntry]) -> tuple[str, ...]:
    """Source terms the sentence used whose agreed translation is not in the translation."""
    target_words = _words(tgt)
    return tuple(
        _display(e.src) for e in matched_entries(src, entries) if not _present(e.tgt, target_words)
    )


def _numbers(text: str) -> Counter[str]:
    return Counter(n.replace(",", ".") for n in _NUMBER.findall(text))


def digits_preserved(src: str, tgt: str) -> bool:
    """The same numbers, the same number of times, in any order ("tooth 36" must not become 63).
    A number spelled out in words is not a digit and counts as lost."""
    return _numbers(src) == _numbers(tgt)
