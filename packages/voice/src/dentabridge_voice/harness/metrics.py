"""Numbers for the harness: latency percentiles, word error rate, chrF++."""

import math
import re
from collections.abc import Sequence

from sacrebleu.metrics.chrf import CHRF

_WORD = re.compile(r"\w+")


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile by linear interpolation between the closest ranks."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold().replace("ё", "е"))


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word edit distance, number of reference words)."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        raise ValueError("the reference has no words")
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1], len(ref)


def wer(reference: str, hypothesis: str) -> float:
    errors, words = word_errors(reference, hypothesis)
    return errors / words


def chrf(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus chrF++ on a 0-100 scale (character 6-grams and word bigrams)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be the same number")
    return float(CHRF(word_order=2).corpus_score(list(hypotheses), [list(references)]).score)
