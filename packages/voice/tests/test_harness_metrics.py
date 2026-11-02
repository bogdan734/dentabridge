import pytest

pytest.importorskip("sacrebleu")
from dentabridge_voice.harness.metrics import chrf, percentile, wer, word_errors


def test_percentiles_known() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    assert percentile([9, 1, 5], 0) == 1
    assert percentile([9, 1, 5], 100) == 9
    assert percentile([7], 95) == 7
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


def test_wer_counts_substitutions_insertions_and_deletions_over_the_reference() -> None:
    assert wer("my tooth hurts", "my tooth hurts") == 0
    assert wer("my tooth hurts", "my teeth hurts") == pytest.approx(1 / 3)
    assert wer("my tooth hurts", "my tooth") == pytest.approx(1 / 3)
    assert wer("my tooth hurts", "oh my tooth really hurts") == pytest.approx(2 / 3)
    assert wer("my tooth hurts", "") == 1
    assert word_errors("a b c d", "a x c") == (2, 4)


def test_wer_ignores_case_punctuation_and_yo() -> None:
    assert wer("Мой зуб болит.", "мой зуб, болит") == 0
    assert wer("Всё хорошо", "все хорошо") == 0
    assert wer("Hello, Doctor!", "hello doctor") == 0


def test_a_reference_must_have_words() -> None:
    with pytest.raises(ValueError, match="reference"):
        wer("...", "anything")


def test_chrf_is_100_for_the_same_text_and_low_for_unrelated_text() -> None:
    assert chrf(["У меня болит зуб"], ["У меня болит зуб"]) == pytest.approx(100.0)
    assert chrf(["У меня болит зуб"], ["Hello there my friend"]) < 10
    assert chrf(["У меня болит зуб"], ["У меня болит зуб сильно"]) > 60
    with pytest.raises(ValueError, match="same number"):
        chrf(["a"], ["a", "b"])
