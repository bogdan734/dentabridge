"""A score for each "possible" / "not_seen" decision, read off the model's own token probabilities.

The answer is JSON the model wrote itself, and its tokens split cleanly: `status`, `":"`, then
`possible` or `not` (+ `_seen`). The probability mass of those two tokens at that position,
normalised, is the model's confidence that the finding is present.
"""

from collections.abc import Mapping, Sequence

from dentabridge_core import FindingCode

_BEFORE_VALUE = ("status", '":"')


def status_scores(
    tokens: Sequence[str], p_possible: Sequence[float], p_not: Sequence[float]
) -> list[float]:
    """One score per status value, in answer order.

    `p_possible[i]` and `p_not[i]` are the probabilities of the tokens `possible` and `not` at
    generated position i, whatever token was actually chosen there.
    """
    scores = []
    for i in range(2, len(tokens)):
        if (tokens[i - 2], tokens[i - 1]) != _BEFORE_VALUE:
            continue
        mass = p_possible[i] + p_not[i]
        if mass <= 0:
            raise ValueError(f"neither status token has probability at position {i}")
        scores.append(p_possible[i] / mass)
    return scores


def scores_by_code(
    codes: Sequence[FindingCode], scores: Sequence[float]
) -> Mapping[FindingCode, float] | None:
    """Pair scores with the codes of the parsed answer; None if they cannot be paired for sure."""
    if len(codes) != len(scores) or len(set(codes)) != len(codes):
        return None
    return dict(zip(codes, scores, strict=True))
