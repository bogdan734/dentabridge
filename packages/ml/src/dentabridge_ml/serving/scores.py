"""A score for each "possible" / "not_seen" decision, read off the model's own token probabilities.

The answer is JSON the model wrote itself. Wherever `"status": "` ends, the next token starts the
value: `possible` or `not` (+ `_seen`). The probability mass of those two tokens at that position,
normalised, is the model's confidence that the finding is present.
"""

import re
from collections.abc import Mapping, Sequence

from dentabridge_core import FindingCode

_BEFORE_VALUE = re.compile(r'"status"\s*:\s*"')


def status_scores(
    tokens: Sequence[str], p_possible: Sequence[float], p_not: Sequence[float]
) -> list[float]:
    """One score per status value, in answer order.

    `p_possible[i]` and `p_not[i]` are the probabilities of the tokens `possible` and `not` at
    generated position i, whatever token was actually chosen there. Raises ValueError when a
    value cannot be scored (it shares a token with its quote, or neither status token has mass).
    """
    starts: dict[int, int] = {}
    text = ""
    for position, token in enumerate(tokens):
        starts[len(text)] = position
        text += token
    scores = []
    for match in _BEFORE_VALUE.finditer(text):
        at = starts.get(match.end())
        if at is None:
            raise ValueError(
                f"a status value does not start on a token boundary (char {match.end()})"
            )
        mass = p_possible[at] + p_not[at]
        if mass <= 0:
            raise ValueError(f"neither status token has probability at position {at}")
        scores.append(p_possible[at] / mass)
    return scores


def scores_by_code(
    codes: Sequence[FindingCode], scores: Sequence[float]
) -> Mapping[FindingCode, float] | None:
    """Pair scores with the codes of the parsed answer; None if they cannot be paired for sure."""
    if len(codes) != len(scores) or len(set(codes)) != len(codes):
        return None
    return dict(zip(codes, scores, strict=True))
