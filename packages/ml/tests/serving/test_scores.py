import pytest

from dentabridge_core import FindingCode
from dentabridge_ml.serving.scores import scores_by_code, status_scores

POSSIBLE = [
    '{"',
    "findings",
    '":[{"',
    "code",
    '":"',
    "c",
    "aries",
    '","',
    "status",
    '":"',
    "possible",
]
NOT_SEEN = ['","', "regions", '":[],', '"', "status", '":"', "not", "_seen", '"}']


def _probs(
    tokens: list[str], at: dict[int, tuple[float, float]]
) -> tuple[list[float], list[float]]:
    return (
        [at.get(i, (0.0, 0.0))[0] for i in range(len(tokens))],
        [at.get(i, (0.0, 0.0))[1] for i in range(len(tokens))],
    )


def test_score_is_the_possible_share_at_each_status_token() -> None:
    tokens = [*POSSIBLE, *NOT_SEEN]
    first, second = tokens.index("possible"), tokens.index("not")
    p_possible, p_not = _probs(tokens, {first: (0.6, 0.2), second: (0.1, 0.7)})
    assert status_scores(tokens, p_possible, p_not) == pytest.approx([0.75, 0.125])


def test_tokens_outside_a_status_value_are_never_scored() -> None:
    tokens = ['{"', "note", '":"', "possible", '"}']  # "possible" in another field
    p_possible, p_not = _probs(tokens, {3: (0.9, 0.0)})
    assert status_scores(tokens, p_possible, p_not) == []


def test_a_status_token_the_model_gave_no_mass_to_is_an_error() -> None:
    tokens = ["status", '":"', "possible"]
    with pytest.raises(ValueError, match="neither"):
        status_scores(tokens, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_scores_are_paired_with_the_answer_codes_in_order() -> None:
    codes = [FindingCode.CARIES, FindingCode.FILLING]
    assert scores_by_code(codes, [0.8, 0.1]) == {FindingCode.CARIES: 0.8, FindingCode.FILLING: 0.1}


@pytest.mark.parametrize(
    ("codes", "scores"),
    [([FindingCode.CARIES], [0.5, 0.5]), ([FindingCode.CARIES, FindingCode.CARIES], [0.5, 0.5])],
)
def test_a_mismatch_gives_no_scores_rather_than_wrong_ones(
    codes: list[FindingCode], scores: list[float]
) -> None:
    assert scores_by_code(codes, scores) is None
