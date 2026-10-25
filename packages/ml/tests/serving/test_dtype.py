import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dentabridge_ml.serving.transformers_backend import dtype_for  # noqa: E402


@pytest.mark.parametrize(
    ("device", "capability", "expected"),
    [
        ("cuda", (7, 5), torch.float16),  # T4: no native bf16
        ("cuda", (8, 0), torch.bfloat16),  # A100
        ("cuda", (9, 0), torch.bfloat16),
        ("mps", None, torch.bfloat16),
        ("cpu", None, torch.bfloat16),
    ],
)
def test_bf16_only_where_the_hardware_has_it(
    device: str, capability: tuple[int, int] | None, expected: object
) -> None:
    assert dtype_for(device, capability) == expected
