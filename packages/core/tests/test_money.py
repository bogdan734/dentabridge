import pytest
from pydantic import ValidationError

from dentabridge_core import Money, PriceRange


def _usd(cents: int) -> Money:
    return Money(amount_minor=cents, currency="USD")


def test_price_range_rejects_disorder() -> None:
    with pytest.raises(ValidationError, match="low <= mid <= high"):
        PriceRange(low=_usd(300), mid=_usd(200), high=_usd(400))
    with pytest.raises(ValidationError, match="low <= mid <= high"):
        PriceRange(low=_usd(100), mid=_usd(300), high=_usd(200))
    assert PriceRange(low=_usd(100), mid=_usd(100), high=_usd(100)).mid.amount_minor == 100


def test_price_range_rejects_mixed_currency() -> None:
    with pytest.raises(ValidationError, match="one currency"):
        PriceRange(low=_usd(100), mid=Money(amount_minor=150, currency="EUR"), high=_usd(200))


@pytest.mark.parametrize(("cents", "currency"), [(-1, "USD"), (100, "usd"), (100, "US")])
def test_money_rejects_bad_values(cents: int, currency: str) -> None:
    with pytest.raises(ValidationError):
        Money(amount_minor=cents, currency=currency)
