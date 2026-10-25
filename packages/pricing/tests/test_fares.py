from datetime import date, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dentabridge_core import FareEstimate, FareRequest
from dentabridge_pricing.airports import load_airports
from dentabridge_pricing.fares import OfflineFareEstimator, UnknownAirport, load_fare_params

AIRPORTS = load_airports()
PARAMS = load_fare_params()
ESTIMATOR = OfflineFareEstimator(PARAMS, AIRPORTS)
CODES = sorted({"LHR", "JFK", "LIS", "MAD", "BKK", "SYD", "IST", "LAX", "ORD", "CDG", "NRT", "GRU"})
BOOKED = date(2026, 10, 25)


def request(
    origin: str = "LHR", dest: str = "LIS", days_ahead: int = 60, travellers: int = 1
) -> FareRequest:
    depart = BOOKED + timedelta(days=days_ahead)
    return FareRequest(
        origin_iata=origin,
        dest_iata=dest,
        depart=depart,
        return_=depart + timedelta(days=7),
        booked_on=BOOKED,
        travellers=travellers,
    )


def spread(estimate: FareEstimate) -> float:
    return estimate.range_usd.high.amount_minor / estimate.range_usd.low.amount_minor


def test_symmetric() -> None:
    assert ESTIMATOR.estimate(request("LHR", "BKK")) == ESTIMATOR.estimate(request("BKK", "LHR"))


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(CODES), st.sampled_from(CODES), st.sampled_from(CODES))
def test_monotone_in_distance(origin: str, near: str, far: str) -> None:
    from dentabridge_pricing.airports import haversine_km

    if len({origin, near, far}) < 3:
        return
    d_near = haversine_km(AIRPORTS[origin], AIRPORTS[near])
    d_far = haversine_km(AIRPORTS[origin], AIRPORTS[far])
    if d_near > d_far:
        near, far = far, near
    near_est = ESTIMATOR.estimate(request(origin, near)).range_usd.mid.amount_minor
    far_est = ESTIMATOR.estimate(request(origin, far)).range_usd.mid.amount_minor
    assert near_est <= far_est


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(CODES), st.sampled_from(CODES), st.integers(0, 400), st.integers(1, 6))
def test_low_le_mid_le_high(origin: str, dest: str, days_ahead: int, travellers: int) -> None:
    if origin == dest:
        return
    r = ESTIMATOR.estimate(request(origin, dest, days_ahead, travellers)).range_usd
    assert r.low.amount_minor <= r.mid.amount_minor <= r.high.amount_minor


def test_linear_in_travellers() -> None:
    one = ESTIMATOR.estimate(request(travellers=1)).range_usd.mid.amount_minor
    for n in (2, 3, 6):
        many = ESTIMATOR.estimate(request(travellers=n)).range_usd.mid.amount_minor
        assert abs(many - n * one) <= n  # rounding to cents only


def test_international_uses_q10_q90_and_domestic_q20_q80() -> None:
    across = ESTIMATOR.estimate(request("LHR", "JFK"))
    assert (across.international, across.interval) == (True, "q10-q90")
    inside = ESTIMATOR.estimate(request("JFK", "LAX"))
    assert (inside.international, inside.interval) == (False, "q20-q80")
    assert spread(across) > spread(inside), "crossing a border widens the interval"


def test_deterministic() -> None:
    assert ESTIMATOR.estimate(request()) == ESTIMATOR.estimate(request())


def test_sensible_magnitudes_for_the_hand_set_priors() -> None:
    transatlantic = ESTIMATOR.estimate(request("LHR", "JFK")).range_usd.mid.amount_minor / 100
    short_hop = ESTIMATOR.estimate(request("LIS", "MAD")).range_usd.mid.amount_minor / 100
    assert 400 < transatlantic < 1500
    assert 100 < short_hop < 500


def test_late_booking_costs_more_and_summer_costs_more() -> None:
    early = ESTIMATOR.estimate(request(days_ahead=100)).range_usd.mid.amount_minor
    late = ESTIMATOR.estimate(request(days_ahead=3)).range_usd.mid.amount_minor
    assert late > early
    january = FareRequest(
        origin_iata="LHR",
        dest_iata="LIS",
        depart=date(2027, 1, 20),
        return_=date(2027, 1, 27),
        booked_on=date(2026, 10, 25),
        travellers=1,
    )
    july = january.model_copy(update={"depart": date(2027, 7, 20), "return_": date(2027, 7, 27)})
    assert (
        ESTIMATOR.estimate(july).range_usd.mid.amount_minor
        > ESTIMATOR.estimate(january).range_usd.mid.amount_minor
    )


def test_params_record_datasets_and_metrics_or_say_they_are_hand_set() -> None:
    assert PARAMS.version == "v0"
    assert PARAMS.basis == "hand_set_v0"
    assert PARAMS.fare_mdape is None
    assert PARAMS.dataset_versions == ()
    assert ESTIMATOR.mdape is None
    assert ESTIMATOR.model_version == "offline-v0"


def test_unknown_airport_and_degenerate_requests_are_refused() -> None:
    with pytest.raises(UnknownAirport):
        ESTIMATOR.estimate(request("LHR", "QQQ"))
    with pytest.raises(ValueError, match="same airport"):
        ESTIMATOR.estimate(request("LHR", "LHR"))
