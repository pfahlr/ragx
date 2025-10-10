import math

import pytest

from pkgs.dsl.budget import (
    BudgetBreach,
    BudgetCharge,
    BudgetExceededError,
    BudgetMeter,
)


@pytest.fixture(name="hard_run_meter")
def fixture_hard_run_meter() -> BudgetMeter:
    return BudgetMeter(scope="run", config={"max_usd": 10.0, "max_tokens": 1000})


def test_can_spend_and_charge_within_limits(hard_run_meter: BudgetMeter) -> None:
    check = hard_run_meter.can_spend({"usd": 4.5, "tokens": 250})
    assert check.allowed
    assert check.breach is None

    charge = hard_run_meter.charge({"usd": 4.5, "tokens": 250})
    assert isinstance(charge, BudgetCharge)
    assert charge.breaches == ()
    remaining = hard_run_meter.remaining()
    assert math.isclose(remaining["usd"], 5.5, rel_tol=1e-9)
    assert remaining["tokens"] == 750


def test_charge_raises_when_hard_limit_exceeded(hard_run_meter: BudgetMeter) -> None:
    hard_run_meter.charge({"usd": 7.25})

    with pytest.raises(BudgetExceededError) as exc_info:
        hard_run_meter.charge({"usd": 3.0})

    err = exc_info.value
    assert err.metric == "usd"
    assert err.scope == "run"
    assert math.isclose(err.limit or 0.0, 10.0, rel_tol=1e-9)
    assert math.isclose(err.attempted, 10.25, rel_tol=1e-9)


def test_soft_budget_warns_but_allows_spend() -> None:
    meter = BudgetMeter(scope="node:soft", config={"mode": "soft", "max_usd": 2.5})

    preflight = meter.can_spend({"usd": 3.0})
    assert preflight.allowed
    assert isinstance(preflight.breach, BudgetBreach)
    assert preflight.breach.level == "soft"

    charge = meter.charge({"usd": 3.0})
    assert charge.breaches and charge.breaches[0].level == "soft"
    remaining = meter.remaining()
    assert remaining["usd"] == 0


@pytest.mark.parametrize("limit", [0, None])
def test_zero_or_none_limits_treated_as_unbounded(limit: float | None) -> None:
    meter = BudgetMeter(scope="unbounded", config={"max_usd": limit, "max_tokens": limit})
    meter.charge({"usd": 10_000, "tokens": 500_000})
    remaining = meter.remaining()
    assert remaining["usd"] is None
    assert remaining["tokens"] is None


def test_time_limit_converted_to_milliseconds() -> None:
    meter = BudgetMeter(scope="timer", config={"time_limit_sec": 1.25})
    meter.charge({"time_ms": 1_000})
    with pytest.raises(BudgetExceededError):
        meter.charge({"time_ms": 300})

