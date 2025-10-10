from __future__ import annotations

import math

import pytest

from pkgs.dsl import (
    BudgetExceededError,
    BudgetMeter,
    BudgetMode,
    Cost,
    CostBreakdown,
)


@pytest.fixture
def hard_meter() -> BudgetMeter:
    return BudgetMeter.from_spec(
        {"max_usd": 1.0, "max_tokens": 1200, "max_calls": 3, "mode": "hard"},
        scope="run",
        label="run_budget",
    )


def test_hard_budget_blocks_excess_spend(hard_meter: BudgetMeter) -> None:
    assert hard_meter.can_spend(Cost(usd=0.4, tokens=200, calls=1))
    charge = hard_meter.charge(Cost(usd=0.4, tokens=200, calls=1))
    assert not charge.breached
    assert charge.remaining.max_usd == pytest.approx(0.6)
    assert charge.remaining.max_tokens == 1000
    assert charge.remaining.max_calls == 2

    assert hard_meter.can_spend(Cost(usd=0.6, tokens=600, calls=1))
    hard_meter.charge(Cost(usd=0.5, tokens=500, calls=1))
    with pytest.raises(BudgetExceededError) as excinfo:
        hard_meter.charge(Cost(usd=0.2, tokens=300, calls=1))

    err = excinfo.value
    assert err.metric == "usd"
    assert math.isclose(err.attempted, 1.1)
    assert math.isclose(err.limit, 1.0)
    assert err.scope == "run_budget"


@pytest.mark.parametrize(
    "limit,charges",
    [
        (0.0, [0.3, 0.5, 1.2]),
        (None, [0.5, 0.4]),
    ],
)
def test_zero_or_missing_limits_are_unbounded(limit: float | None, charges: list[float]) -> None:
    meter = BudgetMeter.from_spec(
        {"max_usd": limit, "mode": "hard"}, scope="node", label="node:alpha"
    )
    for amount in charges:
        assert meter.can_spend(Cost(usd=amount))
        outcome = meter.charge(Cost(usd=amount))
        assert outcome.remaining.max_usd is math.inf


@pytest.mark.parametrize("mode", [BudgetMode.SOFT, "soft"])
def test_soft_budget_warns_but_allows_overage(mode: BudgetMode | str) -> None:
    meter = BudgetMeter.from_spec({"max_usd": 1.0, "mode": mode}, scope="node", label="n1")
    meter.charge(Cost(usd=0.4))
    result = meter.charge(Cost(usd=0.7))
    assert result.breached
    assert result.breach_kind == "soft"
    assert result.remaining.max_usd == pytest.approx(-0.1)
    assert result.remaining.total_spent.usd == pytest.approx(1.1)


@pytest.mark.parametrize(
    "increments,limit",
    [([0.1] * 10, 1.0), ([0.333333] * 3, 1.0)],
)
def test_float_precision_guard(increments: list[float], limit: float) -> None:
    meter = BudgetMeter.from_spec({"max_usd": limit, "mode": "hard"}, scope="run", label="run")
    total = 0.0
    for amount in increments:
        total += amount
        meter.charge(Cost(usd=amount))

    snapshot = meter.snapshot()
    assert snapshot.total_spent.usd == pytest.approx(total)
    assert snapshot.max_usd == pytest.approx(limit - total)


def test_charge_returns_structured_breakdown(hard_meter: BudgetMeter) -> None:
    result = hard_meter.charge(Cost(usd=0.25, tokens=100, calls=1, time_sec=2.5))
    assert isinstance(result.remaining, CostBreakdown)
    assert result.remaining.total_spent.usd == pytest.approx(0.25)
    assert result.remaining.total_spent.tokens == 100
    assert result.remaining.total_spent.calls == 1
    assert result.remaining.total_spent.time_sec == pytest.approx(2.5)


@pytest.mark.parametrize(
    "cost",
    [
        Cost(),
        Cost(usd=None, tokens=None, calls=None, time_sec=None),
        Cost(usd=0.0, tokens=0, calls=0, time_sec=0.0),
    ],
)
def test_cost_defaults_to_zero(cost: Cost) -> None:
    meter = BudgetMeter.from_spec({}, scope="run", label="run")
    result = meter.charge(cost)
    assert not result.breached
    assert result.remaining.total_spent.usd == 0.0
    assert result.remaining.total_spent.tokens == 0
    assert result.remaining.total_spent.calls == 0
    assert result.remaining.total_spent.time_sec == 0.0
