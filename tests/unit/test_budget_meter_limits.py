"""Unit tests for BudgetMeter limit enforcement."""

from __future__ import annotations

import pytest

from pkgs.dsl.budget import (
    BudgetBreachHard,
    BudgetMeter,
    BudgetSpec,
    CostSnapshot,
)


@pytest.fixture
def hard_meter() -> BudgetMeter:
    return BudgetMeter.from_spec(
        BudgetSpec(mode="hard", max_usd=1.0, scope="run"),
        scope="run",
    )


@pytest.fixture
def soft_meter() -> BudgetMeter:
    return BudgetMeter.from_spec(
        BudgetSpec(mode="soft", max_tokens=100, scope="node"),
        scope="node",
    )


def test_can_spend_checks_remaining_budget(hard_meter: BudgetMeter) -> None:
    assert hard_meter.can_spend(CostSnapshot(usd=0.4)) is True
    hard_meter.charge(CostSnapshot(usd=0.6))
    assert hard_meter.can_spend(CostSnapshot(usd=0.5)) is False
    assert hard_meter.remaining().usd == pytest.approx(0.4)


def test_hard_cap_blocks_charge(hard_meter: BudgetMeter) -> None:
    hard_meter.charge(CostSnapshot(usd=0.8, calls=1))
    with pytest.raises(BudgetBreachHard):
        hard_meter.charge(CostSnapshot(usd=0.3))


def test_soft_cap_emits_breach_but_allows_charge(soft_meter: BudgetMeter) -> None:
    outcome = soft_meter.charge(CostSnapshot(tokens_in=40, tokens_out=40))
    assert outcome.soft_breach is False
    outcome = soft_meter.charge(CostSnapshot(tokens_in=30, tokens_out=50))
    assert outcome.soft_breach is True
    assert outcome.breach_kind == "soft"
    assert soft_meter.exceeded is True


def test_unlimited_budget_allows_large_spend() -> None:
    meter = BudgetMeter.unlimited(scope="run")
    assert meter.can_spend(CostSnapshot(usd=1_000_000, calls=10_000)) is True
    outcome = meter.charge(CostSnapshot(usd=500_000, tokens_in=1_000_000))
    assert outcome.soft_breach is False
    assert meter.remaining().usd is None


def test_loop_stop_budget_records_breach_action_stop() -> None:
    meter = BudgetMeter.from_spec(
        BudgetSpec(breach_action="stop", max_calls=2, scope="loop"),
        scope="loop",
    )
    meter.charge(CostSnapshot(calls=1))
    meter.charge(CostSnapshot(calls=1))
    assert meter.exceeded is True
    assert meter.stop_behavior == "stop"
