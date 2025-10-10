"""BudgetMeter behavior contract tests."""

from __future__ import annotations

import math

import pytest

from pkgs.dsl.budget import BudgetExceededError, BudgetMeter, Cost


@pytest.mark.parametrize(
    "charges",
    [
        [Cost(usd=0.25), Cost(usd=0.35)],
        [Cost(usd=0.4), Cost(usd=0.6)],
    ],
)
def test_hard_budget_blocks_spend_when_limit_reached(charges: list[Cost]) -> None:
    meter = BudgetMeter.from_budget({"max_usd": 0.5, "mode": "hard"}, scope="run")

    for cost in charges[:-1]:
        decision = meter.charge(cost)
        assert decision.allowed is True
        assert not decision.breached

    expected_remaining = 0.5 - sum(cost.usd for cost in charges[:-1])
    assert math.isclose(meter.remaining.usd, expected_remaining, rel_tol=1e-9)

    with pytest.raises(BudgetExceededError) as excinfo:
        meter.charge(charges[-1])

    decision = excinfo.value.decision
    assert decision.allowed is False
    assert decision.breached == ("usd",)
    # Meter should not mutate when a hard cap blocks the charge.
    assert math.isclose(meter.spent.usd, sum(c.usd for c in charges[:-1]), rel_tol=1e-9)
    assert math.isclose(meter.remaining.usd, expected_remaining, rel_tol=1e-9)


def test_soft_budget_emits_breach_without_blocking() -> None:
    meter = BudgetMeter.from_budget({"max_tokens": 1000, "mode": "soft"}, scope="node")

    first = meter.charge(Cost(tokens=900))
    assert first.allowed is True
    assert not first.soft_breach
    assert first.breached == ()

    second = meter.charge(Cost(tokens=250))
    assert second.allowed is True
    assert second.soft_breach is True
    assert second.breached == ("tokens",)
    assert meter.spent.tokens == 1150
    assert meter.remaining.tokens == -150


def test_zero_or_none_limits_mean_unlimited() -> None:
    meter = BudgetMeter.from_budget({"max_calls": 0, "max_usd": None}, scope="node")

    for _ in range(5):
        decision = meter.charge(Cost(calls=1, usd=2.5))
        assert decision.allowed is True
        assert not decision.breached

    assert meter.spent.calls == 5
    assert meter.remaining.calls is math.inf
    assert math.isclose(meter.spent.usd, 12.5, rel_tol=1e-9)


def test_can_spend_preview_does_not_mutate_state() -> None:
    meter = BudgetMeter.from_budget({"max_usd": 1.0}, scope="run")

    assert meter.can_spend(Cost(usd=0.75)) is True
    assert meter.spent.usd == 0.0
    assert meter.remaining.usd == 1.0

    # A preview that would breach should not mutate either.
    assert meter.can_spend(Cost(usd=1.5)) is False
    assert meter.spent.usd == 0.0
    assert meter.remaining.usd == 1.0
