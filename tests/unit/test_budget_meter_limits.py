"""Unit tests for the DSL budget meter implementation."""

from __future__ import annotations

import pytest

from pkgs.dsl.budget import BudgetBreachError, BudgetMeter


def test_budget_meter_enforces_hard_caps() -> None:
    meter = BudgetMeter(
        name="node:answer",
        scope="node",
        config={
            "mode": "hard",
            "max_usd": 2.5,
            "max_tokens": 100,
            "max_calls": 5,
        },
    )

    assert meter.can_spend({"usd": 1.0, "tokens": 20, "calls": 1})

    result = meter.charge({"usd": 1.0, "tokens": 20, "calls": 1})
    assert result.cost == {"usd": 1.0, "tokens": 20, "calls": 1}
    assert result.warning is None
    assert meter.remaining()["usd"] == pytest.approx(1.5)
    assert meter.remaining()["tokens"] == 80
    assert meter.remaining()["calls"] == 4

    result = meter.charge({"usd": 1.5, "tokens": 80, "calls": 4})
    assert result.warning is None
    assert meter.remaining()["usd"] == pytest.approx(0.0)
    assert meter.remaining()["tokens"] == 0
    assert meter.remaining()["calls"] == 0
    assert meter.is_exhausted

    assert not meter.can_spend({"usd": 0.01})

    with pytest.raises(BudgetBreachError):
        meter.charge({"usd": 0.1})


def test_budget_meter_soft_budget_emits_warning() -> None:
    meter = BudgetMeter(
        name="answer",
        scope="spec",
        config={
            "mode": "soft",
            "max_tokens": 50,
        },
    )

    first = meter.charge({"tokens": 30})
    assert first.warning is None
    assert meter.remaining()["tokens"] == 20

    second = meter.charge({"tokens": 25})
    assert second.warning is not None
    assert second.warning.scope == "spec:answer"
    assert second.warning.over == {"tokens": 5}
    assert meter.remaining()["tokens"] == 0
    assert meter.is_exhausted

    # Soft budgets should never block spending.
    assert meter.can_spend({"tokens": 10})
    third = meter.charge({"tokens": 10})
    assert third.warning is not None
    assert third.warning.over == {"tokens": 15.0}
