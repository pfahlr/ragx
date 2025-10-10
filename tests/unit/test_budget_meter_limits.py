"""Budget meter enforcement behaviour as defined in the runner spec."""

from __future__ import annotations

import pytest

from pkgs.dsl.budget import BudgetBreachHard, BudgetMeter


def test_budget_meter_hard_cap_blocks_charge() -> None:
    meter = BudgetMeter(
        kind="node",
        subject="seed",
        config={"max_usd": 1.0},
        mode="hard",
    )

    assert meter.can_spend({"usd": 0.4}) is True

    charge = meter.charge({"usd": 0.4})
    assert pytest.approx(0.6) == charge.remaining["usd"]
    assert meter.can_spend({"usd": 0.7}) is False

    with pytest.raises(BudgetBreachHard):
        meter.charge({"usd": 0.7})


def test_budget_meter_soft_mode_flags_breach_without_raising() -> None:
    meter = BudgetMeter(
        kind="node",
        subject="draft",
        config={"max_tokens": 2_000},
        mode="soft",
    )

    assert meter.can_spend({"tokens": 2_500}) is True

    charge = meter.charge({"tokens": 2_500})
    assert charge.breached is True
    assert "tokens" in charge.overages
    assert pytest.approx(500.0) == meter.overages()["tokens"]
    assert meter.exceeded is True


def test_budget_meter_stop_action_defers_hard_breach() -> None:
    meter = BudgetMeter(
        kind="loop",
        subject="refine",
        config={"max_calls": 3, "breach_action": "stop"},
        mode="hard",
    )

    meter.charge({"calls": 2})
    charge = meter.charge({"calls": 2})

    assert charge.breached is True
    assert charge.overages["calls"] == 1
    assert meter.exceeded is True
