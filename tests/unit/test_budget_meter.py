"""Unit tests for the :mod:`pkgs.dsl.budget` BudgetMeter implementation."""

from __future__ import annotations

import pytest

from pkgs.dsl.budget import (
    BudgetBreachHard,
    BudgetMeter,
)


def make_meter(
    scope_type: str,
    scope_id: str,
    config: dict | None,
) -> BudgetMeter:
    """Helper that mirrors runner usage when instantiating meters."""

    return BudgetMeter(scope_type=scope_type, scope_id=scope_id, config=config)


def test_budget_meter_allows_spend_within_limits() -> None:
    meter = make_meter("run", "run", {"max_usd": 2.0})

    assert meter.can_spend({"usd": 1.25}) is True

    charge = meter.charge({"usd": 1.25})
    assert charge.breached is False
    assert meter.remaining().usd == pytest.approx(0.75)


def test_budget_meter_hard_breach_raises() -> None:
    meter = make_meter("node", "alpha", {"max_calls": 2})

    evaluation = meter.evaluate({"calls": 3})
    assert evaluation.breached is True
    assert evaluation.breach_kind == "hard"

    with pytest.raises(BudgetBreachHard) as exc_info:
        meter.charge({"calls": 3}, evaluation=evaluation)

    err = exc_info.value
    assert err.scope_id == "alpha"
    assert err.scope_type == "node"
    assert err.metrics == ("calls",)


def test_budget_meter_soft_breach_warns_and_accumulates() -> None:
    meter = make_meter("node_soft", "alpha", {"mode": "soft", "max_tokens": 100})

    evaluation = meter.evaluate({"tokens": 120})
    assert evaluation.allowed is True
    assert evaluation.breached is True
    assert evaluation.breach_kind == "soft"

    charge = meter.charge({"tokens": 120}, evaluation=evaluation)
    assert charge.breached is True
    assert charge.breach_kind == "soft"
    assert charge.remaining.tokens == -20


def test_budget_meter_zero_limits_are_unbounded() -> None:
    meter = make_meter(
        "node",
        "beta",
        {"max_usd": 0, "max_tokens": 0, "max_calls": 0, "time_limit_sec": 0},
    )

    assert meter.can_spend({"usd": 999.0, "tokens": 5000, "calls": 100}) is True
    charge = meter.charge({"usd": 10.0, "tokens": 2000, "calls": 10, "time_ms": 500})
    assert charge.breached is False
    remaining = meter.remaining()
    assert remaining.usd is None
    assert remaining.tokens is None
    assert remaining.calls is None
    assert remaining.time_ms is None


def test_budget_meter_remaining_snapshot_reflects_limits() -> None:
    meter = make_meter("node", "gamma", {"max_usd": 5.0, "max_tokens": 1000})

    meter.charge({"usd": 1.5, "tokens": 400})
    remaining = meter.remaining()
    assert remaining.usd == pytest.approx(3.5)
    assert remaining.tokens == 600

