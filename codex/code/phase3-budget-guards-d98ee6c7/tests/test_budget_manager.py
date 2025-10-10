from __future__ import annotations

import pytest

from pkgs.dsl.budget import (
    BreachAction,
    BudgetDecision,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
)


@pytest.fixture()
def manager():
    return BudgetManager()


def enter_default_scope(manager: BudgetManager, scope_id: str = "run-1", limit: int = 100) -> None:
    manager.enter_scope(
        scope_type="run",
        scope_id=scope_id,
        spec=BudgetSpec(scope_type="run", limit_ms=limit, mode=BudgetMode.HARD, breach_action=BreachAction.STOP),
    )


def test_preflight_and_commit_within_limit(manager: BudgetManager) -> None:
    enter_default_scope(manager)
    decision = manager.preflight("run", "run-1", CostSnapshot(milliseconds=20))
    assert isinstance(decision, BudgetDecision)
    assert decision.should_stop is False
    assert decision.remaining_ms == pytest.approx(80)
    commit_decision = manager.commit("run", "run-1", CostSnapshot(milliseconds=30))
    assert commit_decision.should_stop is False
    assert commit_decision.remaining_ms == pytest.approx(70)
    assert commit_decision.overage_ms == pytest.approx(0)


def test_soft_budget_warns_but_continues(manager: BudgetManager) -> None:
    manager.enter_scope(
        scope_type="run",
        scope_id="run-soft",
        spec=BudgetSpec(scope_type="run", limit_ms=50, mode=BudgetMode.SOFT, breach_action=BreachAction.WARN),
    )
    preview = manager.preflight("run", "run-soft", CostSnapshot.from_seconds(0.04))
    assert preview.should_stop is False
    assert preview.action == BreachAction.WARN
    commit = manager.commit("run", "run-soft", CostSnapshot(milliseconds=60))
    assert commit.should_stop is False
    assert commit.action == BreachAction.WARN
    assert commit.overage_ms == pytest.approx(10)


def test_hard_budget_stops_on_overage(manager: BudgetManager) -> None:
    enter_default_scope(manager, scope_id="run-hard", limit=30)
    pre = manager.preflight("run", "run-hard", CostSnapshot(milliseconds=25))
    assert pre.should_stop is False
    commit = manager.commit("run", "run-hard", CostSnapshot(milliseconds=40))
    assert commit.should_stop is True
    assert commit.action == BreachAction.STOP
    assert commit.overage_ms == pytest.approx(10)


def test_exit_scope_removes_state(manager: BudgetManager) -> None:
    enter_default_scope(manager, scope_id="temp")
    manager.exit_scope("run", "temp")
    with pytest.raises(KeyError):
        manager.commit("run", "temp", CostSnapshot(milliseconds=1))
