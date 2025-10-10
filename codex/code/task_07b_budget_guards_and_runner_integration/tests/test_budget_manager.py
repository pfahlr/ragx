import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import decimal

import pytest

from codex.code.task_07b_budget_guards_and_runner_integration.budget_manager import BudgetManager
from codex.code.task_07b_budget_guards_and_runner_integration.budget_models import (
    BreachAction,
    BudgetDecision,
    BudgetMode,
    BudgetSpec,
    CostAmount,
)
from codex.code.task_07b_budget_guards_and_runner_integration.budget_models import BudgetBreachError


def test_preview_allows_under_limit():
    manager = BudgetManager()
    spec = BudgetSpec(
        scope_id="run",
        limit=CostAmount.of("10"),
        mode=BudgetMode.HARD,
        breach_action=BreachAction.STOP,
    )
    manager.register_scope("run", spec)

    preview = manager.preview("run", CostAmount.of("3.5"))

    assert preview.decision is BudgetDecision.ALLOW
    assert preview.snapshot.remaining == decimal.Decimal("6.5")
    assert preview.snapshot.overage == decimal.Decimal("0")

    charge = manager.commit("run", CostAmount.of("3.5"))

    assert charge.decision is BudgetDecision.ALLOW
    assert charge.snapshot.spent == decimal.Decimal("3.5")

    summary = manager.summary("run")
    assert summary.total_spent == decimal.Decimal("3.5")
    assert summary.total_remaining == decimal.Decimal("6.5")


def test_preview_stop_on_hard_budget():
    manager = BudgetManager()
    spec = BudgetSpec(
        scope_id="run",
        limit=CostAmount.of("5"),
        mode=BudgetMode.HARD,
        breach_action=BreachAction.STOP,
    )
    manager.register_scope("run", spec)

    preview = manager.preview("run", CostAmount.of("6"))

    assert preview.decision is BudgetDecision.STOP
    assert preview.snapshot.overage == decimal.Decimal("1")
    assert preview.breach is not None
    assert preview.breach.reason == "limit_exceeded"

    with pytest.raises(BudgetBreachError):
        manager.commit("run", CostAmount.of("6"))


def test_soft_budget_warns_and_allows_commit():
    manager = BudgetManager()
    spec = BudgetSpec(
        scope_id="run",
        limit=CostAmount.of("5"),
        mode=BudgetMode.SOFT,
        breach_action=BreachAction.WARN,
    )
    manager.register_scope("run", spec)

    preview = manager.preview("run", CostAmount.of("6"))

    assert preview.decision is BudgetDecision.WARN
    assert preview.snapshot.overage == decimal.Decimal("1")
    assert preview.breach is not None

    charge = manager.commit("run", CostAmount.of("6"))
    assert charge.decision is BudgetDecision.WARN
    assert charge.snapshot.spent == decimal.Decimal("6")

    summary = manager.summary("run")
    assert summary.total_overage == decimal.Decimal("1")


def test_commit_requires_registered_scope():
    manager = BudgetManager()

    with pytest.raises(KeyError):
        manager.preview("missing", CostAmount.of("1"))

    spec = BudgetSpec(
        scope_id="run",
        limit=CostAmount.of("5"),
    )
    manager.register_scope("run", spec)

    with pytest.raises(KeyError):
        manager.commit("missing", CostAmount.of("1"))


def test_negative_cost_rejected():
    manager = BudgetManager()
    spec = BudgetSpec(scope_id="run", limit=CostAmount.of("5"))
    manager.register_scope("run", spec)

    with pytest.raises(ValueError):
        manager.preview("run", CostAmount.of("-1"))

    manager.preview("run", CostAmount.of("1"))

    with pytest.raises(ValueError):
        manager.commit("run", CostAmount.of("-2"))
