import decimal
import pathlib
import sys
from typing import Sequence

sys.path.append(str(pathlib.Path(__file__).resolve().parents[4]))

import pytest

from codex.code.work.runner.budget_manager import BudgetBreachError, BudgetManager
from codex.code.work.runner.budget_models import (
    BudgetScope,
    BudgetSpec,
    BudgetChargeOutcome,
    CostSnapshot,
)


@pytest.fixture
def run_scope() -> BudgetScope:
    return BudgetScope.run("run-123")


@pytest.fixture
def node_scope() -> BudgetScope:
    return BudgetScope.node("node-alpha")


@pytest.fixture
def spec_scope() -> BudgetScope:
    return BudgetScope.spec("node-alpha", "completion")


@pytest.fixture
def budget_manager(run_scope: BudgetScope, node_scope: BudgetScope, spec_scope: BudgetScope) -> BudgetManager:
    budgets = {
        run_scope: BudgetSpec.from_dict({"mode": "hard", "max_usd": "5.00", "max_calls": 10}),
        node_scope: BudgetSpec.from_dict({"mode": "hard", "max_usd": "3.00", "max_calls": 4}),
        spec_scope: BudgetSpec.from_dict({"mode": "soft", "max_tokens": 120, "breach_action": "warn"}),
    }
    return BudgetManager(budgets=budgets)


def collect_metric(outcomes: Sequence[BudgetChargeOutcome], scope: BudgetScope, key: str) -> decimal.Decimal:
    for outcome in outcomes:
        if outcome.scope == scope:
            return outcome.remaining[key]
    raise AssertionError(f"Scope {scope} not found in outcomes")


def test_hard_budget_enforces_stop(budget_manager: BudgetManager, run_scope: BudgetScope, node_scope: BudgetScope) -> None:
    cost = CostSnapshot.from_costs({"usd": "2.00", "calls": 1})
    outcomes = budget_manager.charge([run_scope, node_scope], cost)
    assert not any(outcome.breached for outcome in outcomes)
    assert collect_metric(outcomes, node_scope, "usd") == decimal.Decimal("1.00")

    # Second charge should breach the node hard limit (3 USD total)
    with pytest.raises(BudgetBreachError) as exc_info:
        budget_manager.charge([run_scope, node_scope], cost)

    error = exc_info.value
    assert error.outcome.scope == node_scope
    assert error.outcome.breached is True
    assert error.outcome.action == "error"
    assert error.outcome.overages["usd"] == decimal.Decimal("1.00")

    # Run scope should still have recorded the spend even after the exception
    preview = budget_manager.preview([run_scope], CostSnapshot.zero())
    assert collect_metric(preview, run_scope, "usd") == decimal.Decimal("1.00")


def test_soft_budget_warns_and_tracks_overage(
    budget_manager: BudgetManager,
    run_scope: BudgetScope,
    node_scope: BudgetScope,
    spec_scope: BudgetScope,
) -> None:
    big_cost = CostSnapshot.from_costs({"tokens": 140})
    outcomes = budget_manager.charge([run_scope, node_scope, spec_scope], big_cost)
    spec_outcome = next(outcome for outcome in outcomes if outcome.scope == spec_scope)

    assert spec_outcome.breached is True
    assert spec_outcome.action == "warn"
    assert spec_outcome.overages["tokens"] == decimal.Decimal(20)

    # Additional charge should accumulate overage without raising
    more_cost = CostSnapshot.from_costs({"tokens": 10})
    more_outcomes = budget_manager.charge([run_scope, spec_scope], more_cost)
    spec_outcome = next(outcome for outcome in more_outcomes if outcome.scope == spec_scope)
    assert spec_outcome.overages["tokens"] == decimal.Decimal(30)


def test_preview_does_not_mutate_state(budget_manager: BudgetManager, run_scope: BudgetScope) -> None:
    preview_cost = CostSnapshot.from_costs({"usd": "1.00"})
    preview = budget_manager.preview([run_scope], preview_cost)
    outcome = preview[0]
    assert outcome.cost.usd == decimal.Decimal("1.00")
    assert outcome.breached is False

    # After preview, a real charge should see the original limit
    budget_manager.charge([run_scope], CostSnapshot.from_costs({"usd": "1.00"}))
    updated = budget_manager.preview([run_scope], CostSnapshot.zero())
    outcome = updated[0]
    assert outcome.remaining["usd"] == decimal.Decimal("4.00")
