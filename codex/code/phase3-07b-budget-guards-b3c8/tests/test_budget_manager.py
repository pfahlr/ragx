import math
from dataclasses import FrozenInstanceError

import pytest

from dsl import budget as budget_module
from dsl.budget import (
    BreachAction,
    BudgetDecisionStatus,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    Cost,
    ScopeKey,
)


@pytest.fixture
def manager():
    specs = {
        ScopeKey(scope_type="run", scope_id="run"): BudgetSpec(
            scope_type="run",
            scope_id="run",
            limit_ms=150,
            mode=BudgetMode.SOFT,
            breach_action=BreachAction.WARN,
        ),
        ScopeKey(scope_type="loop", scope_id="loop1"): BudgetSpec(
            scope_type="loop",
            scope_id="loop1",
            limit_ms=120,
            mode=BudgetMode.SOFT,
            breach_action=BreachAction.WARN,
        ),
        ScopeKey(scope_type="node", scope_id="node1"): BudgetSpec(
            scope_type="node",
            scope_id="node1",
            limit_ms=80,
            mode=BudgetMode.HARD,
            breach_action=BreachAction.STOP,
        ),
    }
    return BudgetManager(specs)


def test_cost_normalization_supports_seconds_and_ms():
    cost_seconds = Cost.from_seconds(0.25)
    cost_ms = Cost(milliseconds=250.0)
    assert math.isclose(cost_seconds.milliseconds, 250.0)
    assert cost_seconds == cost_ms


def test_preflight_warns_for_soft_budget(manager):
    decision = manager.preflight(ScopeKey("run", "run"), Cost(milliseconds=160))
    assert decision.status is BudgetDecisionStatus.WARN
    assert decision.breach is not None
    assert decision.breach.action is BreachAction.WARN


def test_commit_updates_remaining_and_overage(manager):
    outcome = manager.commit(ScopeKey("loop", "loop1"), Cost(milliseconds=130))
    assert outcome.decision.status is BudgetDecisionStatus.WARN
    assert outcome.charge.spent_ms == pytest.approx(130.0)
    assert outcome.charge.remaining_ms == pytest.approx(0.0)
    assert outcome.charge.overage_ms == pytest.approx(10.0)
    assert outcome.breach is not None
    assert manager.remaining(ScopeKey("loop", "loop1")) == pytest.approx(0.0)


def test_hard_budget_commit_requires_stop(manager):
    manager.commit(ScopeKey("node", "node1"), Cost(milliseconds=60))
    decision = manager.preflight(ScopeKey("node", "node1"), Cost(milliseconds=30))
    assert decision.status is BudgetDecisionStatus.STOP
    outcome = manager.commit(ScopeKey("node", "node1"), Cost(milliseconds=30))
    assert outcome.decision.status is BudgetDecisionStatus.STOP
    assert outcome.requires_stop is True
    assert outcome.breach is not None
    assert outcome.charge.overage_ms == pytest.approx(10.0)


def test_unknown_scope_raises_value_error(manager):
    with pytest.raises(KeyError):
        manager.preflight(ScopeKey("node", "missing"), Cost(milliseconds=1))


def test_budget_dataclasses_are_frozen(manager):
    outcome = manager.commit(ScopeKey("loop", "loop1"), Cost(milliseconds=130))
    with pytest.raises(FrozenInstanceError):
        outcome.charge.spent_ms = 0.0  # type: ignore[misc]

    with pytest.raises(AttributeError):
        outcome.decision.new_field = 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "milliseconds, expected",
    [
        (0.0, 0.0),
        (1e-6, 0.0),
        (100.0, 100.0),
    ],
)
def test_cost_rounding(milliseconds, expected):
    assert Cost(milliseconds=milliseconds).milliseconds == pytest.approx(expected)
