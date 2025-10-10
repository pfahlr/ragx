import pytest

from pkgs.dsl.budget import BudgetBreachError, BudgetMode, BudgetSpec, CostSnapshot
from pkgs.dsl.budget_manager import BudgetManager, BudgetScope
from pkgs.dsl.trace import TraceEventEmitter


def test_budget_manager_preflight_and_commit() -> None:
    emitter = TraceEventEmitter()
    manager = BudgetManager(
        run_spec=BudgetSpec.from_mapping({"max_usd": 3.0, "mode": "hard"}),
        trace=emitter,
    )
    cost = CostSnapshot(usd=1.25)
    preview = manager.preflight(BudgetScope("run", "run"), cost)
    assert preview.remaining.usd == pytest.approx(1.75)
    commit = manager.commit(BudgetScope("run", "run"), cost)
    assert commit.spent.usd == pytest.approx(1.25)
    assert emitter.events[-1].event == "budget_commit"


def test_budget_manager_soft_warning_accumulates() -> None:
    manager = BudgetManager(run_spec=None, trace=None)
    manager.configure_scope(
        BudgetScope("node", "n1"),
        BudgetSpec.from_mapping({"max_usd": 1.0, "mode": "soft"}),
    )
    decision = manager.commit(BudgetScope("node", "n1"), CostSnapshot(usd=1.5))
    assert decision.breached
    assert manager.warnings == (decision,)


def test_budget_manager_hard_breach_raises() -> None:
    manager = BudgetManager(
        run_spec=BudgetSpec.from_mapping({"max_usd": 1.0, "mode": "hard"}),
        trace=None,
    )
    manager.commit(BudgetScope("run", "run"), CostSnapshot(usd=0.6))
    with pytest.raises(BudgetBreachError):
        manager.commit(BudgetScope("run", "run"), CostSnapshot(usd=0.5))


def test_budget_manager_loop_budget_stops_iteration() -> None:
    manager = BudgetManager(run_spec=None, trace=None)
    manager.configure_scope(
        BudgetScope("loop", "loop-1"),
        BudgetSpec.from_mapping(
            {"max_calls": 2, "breach_action": "stop", "mode": "hard"}
        ),
    )
    first = manager.commit(BudgetScope("loop", "loop-1"), CostSnapshot(calls=1))
    assert not first.should_stop
    second = manager.commit(BudgetScope("loop", "loop-1"), CostSnapshot(calls=1))
    assert second.should_stop


def test_budget_manager_time_limit_enforced() -> None:
    manager = BudgetManager(
        run_spec=BudgetSpec.from_mapping({"time_limit_sec": 1.0, "mode": "hard"}),
        trace=None,
    )
    manager.commit(BudgetScope("run", "run"), CostSnapshot(elapsed_ms=500))
    with pytest.raises(BudgetBreachError):
        manager.commit(BudgetScope("run", "run"), CostSnapshot(elapsed_ms=600))
