"""BudgetManager orchestration tests."""

from __future__ import annotations

import pytest

from codex.code.integrate_budget_guards_runner_p3.dsl import budget_models as bm
from codex.code.integrate_budget_guards_runner_p3.dsl.budget_manager import (
    BudgetBreachError,
    BudgetManager,
)
from codex.code.integrate_budget_guards_runner_p3.dsl.trace import TraceEventEmitter


@pytest.fixture()
def trace_emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


@pytest.fixture()
def budget_specs() -> list[bm.BudgetSpec]:
    return [
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 60}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 60}),
            mode="hard",
            breach_action="stop",
        ),
    ]


@pytest.fixture()
def manager(trace_emitter: TraceEventEmitter, budget_specs: list[bm.BudgetSpec]) -> BudgetManager:
    return BudgetManager(specs=budget_specs, trace=trace_emitter)


def test_commit_records_spend_and_emits_trace(manager: BudgetManager, trace_emitter: TraceEventEmitter) -> None:
    scope = bm.ScopeKey("run", "run-1")
    manager.enter_scope(scope)
    cost = bm.CostSnapshot.from_raw({"time_ms": 20})

    decision = manager.preview_charge(scope, cost)
    assert decision.should_stop is False

    manager.commit_charge(decision)
    remaining = manager.spent(scope, "run-soft")
    assert remaining.time_ms == pytest.approx(20.0)

    events = [evt for evt in trace_emitter.events if evt.event == "budget_charge"]
    assert len(events) == 2

    manager.exit_scope(scope)


def test_commit_raises_on_blocking_outcome(manager: BudgetManager, trace_emitter: TraceEventEmitter) -> None:
    scope = bm.ScopeKey("run", "run-stop")
    manager.enter_scope(scope)
    cost = bm.CostSnapshot.from_raw({"time_ms": 80})

    decision = manager.preview_charge(scope, cost)
    assert decision.should_stop is True

    manager.record_breach(decision)
    breach_events = [evt for evt in trace_emitter.events if evt.event == "budget_breach"]
    assert len(breach_events) == 2

    with pytest.raises(BudgetBreachError) as excinfo:
        manager.commit_charge(decision)
    assert excinfo.value.scope == scope

    manager.exit_scope(scope)


def test_spent_history_available_after_exit(manager: BudgetManager) -> None:
    scope = bm.ScopeKey("run", "history")
    manager.enter_scope(scope)
    cost = bm.CostSnapshot.from_raw({"time_ms": 30})
    decision = manager.preview_charge(scope, cost)
    manager.commit_charge(decision)
    manager.exit_scope(scope)

    spent = manager.spent(scope, "run-soft")
    assert spent.time_ms == pytest.approx(30.0)
