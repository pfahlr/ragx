import pytest

from dsl.budget import (
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
    BudgetHardStop,
)
from dsl.trace import InMemoryTraceWriter, TraceEventEmitter


def make_spec(scope: str, limit: float, mode: BudgetMode, breach_action: str = "warn") -> BudgetSpec:
    return BudgetSpec(scope_id=scope, limits={"time_ms": limit}, mode=mode, breach_action=breach_action)


def make_cost(ms: float) -> CostSnapshot:
    return CostSnapshot.from_raw({"time_ms": ms})


def test_soft_budget_emits_warning_and_does_not_stop():
    writer = InMemoryTraceWriter()
    emitter = TraceEventEmitter(writer)
    manager = BudgetManager(emitter)

    spec = make_spec("run-1", 1000.0, BudgetMode.SOFT, breach_action="warn")
    decision = manager.preflight("run-1", "run-1", "run", "n1", spec, make_cost(900.0))
    outcome = manager.commit(decision)

    assert outcome.should_stop is False
    assert outcome.breach_kind == "none"

    # Second charge pushes over the limit but should warn only.
    decision_over = manager.preflight("run-1", "run-1", "run", "n2", spec, make_cost(200.0))
    outcome_over = manager.commit(decision_over)
    assert outcome_over.should_stop is False
    assert outcome_over.breach_kind == "soft"

    events = writer.snapshot()
    budget_events = [event for event in events if event["event"].startswith("budget_")]
    assert any(event["event"] == "budget_charge" for event in budget_events)
    breach_events = [event for event in budget_events if event["event"] == "budget_breach"]
    assert breach_events, "soft overage should emit budget_breach event"
    assert breach_events[-1]["severity"] == "soft"
    assert breach_events[-1]["overages"]["time_ms"] == pytest.approx(100.0)


def test_hard_budget_raises_and_emits_hard_breach():
    writer = InMemoryTraceWriter()
    emitter = TraceEventEmitter(writer)
    manager = BudgetManager(emitter)

    spec = make_spec("run-2", 100.0, BudgetMode.HARD, breach_action="stop")
    decision = manager.preflight("run-2", "run-2", "run", "n1", spec, make_cost(150.0))

    with pytest.raises(BudgetHardStop):
        manager.commit(decision)

    events = writer.snapshot()
    breach = [event for event in events if event["event"] == "budget_breach"][-1]
    assert breach["severity"] == "hard"
    assert breach["overages"]["time_ms"] == pytest.approx(50.0)
    # Ensure the account was not mutated after the hard stop
    decision_retry = manager.preflight("run-2", "run-2", "run", "n2", spec, make_cost(50.0))
    assert decision_retry.spent.get("time_ms", 0.0) == pytest.approx(0.0)
    assert decision_retry.should_stop is False
