"""Budget model behaviour tests."""

from __future__ import annotations

import pytest

from codex.code.integrate_budget_guards_runner_p3.dsl import budget_models as bm
from codex.code.integrate_budget_guards_runner_p3.dsl.trace import TraceEventEmitter


def test_cost_snapshot_from_raw_normalises_seconds_and_tokens() -> None:
    snapshot = bm.CostSnapshot.from_raw({"time_s": 1.5, "tokens": 42})
    assert snapshot.time_ms == pytest.approx(1500.0)
    assert snapshot.tokens == 42

    empty = bm.CostSnapshot.from_raw(None)
    assert empty.time_ms == 0.0
    assert empty.tokens == 0


def test_budget_charge_outcome_compute_tracks_overage() -> None:
    spec = bm.BudgetSpec(
        name="node-hard",
        scope_type="node",
        limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
        mode="hard",
        breach_action="stop",
    )
    prior = bm.CostSnapshot.from_raw({"time_ms": 70})
    cost = bm.CostSnapshot.from_raw({"time_ms": 50})

    outcome = bm.BudgetChargeOutcome.compute(spec=spec, prior=prior, cost=cost)

    assert outcome.charge.new_total.time_ms == pytest.approx(120.0)
    assert outcome.charge.overage.time_ms == pytest.approx(20.0)
    assert outcome.breached is True
    assert outcome.should_stop is True


def test_budget_decision_emits_trace_records_and_identifies_blocking() -> None:
    run_scope = bm.ScopeKey(scope_type="run", scope_id="run-1")
    soft_spec = bm.BudgetSpec(
        name="run-soft",
        scope_type="run",
        limit=bm.CostSnapshot.from_raw({"time_ms": 60}),
        mode="soft",
        breach_action="warn",
    )
    hard_spec = bm.BudgetSpec(
        name="run-hard",
        scope_type="run",
        limit=bm.CostSnapshot.from_raw({"time_ms": 50}),
        mode="hard",
        breach_action="stop",
    )
    cost = bm.CostSnapshot.from_raw({"time_ms": 40})

    soft_outcome = bm.BudgetChargeOutcome.compute(spec=soft_spec, prior=bm.CostSnapshot.zero(), cost=cost)
    hard_outcome = bm.BudgetChargeOutcome.compute(spec=hard_spec, prior=bm.CostSnapshot.from_raw({"time_ms": 30}), cost=cost)

    decision = bm.BudgetDecision.make(scope=run_scope, cost=cost, outcomes=[soft_outcome, hard_outcome])

    assert decision.breached is True
    assert decision.should_stop is True
    assert decision.blocking is hard_outcome

    emitter = TraceEventEmitter()
    decision.to_trace_records(emitter=emitter, event="budget_charge")

    events = emitter.events
    assert len(events) == 2
    for event in events:
        with pytest.raises(TypeError):
            event.payload["spec_name"] = "mutate"  # type: ignore[index]
        assert event.payload["scope_type"] == "run"
        assert event.payload["scope_id"] == "run-1"
