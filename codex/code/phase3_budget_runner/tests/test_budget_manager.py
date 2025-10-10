import pytest

from codex.code.phase3_budget_runner.dsl import budget_manager as m
from codex.code.phase3_budget_runner.dsl import budget_models as bm
from codex.code.phase3_budget_runner.dsl.trace import TraceEventEmitter


def make_manager(*specs: bm.BudgetSpec) -> tuple[m.BudgetManager, TraceEventEmitter]:
    emitter = TraceEventEmitter()
    manager = m.BudgetManager(specs=specs, trace=emitter)
    return manager, emitter


def test_budget_manager_scope_lifecycle_and_charge_tracing():
    run_spec = bm.BudgetSpec(
        name="latency",
        scope_type="run",
        limit=bm.CostSnapshot(time_ms=200.0, tokens=200),
        mode="hard",
        breach_action="stop",
    )
    manager, emitter = make_manager(run_spec)
    run_scope = bm.ScopeKey(scope_type="run", scope_id="run-42")

    manager.enter_scope(run_scope)
    with pytest.raises(KeyError):
        manager.enter_scope(run_scope)

    decision = manager.preview_charge(run_scope, bm.CostSnapshot(time_ms=40.0, tokens=20))
    assert decision.should_stop is False
    manager.commit_charge(decision)

    spent = manager.spent(run_scope, "latency")
    assert spent.time_ms == pytest.approx(40.0)
    assert spent.tokens == 20

    events = emitter.events
    assert len(events) == 1
    assert events[0].event == "budget_charge"
    assert events[0].payload["spec_name"] == "latency"

    manager.exit_scope(run_scope)
    with pytest.raises(KeyError):
        manager.exit_scope(run_scope)

    # history lookup still works after exit
    spent_after = manager.spent(run_scope, "latency")
    assert spent_after.time_ms == pytest.approx(40.0)


def test_budget_manager_warns_on_soft_breach_and_emits_breach_trace():
    soft_spec = bm.BudgetSpec(
        name="tokens",
        scope_type="run",
        limit=bm.CostSnapshot(time_ms=0.0, tokens=50),
        mode="soft",
        breach_action="warn",
    )
    manager, emitter = make_manager(soft_spec)
    scope = bm.ScopeKey(scope_type="run", scope_id="run-warn")
    manager.enter_scope(scope)

    decision = manager.preview_charge(scope, bm.CostSnapshot(time_ms=0.0, tokens=80))
    assert decision.breached is True
    assert decision.should_stop is False

    manager.record_breach(decision)
    manager.commit_charge(decision)

    events = [event.event for event in emitter.events]
    assert events == ["budget_breach", "budget_charge"]


def test_budget_manager_hard_stop_raises_and_preserves_trace():
    hard_spec = bm.BudgetSpec(
        name="latency",
        scope_type="node",
        limit=bm.CostSnapshot(time_ms=30.0, tokens=0),
        mode="hard",
        breach_action="stop",
    )
    manager, emitter = make_manager(hard_spec)
    scope = bm.ScopeKey(scope_type="node", scope_id="n-1")
    manager.enter_scope(scope)

    decision = manager.preview_charge(scope, bm.CostSnapshot(time_ms=35.0, tokens=0))
    assert decision.should_stop is True

    manager.record_breach(decision)
    with pytest.raises(m.BudgetBreachError) as exc:
        manager.commit_charge(decision)

    assert "latency" in str(exc.value)
    assert exc.value.scope == scope
    assert exc.value.outcome.spec is hard_spec

    events = emitter.events
    assert events[0].event == "budget_breach"
    assert events[0].payload["breached"] is True

