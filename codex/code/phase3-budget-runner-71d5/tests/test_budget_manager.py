import pytest

from phase3_budget_runner_71d5.budgeting import (
    BreachAction,
    BudgetContext,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
    ScopeKey,
    ScopeType,
)
from phase3_budget_runner_71d5.trace import TraceEventEmitter, TraceRecorder


@pytest.fixture()
def manager():
    recorder = TraceRecorder()
    emitter = TraceEventEmitter(recorder=recorder)
    mgr = BudgetManager(trace_emitter=emitter)
    return mgr, recorder


def test_hard_budget_triggers_stop_and_emits_breach(manager):
    mgr, recorder = manager
    run_scope = ScopeKey(ScopeType.RUN, "run-1")
    node_scope = ScopeKey(ScopeType.NODE, "node-1")

    mgr.enter_scope(run_scope, BudgetSpec("run", CostSnapshot.from_raw({"tokens": 100})))
    mgr.enter_scope(
        node_scope,
        BudgetSpec("node", CostSnapshot.from_raw({"tokens": 40})),
        parent=run_scope,
    )

    ctx = BudgetContext(run=run_scope, node=node_scope)
    preview = mgr.preview(ctx, CostSnapshot.from_raw({"tokens": 30}), label="estimate")
    assert preview.should_stop is False

    commit = mgr.commit(ctx, CostSnapshot.from_raw({"tokens": 50}), label="execute")
    assert commit.should_stop is True
    node_outcome = commit.outcomes[node_scope]
    assert pytest.approx(node_outcome.spent.metrics["tokens"], rel=1e-6) == 50
    assert node_outcome.breached is True
    assert pytest.approx(node_outcome.overages.metrics["tokens"], rel=1e-6) == 10
    assert any(evt.event == "budget_breach" for evt in recorder.events)


def test_soft_warn_allows_progress_and_tracks_warnings(manager):
    mgr, recorder = manager
    run_scope = ScopeKey(ScopeType.RUN, "run-soft")
    loop_scope = ScopeKey(ScopeType.LOOP, "loop-1")

    mgr.enter_scope(
        run_scope,
        BudgetSpec(
            "run",
            CostSnapshot.from_raw({"tokens": 200}),
            mode=BudgetMode.SOFT,
            breach_action=BreachAction.WARN,
        ),
    )
    mgr.enter_scope(
        loop_scope,
        BudgetSpec(
            "loop",
            CostSnapshot.from_raw({"tokens": 60}),
            mode=BudgetMode.SOFT,
            breach_action=BreachAction.WARN,
        ),
        parent=run_scope,
    )

    ctx = BudgetContext(run=run_scope, loop=loop_scope)

    first_charge = mgr.commit(ctx, CostSnapshot.from_raw({"tokens": 50}), label="iteration-1")
    assert first_charge.should_stop is False
    assert not first_charge.outcomes[loop_scope].breached

    second_charge = mgr.commit(ctx, CostSnapshot.from_raw({"tokens": 20}), label="iteration-2")
    assert second_charge.should_stop is False
    warnings = second_charge.outcomes[loop_scope].warnings
    assert any("breached" in warning for warning in warnings)

    breach_events = [evt for evt in recorder.events if evt.event == "budget_breach"]
    assert len(breach_events) == 1
    assert breach_events[0].payload["scope_type"] == "loop"
