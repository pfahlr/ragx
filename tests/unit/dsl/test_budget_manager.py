import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetBreachError, BudgetManager
from pkgs.dsl.trace import TraceEventEmitter


@pytest.fixture()
def trace_emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


def make_manager(emitter: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 150}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    return BudgetManager(specs=specs, trace=emitter)


class TestBudgetManager:
    def test_preview_and_commit_updates_spend(self, trace_emitter: TraceEventEmitter) -> None:
        manager = make_manager(trace_emitter)
        scope = bm.ScopeKey(scope_type="run", scope_id="run-1")
        manager.enter_scope(scope)
        decision = manager.preview_charge(scope, bm.CostSnapshot.from_raw({"time_ms": 40}))
        assert decision.breached is False
        manager.commit_charge(decision)
        assert manager.spent(scope, "run-hard").time_ms == pytest.approx(40.0)
        assert manager.spent(scope, "run-soft").time_ms == pytest.approx(40.0)
        events = trace_emitter.events
        assert events[-1].event == "budget_charge"
        assert events[-1].payload["spec_name"] == "run-soft"

    def test_hard_breach_raises_and_emits(self, trace_emitter: TraceEventEmitter) -> None:
        manager = make_manager(trace_emitter)
        scope = bm.ScopeKey(scope_type="run", scope_id="run-99")
        manager.enter_scope(scope)
        manager.commit_charge(
            manager.preview_charge(scope, bm.CostSnapshot.from_raw({"time_ms": 90}))
        )
        decision = manager.preview_charge(scope, bm.CostSnapshot.from_raw({"time_ms": 20}))
        assert decision.should_stop is True
        manager.record_breach(decision)
        with pytest.raises(BudgetBreachError):
            manager.commit_charge(decision)
        events = trace_emitter.events
        assert any(
            evt.event == "budget_breach" and evt.payload["spec_name"] == "run-hard"
            for evt in events
        )

    def test_soft_breach_warns_but_commits(self, trace_emitter: TraceEventEmitter) -> None:
        manager = BudgetManager(
            specs=[
                bm.BudgetSpec(
                    name="run-soft",
                    scope_type="run",
                    limit=bm.CostSnapshot.from_raw({"time_ms": 150}),
                    mode="soft",
                    breach_action="warn",
                )
            ],
            trace=trace_emitter,
        )
        scope = bm.ScopeKey(scope_type="run", scope_id="run-2")
        manager.enter_scope(scope)
        manager.commit_charge(
            manager.preview_charge(scope, bm.CostSnapshot.from_raw({"time_ms": 150}))
        )
        decision = manager.preview_charge(scope, bm.CostSnapshot.from_raw({"time_ms": 10}))
        assert decision.breached is True
        assert decision.should_stop is False
        manager.record_breach(decision)
        manager.commit_charge(decision)
        assert manager.spent(scope, "run-soft").time_ms == pytest.approx(160.0)
        events = [evt.event for evt in trace_emitter.events]
        assert "budget_breach" in events
        assert events.count("budget_charge") >= 2

    def test_exit_scope_validates(self, trace_emitter: TraceEventEmitter) -> None:
        manager = make_manager(trace_emitter)
        scope = bm.ScopeKey(scope_type="run", scope_id="run-3")
        manager.enter_scope(scope)
        manager.exit_scope(scope)
        with pytest.raises(KeyError):
            manager.exit_scope(scope)
