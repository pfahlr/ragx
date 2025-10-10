"""Unit tests validating the BudgetManager lifecycle."""

from __future__ import annotations

import pytest

from . import load_module

budget_models = load_module("budget_models")
budget_manager_mod = load_module("budget_manager")
trace_mod = load_module("trace_emitter")

BudgetSpec = budget_models.BudgetSpec
BudgetMode = budget_models.BudgetMode
CostSnapshot = budget_models.CostSnapshot
BudgetManager = budget_manager_mod.BudgetManager
BudgetBreachError = budget_manager_mod.BudgetBreachError
TraceEventEmitter = trace_mod.TraceEventEmitter


class EventCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def __call__(self, event: dict[str, object]) -> None:
        self.events.append(event)


class TestBudgetManager:
    def setup_method(self) -> None:
        self.collector = EventCollector()
        self.emitter = TraceEventEmitter(
            writer=self.collector,
            flow_id="flow-123",
            run_id="run-xyz",
            metadata={"environment": "test"},
        )
        self.manager = BudgetManager(emitter=self.emitter)

    def teardown_method(self) -> None:
        # Ensure the root scope is cleared between tests if it was opened.
        for scope_id in list(self.manager.active_scope_ids):
            self.manager.exit_scope(scope_id)

    def _open_run_scope(self, *, limit: float, mode: BudgetMode = BudgetMode.STOP) -> str:
        scope_id = "run-scope"
        self.manager.enter_scope(
            scope_type="run",
            scope_id=scope_id,
            spec=BudgetSpec(limits={"tokens": limit}, mode=mode),
        )
        return scope_id

    def _open_node_scope(
        self, parent_scope: str, *, limit: float, mode: BudgetMode = BudgetMode.STOP
    ) -> str:
        scope_id = "node-1"
        self.manager.enter_scope(
            scope_type="node",
            scope_id=scope_id,
            spec=BudgetSpec(limits={"tokens": limit}, mode=mode),
            parent_scope=parent_scope,
        )
        return scope_id

    def test_commit_updates_remaining_and_emits_charge(self) -> None:
        run_scope = self._open_run_scope(limit=100)
        node_scope = self._open_node_scope(run_scope, limit=60)
        preview = self.manager.preview(node_scope, {"tokens": 10})
        assert not preview.hard_breach

        self.manager.commit(
            preview,
            node_id=node_scope,
            loop_iteration=0,
        )

        # Expect two charge events: node and run scope.
        charge_events = [e for e in self.collector.events if e["event"] == "budget_charge"]
        assert len(charge_events) == 2
        for event in charge_events:
            assert event["cost"]["tokens"] == pytest.approx(10.0)
            assert event["remaining"]["tokens"] >= 0.0

    def test_warn_mode_emits_breach_without_raising(self) -> None:
        run_scope = self._open_run_scope(limit=35, mode=BudgetMode.WARN)
        node_scope = self._open_node_scope(run_scope, limit=30, mode=BudgetMode.WARN)

        preview = self.manager.preview(node_scope, {"tokens": 45})
        assert preview.has_breach

        self.manager.commit(
            preview,
            node_id=node_scope,
            loop_iteration=1,
        )

        breach_events = [e for e in self.collector.events if e["event"] == "budget_breach"]
        assert len(breach_events) == 2
        assert all(event["breach_action"] == "warn" for event in breach_events)

    def test_stop_mode_raises_on_breach(self) -> None:
        run_scope = self._open_run_scope(limit=18, mode=BudgetMode.STOP)
        node_scope = self._open_node_scope(run_scope, limit=20, mode=BudgetMode.STOP)

        preview = self.manager.preview(node_scope, {"tokens": 25})
        assert preview.hard_breach

        with pytest.raises(BudgetBreachError):
            self.manager.commit(
                preview,
                node_id=node_scope,
                loop_iteration=2,
            )

        breach_events = [e for e in self.collector.events if e["event"] == "budget_breach"]
        assert len(breach_events) == 2
        assert all(event["breach_action"] == "stop" for event in breach_events)

    def test_remaining_budget_snapshot(self) -> None:
        run_scope = self._open_run_scope(limit=100)
        node_scope = self._open_node_scope(run_scope, limit=30)
        preview = self.manager.preview(node_scope, {"tokens": 10})
        self.manager.commit(preview, node_id=node_scope, loop_iteration=0)

        remaining = self.manager.remaining_budget(node_scope)
        assert isinstance(remaining, CostSnapshot)
        assert remaining.metrics["tokens"] == pytest.approx(20.0)

