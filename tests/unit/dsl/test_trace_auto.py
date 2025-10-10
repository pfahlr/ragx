"""Trace emitter regression coverage derived from Codex P3 reviews."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetBreachError, BudgetManager
from pkgs.dsl.trace import TraceEventEmitter


@pytest.fixture()
def emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


def _manager_with_events(emitter: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 10}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 5}),
            mode="hard",
            breach_action="stop",
        ),
    ]
    manager = BudgetManager(specs=specs, trace=emitter)
    run_scope = bm.ScopeKey(scope_type="run", scope_id="run-1")
    node_scope = bm.ScopeKey(scope_type="node", scope_id="node-1")
    manager.enter_scope(run_scope)
    manager.enter_scope(node_scope)

    cost = bm.CostSnapshot.from_raw({"time_ms": 6})
    decision = manager.preview_charge(node_scope, cost)
    manager.record_breach(decision)
    with pytest.raises(BudgetBreachError):
        manager.commit_charge(decision)
    return manager


def test_trace_payload_schema_validation(emitter: TraceEventEmitter) -> None:
    _manager_with_events(emitter)
    for event in emitter.events:
        assert set(event.payload.keys()) == {
            "scope_type",
            "scope_id",
            "spec_name",
            "mode",
            "breach_action",
            "breached",
            "should_stop",
            "cost",
            "prior",
            "new_total",
            "remaining",
            "overage",
        }
        for key in ("cost", "prior", "new_total", "remaining", "overage"):
            assert {"time_ms", "tokens"} == set(event.payload[key].keys())


def test_trace_writer_snapshot_returns_deeply_immutable_payloads(
    emitter: TraceEventEmitter,
) -> None:
    _manager_with_events(emitter)
    event = emitter.events[0]
    assert isinstance(event.payload, MappingProxyType)
    inner = event.payload["cost"]
    assert isinstance(inner, MappingProxyType)
    with pytest.raises(TypeError):
        inner["time_ms"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        event.payload["cost"] = MappingProxyType(  # type: ignore[index]
            {"time_ms": 1.0, "tokens": 0}
        )


def test_trace_emitter_sink_error_handling(emitter: TraceEventEmitter) -> None:
    calls: list[str] = []

    def sink(_: object) -> None:
        calls.append("called")
        raise RuntimeError("sink failure")

    emitter.attach_sink(sink)
    with pytest.raises(RuntimeError):
        emitter.emit("run_start", scope_type="run", scope_id="r1")
    assert calls == ["called"]


def test_trace_validator_error_context(emitter: TraceEventEmitter) -> None:
    captured: list[str] = []

    def validator(event) -> None:
        if event.event == "budget_breach":
            captured.append(event.scope_id)
            raise ValueError("invalid payload")

    emitter.attach_validator(validator)
    specs = [
        bm.BudgetSpec(
            name="run",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 10}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 5}),
            mode="hard",
            breach_action="stop",
        ),
    ]
    manager = BudgetManager(specs=specs, trace=emitter)
    run_scope = bm.ScopeKey(scope_type="run", scope_id="run-val")
    node_scope = bm.ScopeKey(scope_type="node", scope_id="node-val")
    manager.enter_scope(run_scope)
    manager.enter_scope(node_scope)
    cost = bm.CostSnapshot.from_raw({"time_ms": 7})
    decision = manager.preview_charge(node_scope, cost)
    with pytest.raises(ValueError):
        manager.record_breach(decision)
    assert captured == ["node-val"]


def test_trace_emitter_clear_resets_events(emitter: TraceEventEmitter) -> None:
    emitter.emit("run_start", scope_type="run", scope_id="r1")
    assert emitter.events
    emitter.clear()
    assert emitter.events == ()
