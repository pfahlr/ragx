from __future__ import annotations

from types import MappingProxyType

import pytest

from pkgs.dsl.budget import BreachAction, BudgetDecision
from pkgs.dsl.trace import TraceEventEmitter


def make_decision(scope_id: str = "run-1", overage: float = 0.0, remaining: float = 50.0, action: BreachAction = BreachAction.WARN):
    return BudgetDecision(
        scope_id=scope_id,
        should_stop=action == BreachAction.STOP,
        action=action,
        remaining_ms=remaining,
        overage_ms=overage,
    )


def test_budget_charge_payload_is_immutable(trace_collector) -> None:
    writer, events = trace_collector
    emitter = TraceEventEmitter(writer)
    emitter.emit_budget_charge(make_decision(overage=5.0, remaining=45.0))
    assert events[0][0] == "budget_charge"
    payload = events[0][1]
    assert isinstance(payload, MappingProxyType)
    assert payload["scope_id"] == "run-1"
    assert payload["overage_ms"] == 5.0
    with pytest.raises(TypeError):
        payload["overage_ms"] = 10


def test_budget_breach_marks_breach_event(trace_collector) -> None:
    writer, events = trace_collector
    emitter = TraceEventEmitter(writer)
    emitter.emit_budget_breach(make_decision(scope_id="node-1", overage=12.0, remaining=-2.0, action=BreachAction.STOP))
    assert events[0][0] == "budget_breach"
    payload = events[0][1]
    assert payload["scope_id"] == "node-1"
    assert payload["breach_action"] == "stop"
    assert payload["overage_ms"] == 12.0


def test_policy_events_passthrough_payload(trace_collector) -> None:
    writer, events = trace_collector
    emitter = TraceEventEmitter(writer)
    emitter.emit_policy_event("policy_push", scope="node-2", payload={"policy": "allow"})
    assert events[0][0] == "policy_push"
    payload = events[0][1]
    assert isinstance(payload, MappingProxyType)
    assert payload["scope"] == "node-2"
    assert payload["policy"] == "allow"

