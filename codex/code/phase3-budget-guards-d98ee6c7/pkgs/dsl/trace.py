"""Trace utilities for the Phase 3 sandbox implementation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .budget import BudgetDecision


class TraceWriter(Protocol):
    """Protocol for writing trace events."""

    def write(self, event: str, payload: Mapping[str, Any]) -> None:
        ...


def _immutable(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


class TraceEventEmitter:
    """Emit immutable trace payloads through a TraceWriter."""

    def __init__(self, writer: TraceWriter) -> None:
        self._writer = writer

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        self._writer.write(event, _immutable(payload))

    def emit_budget_charge(self, decision: BudgetDecision) -> None:
        payload = {
            "scope_id": decision.scope_id,
            "remaining_ms": decision.remaining_ms,
            "overage_ms": decision.overage_ms,
            "action": decision.action.value,
            "should_stop": decision.should_stop,
        }
        self.emit("budget_charge", payload)

    def emit_budget_breach(self, decision: BudgetDecision) -> None:
        payload = {
            "scope_id": decision.scope_id,
            "remaining_ms": decision.remaining_ms,
            "overage_ms": decision.overage_ms,
            "breach_action": decision.action.value,
            "should_stop": decision.should_stop,
        }
        self.emit("budget_breach", payload)

    def emit_policy_event(self, event: str, scope: str, payload: Mapping[str, Any]) -> None:
        enriched = {"scope": scope, "event": event, **payload}
        self.emit(event, enriched)
