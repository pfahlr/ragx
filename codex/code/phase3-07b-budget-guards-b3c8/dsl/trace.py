"""Trace emission helpers for the sandbox FlowRunner."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, List, Mapping

from .budget import BudgetBreach, BudgetChargeOutcome, BudgetDecisionStatus, ScopeKey


@dataclass(frozen=True)
class TraceEvent:
    """Immutable trace event captured for assertions."""

    event: str
    scope: str
    payload: Mapping[str, object]


class TraceEventEmitter:
    """Collects trace events with immutable payloads."""

    def __init__(self) -> None:
        self._events: List[TraceEvent] = []

    def emit_policy_push(self, node_id: str) -> None:
        self._emit("policy_push", scope=node_id, payload={})

    def emit_policy_resolved(self, node_id: str, policy_name: str | None = None) -> None:
        payload: dict[str, object] = {}
        if policy_name:
            payload["policy"] = policy_name
        self._emit("policy_resolved", scope=node_id, payload=payload)

    def emit_policy_pop(self, node_id: str) -> None:
        self._emit("policy_pop", scope=node_id, payload={})

    def emit_policy_violation(self, node_id: str, reason: str | None = None) -> None:
        payload = {} if reason is None else {"reason": reason}
        self._emit("policy_violation", scope=node_id, payload=payload)

    def emit_budget_charge(self, outcome: BudgetChargeOutcome) -> None:
        charge_payload = dict(outcome.charge.as_payload())
        charge_payload.update(
            {
                "scope": str(outcome.charge.scope),
                "decision_status": outcome.decision.status.value,
            }
        )
        if outcome.breach:
            charge_payload["breach_kind"] = outcome.breach.breach_kind
            charge_payload["action"] = outcome.breach.action.value
        self._emit("budget_charge", scope=str(outcome.charge.scope), payload=charge_payload)

    def emit_budget_breach(self, breach: BudgetBreach) -> None:
        payload = {
            "scope": str(breach.scope),
            "breach_kind": breach.breach_kind,
            "limit_ms": breach.limit_ms,
            "attempted_ms": breach.attempted_ms,
            "action": breach.action.value,
        }
        self._emit("budget_breach", scope=str(breach.scope), payload=payload)

    def emit_loop_summary(
        self,
        loop: ScopeKey,
        iterations_completed: int,
        stop_reason: str | None,
    ) -> None:
        payload: dict[str, object] = {
            "loop_id": loop.scope_id,
            "iterations_completed": iterations_completed,
        }
        if stop_reason:
            payload["stop_reason"] = stop_reason
        self._emit("loop_summary", scope=str(loop), payload=payload)

    def _emit(self, event: str, scope: str, payload: Mapping[str, object]) -> None:
        immutable_payload = MappingProxyType(dict(payload))
        self._events.append(TraceEvent(event=event, scope=scope, payload=immutable_payload))

    @property
    def events(self) -> Iterable[TraceEvent]:
        return tuple(self._events)

    def reset(self) -> None:  # pragma: no cover - utility for manual testing
        self._events.clear()
