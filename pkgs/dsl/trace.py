"""Trace emission utilities shared by FlowRunner collaborators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .budget import BudgetCharge
from .models import PolicyResolution, mapping_proxy

__all__ = ["TraceEvent", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Structured event emitted by the FlowRunner."""

    event: str
    scope: str
    data: Mapping[str, object]


class TraceEventEmitter:
    """Collect and emit immutable trace events."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def reset(self) -> None:
        """Clear previously captured events."""

        self._events.clear()

    def emit(self, event: str, scope: str, payload: Mapping[str, object]) -> None:
        record = TraceEvent(event=event, scope=scope, data=mapping_proxy(payload))
        self._events.append(record)

    def budget_charge(
        self,
        charge: BudgetCharge,
        *,
        loop_iteration: int | None,
        run_id: str,
    ) -> None:
        payload: dict[str, object] = {
            "run_id": run_id,
            "scope_kind": charge.scope.kind,
            "scope_id": charge.scope.identifier,
            "mode": charge.spec.mode.value,
            "breach_action": charge.action,
            "cost": dict(charge.cost),
            "remaining": dict(charge.remaining),
        }
        if charge.overages:
            payload["overages"] = dict(charge.overages)
        if loop_iteration is not None:
            payload["loop_iteration"] = loop_iteration
        self.emit("budget_charge", f"{charge.scope.kind}:{charge.scope.identifier}", payload)

    def budget_breach(
        self,
        charge: BudgetCharge,
        *,
        loop_iteration: int | None,
        run_id: str,
        stop_reason: str,
    ) -> None:
        payload: dict[str, object] = {
            "run_id": run_id,
            "scope_kind": charge.scope.kind,
            "scope_id": charge.scope.identifier,
            "breach_action": charge.action,
            "remaining": dict(charge.remaining),
            "overages": dict(charge.overages),
            "stop_reason": stop_reason,
        }
        if loop_iteration is not None:
            payload["loop_iteration"] = loop_iteration
        self.emit("budget_breach", f"{charge.scope.kind}:{charge.scope.identifier}", payload)

    def policy_resolved(
        self,
        node_id: str,
        resolution: PolicyResolution,
        *,
        run_id: str,
        iteration: int | None,
    ) -> None:
        payload: dict[str, object] = {
            "run_id": run_id,
            "node_id": node_id,
            "allowed": tuple(sorted(resolution.allowed)),
            "denied": {name: tuple(reasons) for name, reasons in resolution.denied.items()},
            "stack_depth": resolution.stack_depth,
        }
        if iteration is not None:
            payload["loop_iteration"] = iteration
        self.emit("policy_resolved", f"node:{node_id}", payload)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Expose captured events as an immutable sequence."""

        return tuple(self._events)

    def extend(self, events: Iterable[TraceEvent]) -> None:  # pragma: no cover - future use
        self._events.extend(events)
