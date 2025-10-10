"""Structured trace emission helpers for FlowRunner budget integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from pkgs.dsl.policy import PolicyDenial, PolicyTraceEvent

from .budget_models import BudgetCharge, BudgetPreview, mapping_proxy

__all__ = ["TraceEventEmitter"]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(val) for key, val in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class TraceEventEmitter:
    """Emit immutable trace payloads for budgets and policies."""

    def __init__(
        self,
        *,
        writer: Callable[[Mapping[str, Any]], None],
        flow_id: str,
        run_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._writer = writer
        self._flow_id = flow_id
        self._run_id = run_id
        self._metadata = mapping_proxy({"flow_id": flow_id, "run_id": run_id, **(metadata or {})})

    # ------------------------------------------------------------------
    # Policy integration helpers
    # ------------------------------------------------------------------
    @property
    def policy_event_sink(self) -> Callable[[PolicyTraceEvent], None]:
        def sink(event: PolicyTraceEvent) -> None:
            payload = {
                "event": event.event,
                "scope": event.scope,
                "data": dict(event.data),
            }
            self._emit(event.event, payload)

        return sink

    def emit_policy_resolved(self, *, node_id: str, resolution: Mapping[str, Any]) -> None:
        payload = {
            "event": "policy_resolved",
            "scope": "node",
            "node_id": node_id,
            "allowed": sorted(resolution["allowed"]),
            "denied": {name: list(reasons) for name, reasons in resolution["denied"].items()},
            "policy_stack_depth": resolution.get("stack_depth", 0),
        }
        self._emit("policy_resolved", payload)

    def emit_policy_violation(self, *, node_id: str, denial: PolicyDenial) -> None:
        payload = {
            "event": "policy_violation",
            "scope": denial.decision.denied_by or denial.decision.granted_by or "stack",
            "node_id": node_id,
            "tool": denial.tool,
            "reasons": list(denial.reasons),
            "policy_stack_depth": denial.decision.stack_depth if hasattr(denial.decision, "stack_depth") else 0,
        }
        self._emit("policy_violation", payload)

    # ------------------------------------------------------------------
    # Budget events
    # ------------------------------------------------------------------
    def emit_budget_charge(
        self,
        *,
        node_id: str,
        loop_iteration: int,
        charge: BudgetCharge,
    ) -> None:
        payload = {
            "event": "budget_charge",
            "scope": charge.scope_type,
            "scope_id": charge.scope_id,
            "node_id": node_id,
            "loop_iteration": loop_iteration,
            "cost": dict(charge.cost.metrics),
            "remaining": dict(charge.remaining),
            "overages": dict(charge.overages),
            "breached": charge.breached,
            "breach_action": charge.mode.value,
        }
        self._emit("budget_charge", payload)

    def emit_budget_breach(
        self,
        *,
        node_id: str,
        loop_iteration: int,
        preview: BudgetPreview,
        charge: BudgetCharge | None = None,
    ) -> None:
        breached_charges = [charge] if charge is not None else [
            candidate for candidate in preview.charges if candidate.breached
        ]
        payload = {
            "event": "budget_breach",
            "node_id": node_id,
            "loop_iteration": loop_iteration,
            "breach_action": breached_charges[0].mode.value if breached_charges else "unknown",
            "scope_chain": [
                {
                    "scope_id": breached.scope_id,
                    "scope_type": breached.scope_type,
                    "breach_action": breached.mode.value,
                    "overages": dict(breached.overages),
                }
                for breached in breached_charges
            ],
        }
        self._emit("budget_breach", payload)

    def emit_loop_stop(
        self,
        *,
        scope: str,
        node_id: str,
        loop_iteration: int,
        stop_reason: str,
    ) -> None:
        payload = {
            "event": "loop_stop",
            "scope": scope,
            "node_id": node_id,
            "loop_iteration": loop_iteration,
            "stop_reason": stop_reason,
        }
        self._emit("loop_stop", payload)

    # ------------------------------------------------------------------
    # Core writer helper
    # ------------------------------------------------------------------
    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        envelope = {
            "event": event,
            "flow_id": self._flow_id,
            "run_id": self._run_id,
            "metadata": self._metadata,
        }
        envelope.update(payload)
        self._writer(_freeze(envelope))

