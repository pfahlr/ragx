"""Trace writer abstractions reused by the budget manager and FlowRunner.

The design mirrors immutable event payloads introduced in
`codex/integrate-budget-guards-with-runner-zwi2ny` while enriching breach
metadata with the overage accounting popularised in the pbdel9 branch. Policy
hooks emit `policy_resolved` events explicitly per POSTEXECUTION directives.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType


class TraceWriter:
    """Abstract trace sink used by runtime collaborators."""

    def emit(self, event: str, payload: Mapping[str, object]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def snapshot(self) -> tuple[Mapping[str, object], ...]:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryTraceWriter(TraceWriter):
    """Simple trace writer that buffers immutable trace events in memory."""

    def __init__(self) -> None:
        self._events: list[Mapping[str, object]] = []

    def emit(self, event: str, payload: Mapping[str, object]) -> None:
        combined: MutableMapping[str, object] = {"event": event}
        for key, value in payload.items():
            combined[key] = _freeze_value(value)
        self._events.append(MappingProxyType(combined))

    def snapshot(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._events)


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        frozen_inner: MutableMapping[str, object] = {}
        for inner_key, inner_value in value.items():
            frozen_inner[str(inner_key)] = _freeze_value(inner_value)
        return MappingProxyType(frozen_inner)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PolicyTrace:
    node_id: str
    adapter: str
    run_id: str


class TraceEventEmitter:
    """High-level trace emitter shared by PolicyStack, BudgetManager, and FlowRunner."""

    def __init__(self, writer: TraceWriter) -> None:
        self._writer = writer

    def policy_push(self, node_id: str, adapter: str, run_id: str) -> None:
        self._writer.emit(
            "policy_push",
            {"node_id": node_id, "adapter": adapter, "run_id": run_id},
        )

    def policy_resolved(self, node_id: str, adapter: str, run_id: str) -> None:
        self._writer.emit(
            "policy_resolved",
            {"node_id": node_id, "adapter": adapter, "run_id": run_id},
        )

    def policy_violation(self, node_id: str, adapter: str, run_id: str, reason: str) -> None:
        self._writer.emit(
            "policy_violation",
            {
                "node_id": node_id,
                "adapter": adapter,
                "run_id": run_id,
                "reason": reason,
            },
        )

    def budget_charge(
        self,
        *,
        scope: str,
        scope_type: str,
        run_id: str,
        node_id: str,
        cost: Mapping[str, float],
        remaining: Mapping[str, float],
    ) -> None:
        self._writer.emit(
            "budget_charge",
            {
                "scope": scope,
                "scope_type": scope_type,
                "run_id": run_id,
                "node_id": node_id,
                "cost": dict(cost),
                "remaining": dict(remaining),
            },
        )

    def budget_breach(
        self,
        *,
        scope: str,
        scope_type: str,
        run_id: str,
        node_id: str,
        overages: Mapping[str, float],
        remaining: Mapping[str, float],
        severity: str,
    ) -> None:
        self._writer.emit(
            "budget_breach",
            {
                "scope": scope,
                "scope_type": scope_type,
                "run_id": run_id,
                "node_id": node_id,
                "overages": dict(overages),
                "remaining": dict(remaining),
                "severity": severity,
            },
        )

    def loop_summary(
        self,
        *,
        run_id: str,
        node_id: str,
        stop_reason: str | None,
        spent: Mapping[str, float],
    ) -> None:
        self._writer.emit(
            "loop_summary",
            {
                "run_id": run_id,
                "node_id": node_id,
                "stop_reason": stop_reason,
                "spent": dict(spent),
            },
        )
