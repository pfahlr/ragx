"""Trace event helpers shared between policy and budget layers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, MutableMapping, Sequence

from .budgeting import BudgetScope

__all__ = ["TraceEvent", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable trace event record."""

    event: str
    scope_type: str
    scope_id: str
    payload: Mapping[str, object]


class TraceEventEmitter:
    """Collect and emit trace events for tests and integrations."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    # ------------------------------------------------------------------
    # Budget events
    # ------------------------------------------------------------------
    def emit_budget_event(
        self,
        event: str,
        scope: BudgetScope,
        payload: Mapping[str, object],
    ) -> TraceEvent:
        enriched: MutableMapping[str, object] = {
            "scope_type": scope.scope_type,
            "scope_id": scope.identifier,
        }
        enriched.update(payload)
        record = TraceEvent(
            event=event,
            scope_type=scope.scope_type,
            scope_id=scope.identifier,
            payload=MappingProxyType(dict(enriched)),
        )
        self._events.append(record)
        return record

    # ------------------------------------------------------------------
    # Policy events
    # ------------------------------------------------------------------
    def emit_policy_event(
        self,
        event: str,
        *,
        scope: str,
        payload: Mapping[str, object],
    ) -> TraceEvent:
        record = TraceEvent(
            event=event,
            scope_type="policy",
            scope_id=scope,
            payload=MappingProxyType(dict(payload)),
        )
        self._events.append(record)
        return record

    # ------------------------------------------------------------------
    # Loop events
    # ------------------------------------------------------------------
    def emit_loop_stop(
        self,
        scope: BudgetScope,
        *,
        reason: str,
        iterations: int,
    ) -> TraceEvent:
        record = TraceEvent(
            event="loop_stop",
            scope_type=scope.scope_type,
            scope_id=scope.identifier,
            payload=MappingProxyType(
                {
                    "reason": reason,
                    "iterations": iterations,
                }
            ),
        )
        self._events.append(record)
        return record

    @property
    def events(self) -> Sequence[TraceEvent]:
        return tuple(self._events)
