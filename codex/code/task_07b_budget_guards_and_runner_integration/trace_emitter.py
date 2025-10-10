"""Trace emitter used by FlowRunner and BudgetManager tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence


@dataclass(frozen=True)
class TraceEvent:
    """Immutable trace event."""

    event: str
    scope_type: str
    scope_id: str
    payload: Mapping[str, Any]


class TraceEventEmitter:
    """Collects trace events in the order they were emitted."""

    def __init__(self) -> None:
        self._events: List[TraceEvent] = []

    def emit(self, event: str, *, scope_type: str, scope_id: str, payload: Mapping[str, Any]) -> TraceEvent:
        trace = TraceEvent(event=event, scope_type=scope_type, scope_id=scope_id, payload=dict(payload))
        self._events.append(trace)
        return trace

    def get_events(self) -> Sequence[TraceEvent]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()


__all__ = ["TraceEvent", "TraceEventEmitter"]
