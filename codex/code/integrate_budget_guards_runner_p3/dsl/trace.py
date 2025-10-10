"""Immutable trace event emitter used by budget and policy layers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = ["TraceEvent", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable trace record."""

    event: str
    scope_type: str
    scope_id: str
    payload: Mapping[str, Any]


class TraceEventEmitter:
    """Emit immutable trace events with optional sink forwarding."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._sink: Callable[[TraceEvent], None] | None = None

    def attach_sink(self, sink: Callable[[TraceEvent], None] | None) -> None:
        """Attach or clear an external sink that receives emitted events."""

        self._sink = sink

    def emit(
        self,
        event: str,
        *,
        scope_type: str,
        scope_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        data = dict(payload or {})
        record = TraceEvent(
            event=event,
            scope_type=scope_type,
            scope_id=scope_id,
            payload=MappingProxyType(data),
        )
        self._events.append(record)
        if self._sink is not None:
            self._sink(record)
        return record

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()
