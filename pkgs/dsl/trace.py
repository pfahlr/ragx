"""Shared trace event utilities for DSL components."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

__all__ = ["TraceEvent", "TraceEventEmitter", "freeze_payload"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable trace event record."""

    event: str
    scope_type: str
    scope_id: str
    payload: Mapping[str, object]


def freeze_payload(data: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable, recursively frozen mapping suitable for traces."""

    def _freeze(value: object) -> object:
        if isinstance(value, Mapping):
            return MappingProxyType({k: _freeze(v) for k, v in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(_freeze(v) for v in value)
        if isinstance(value, set):
            return frozenset(_freeze(v) for v in value)
        return value

    return MappingProxyType({} if data is None else {k: _freeze(v) for k, v in data.items()})


class TraceEventEmitter:
    """Emit immutable trace events to in-memory buffers and optional sinks."""

    def __init__(self, *, sinks: Sequence[Callable[[TraceEvent], None]] | None = None) -> None:
        self._events: list[TraceEvent] = []
        self._sinks: list[Callable[[TraceEvent], None]] = list(sinks or [])

    def subscribe(self, sink: Callable[[TraceEvent], None]) -> None:
        """Register an additional sink invoked for every emitted event."""

        self._sinks.append(sink)

    def emit(
        self,
        *,
        event: str,
        scope_type: str,
        scope_id: str,
        payload: Mapping[str, object] | None = None,
    ) -> TraceEvent:
        frozen_payload = freeze_payload(payload)
        record = TraceEvent(
            event=event,
            scope_type=scope_type,
            scope_id=scope_id,
            payload=frozen_payload,
        )
        self._events.append(record)
        for sink in list(self._sinks):
            sink(record)
        return record

    @property
    def events(self) -> Sequence[TraceEvent]:
        """Return a snapshot of emitted events."""

        return tuple(self._events)

    def extend(self, events: Iterable[TraceEvent]) -> None:
        """Append pre-created events to the emitter buffer."""

        for event in events:
            self._events.append(event)
