"""Shared trace event primitives for budget/policy instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, MutableMapping, Sequence

__all__ = ["TraceEvent", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable trace event emitted by the runner subsystems."""

    event: str
    scope_type: str
    scope_id: str
    payload: Mapping[str, object]


class TraceEventEmitter:
    """Collect trace events and optionally forward them to an external sink."""

    def __init__(self, sink: Callable[[TraceEvent], None] | None = None) -> None:
        self._sink = sink
        self._events: list[TraceEvent] = []

    @staticmethod
    def _freeze_payload(payload: Mapping[str, object] | MutableMapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(payload))

    def emit(
        self,
        event: str,
        scope_type: str,
        scope_id: str,
        payload: Mapping[str, object] | MutableMapping[str, object],
    ) -> TraceEvent:
        record = TraceEvent(
            event=event,
            scope_type=scope_type,
            scope_id=scope_id,
            payload=self._freeze_payload(payload),
        )
        self._events.append(record)
        if self._sink is not None:
            self._sink(record)
        return record

    @property
    def events(self) -> Sequence[TraceEvent]:
        return tuple(self._events)

    def clear(self) -> None:
        """Remove all buffered events. Useful in tests."""

        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
