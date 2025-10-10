"""Trace writer utilities for the DSL runner."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["TraceEvent", "TraceWriter", "InMemoryTraceWriter"]


@dataclass(slots=True, frozen=True)
class TraceEvent:
    """Single trace event emitted during runner execution."""

    event: str
    payload: Mapping[str, object]


class TraceWriter:
    """Abstract trace sink used by the runner."""

    def emit(self, event: str, payload: Mapping[str, object]) -> None:  # pragma: no cover
        raise NotImplementedError

    def snapshot(self) -> Sequence[Mapping[str, object]]:  # pragma: no cover
        raise NotImplementedError


class InMemoryTraceWriter(TraceWriter):
    """Simple trace writer that buffers events in memory."""

    def __init__(self) -> None:
        self._events: list[MutableMapping[str, object]] = []

    def emit(self, event: str, payload: Mapping[str, object]) -> None:
        record: MutableMapping[str, object] = {"event": event}
        record.update(payload)
        self._events.append(record)

    def snapshot(self) -> Sequence[Mapping[str, object]]:
        return tuple(MappingProxyType(dict(event)) for event in self._events)

    @property
    def events(self) -> Sequence[Mapping[str, object]]:
        return self.snapshot()
