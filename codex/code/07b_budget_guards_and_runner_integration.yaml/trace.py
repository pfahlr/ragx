from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, MutableMapping, Sequence


__all__ = ["TraceEvent", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event: str
    scope: str
    payload: Mapping[str, object]


class TraceEventEmitter:
    def __init__(self, *, sink: Callable[[TraceEvent], None] | None = None) -> None:
        self._sink = sink
        self._events: list[TraceEvent] = []

    def emit(self, event: str, scope: str, payload: Mapping[str, object] | MappingProxyType) -> None:
        frozen_payload = _freeze_payload(payload)
        record = TraceEvent(event=event, scope=scope, payload=frozen_payload)
        self._events.append(record)
        if self._sink is not None:
            self._sink(record)

    @property
    def events(self) -> Sequence[TraceEvent]:
        return tuple(self._events)


def _freeze_payload(data: Mapping[str, object] | MappingProxyType) -> Mapping[str, object]:
    frozen: MutableMapping[str, object] = {}
    for key, value in data.items():
        frozen[key] = _freeze_value(value)
    return MappingProxyType(dict(frozen))


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_payload(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_value(item) for item in value)
    return value
