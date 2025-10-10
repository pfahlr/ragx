from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

__all__ = ["TraceEvent", "TraceRecorder", "TraceEventEmitter"]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event: str
    scope: str
    payload: Mapping[str, object]


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> Sequence[TraceEvent]:
        return tuple(self._events)


class TraceEventEmitter:
    """Emit immutable trace events to an optional recorder and sink."""

    def __init__(
        self,
        *,
        recorder: TraceRecorder | None = None,
        sink: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self._recorder = recorder
        self._sink = sink

    def emit(
        self,
        event: str,
        *,
        scope: str,
        payload: Mapping[str, object] | None = None,
    ) -> TraceEvent:
        sanitized = self._sanitize_payload(payload or {})
        record = TraceEvent(event=event, scope=scope, payload=sanitized)
        if self._recorder is not None:
            self._recorder.record(record)
        if self._sink is not None:
            self._sink(record)
        return record

    def policy_sink(self) -> Callable[[object], None]:
        """Bridge :class:`pkgs.dsl.policy.PolicyStack` events to this emitter."""

        def _sink(policy_event: object) -> None:
            event_name = getattr(policy_event, "event")
            scope = getattr(policy_event, "scope")
            data = getattr(policy_event, "data")
            if not isinstance(data, Mapping):
                raise TypeError("policy trace data must be a mapping")
            self.emit(event_name, scope=scope, payload=data)

        return _sink

    def _sanitize_payload(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType({key: self._freeze(value) for key, value in payload.items()})

    def _freeze(self, value: object) -> object:
        if isinstance(value, Mapping):
            return {key: self._freeze(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze(item) for item in value)
        return value
