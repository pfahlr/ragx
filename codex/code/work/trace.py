from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pkgs.dsl.models import mapping_proxy


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable trace payload routed through :class:`TraceEventEmitter`."""

    event: str
    scope: str
    scope_type: str
    payload: Mapping[str, object]
    breach_kind: str | None = None
    metadata: Mapping[str, object] = field(default_factory=mapping_proxy)


class TraceWriter:
    """Optional sink that can consume emitted trace events."""

    def __init__(self, sink: Callable[[TraceEvent], None] | None = None) -> None:
        self._sink = sink
        self._events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self._events.append(event)
        if self._sink is not None:
            self._sink(event)

    @property
    def events(self) -> Sequence[TraceEvent]:
        return tuple(self._events)


class TraceEventEmitter:
    """Factory that normalises trace payloads before forwarding them."""

    def __init__(self, writer: TraceWriter | None = None) -> None:
        self._writer = writer
        self._buffer: list[TraceEvent] = []

    def emit(
        self,
        *,
        event: str,
        scope: str,
        payload: Mapping[str, object] | None = None,
        scope_type: str = "runtime",
        breach_kind: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TraceEvent:
        frozen_payload = mapping_proxy(payload or {})
        frozen_metadata = mapping_proxy(metadata or {})
        trace_event = TraceEvent(
            event=event,
            scope=scope,
            scope_type=scope_type,
            payload=frozen_payload,
            breach_kind=breach_kind,
            metadata=frozen_metadata,
        )
        self._buffer.append(trace_event)
        if self._writer is not None:
            self._writer.emit(trace_event)
        return trace_event

    @property
    def events(self) -> Sequence[TraceEvent]:
        return tuple(self._buffer)
