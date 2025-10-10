"""Trace utilities for FlowRunner components."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .models import mapping_proxy

__all__ = [
    "RunnerTraceEvent",
    "RunnerTraceRecorder",
    "emit_trace_event",
]


@dataclass(frozen=True, slots=True)
class RunnerTraceEvent:
    """Structured trace event emitted by the runner."""

    event: str
    scope_type: str | None
    scope_id: str | None
    data: Mapping[str, object]


class RunnerTraceRecorder:
    """Recorder used in tests to capture FlowRunner trace events."""

    def __init__(self) -> None:
        self._events: list[RunnerTraceEvent] = []

    def record(self, event: RunnerTraceEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> Sequence[RunnerTraceEvent]:
        return tuple(self._events)


def emit_trace_event(
    recorder: RunnerTraceRecorder | None,
    sink: Callable[[RunnerTraceEvent], None] | None,
    *,
    event: str,
    scope_type: str | None,
    scope_id: str | None,
    payload: Mapping[str, object],
) -> RunnerTraceEvent:
    """Helper that mirrors policy tracing with immutable payloads."""

    record = RunnerTraceEvent(
        event=event,
        scope_type=scope_type,
        scope_id=scope_id,
        data=mapping_proxy(payload),
    )
    if recorder is not None:
        recorder.record(record)
    if sink is not None:
        sink(record)
    return record

