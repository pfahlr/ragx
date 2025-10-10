"""Trace emission utilities wrapping an underlying TraceWriter."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Protocol


class TraceWriter(Protocol):
    def write(self, event_name: str, payload: Mapping[str, object]) -> None:
        ...


class TraceEventEmitter:
    """Emit immutable trace payloads for runner subsystems."""

    def __init__(self, trace_writer: TraceWriter) -> None:
        self._trace_writer = trace_writer

    def write(self, event_name: str, payload: Mapping[str, object]) -> None:
        self._trace_writer.write(event_name, MappingProxyType(dict(payload)))

    def emit(self, event_name: str, **payload: object) -> None:
        self.write(event_name, payload)

    def emit_budget(self, event_name: str, budget_payload: Mapping[str, object]) -> None:
        self.write(event_name, budget_payload)

    def emit_policy_resolution(self, node_id: str, allow: list[str], deny: list[str]) -> None:
        self.emit(
            "policy_resolved",
            node_id=node_id,
            allow=tuple(sorted(allow)),
            deny=tuple(sorted(deny)),
        )
