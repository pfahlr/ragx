"""Shared trace event emitter for budget and policy integrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze_value(value: Any) -> Any:
    """Recursively convert mappings/sequences into immutable counterparts."""

    if isinstance(value, Mapping):
        frozen = {key: _freeze_value(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable mapping proxy for the provided payload."""

    if payload is None:
        return MappingProxyType({})
    frozen = {key: _freeze_value(value) for key, value in payload.items()}
    return MappingProxyType(frozen)

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
        self._validator: Callable[[TraceEvent], None] | None = None

    def attach_sink(self, sink: Callable[[TraceEvent], None] | None) -> None:
        """Attach or clear an external sink that receives emitted events."""

        self._sink = sink

    def attach_validator(
        self, validator: Callable[[TraceEvent], None] | None
    ) -> None:
        """Register a validator invoked for every emitted event."""

        self._validator = validator

    def emit(
        self,
        event: str,
        *,
        scope_type: str,
        scope_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        immutable_payload = _freeze_payload(payload)
        record = TraceEvent(
            event=event,
            scope_type=scope_type,
            scope_id=scope_id,
            payload=immutable_payload,
        )
        if self._validator is not None:
            self._validator(record)
        self._events.append(record)
        if self._sink is not None:
            self._sink(record)
        return record

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()
