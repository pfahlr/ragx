"""Utilities to bridge policy trace events into the shared trace emitter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pkgs.dsl.policy import PolicyTraceEvent

from .trace import TraceEventEmitter

__all__ = ["PolicyTraceBridge"]


class PolicyTraceBridge:
    """Forward :class:`PolicyTraceEvent` instances to a :class:`TraceEventEmitter`."""

    def __init__(
        self,
        emitter: TraceEventEmitter,
        downstream: Callable[[PolicyTraceEvent], None] | None = None,
        *,
        scope_type: str = "policy",
    ) -> None:
        self._emitter = emitter
        self._downstream = downstream
        self._scope_type = scope_type

    def __call__(self, event: PolicyTraceEvent) -> None:
        payload: dict[str, Any] = dict(event.data)
        self._emitter.emit(
            event.event,
            scope_type=self._scope_type,
            scope_id=event.scope,
            payload=payload,
        )
        if self._downstream is not None:
            self._downstream(event)
