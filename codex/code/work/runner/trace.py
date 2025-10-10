"""Trace emission helpers for FlowRunner budget integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, MutableMapping, Protocol

from .budget_models import BudgetChargeOutcome


class TraceWriter(Protocol):
    """Protocol describing the trace sink used by the runner."""

    def emit(self, event: str, payload: Mapping[str, object]) -> None:  # pragma: no cover - protocol
        ...


@dataclass
class InMemoryTraceWriter:
    """Simple TraceWriter used by unit tests."""

    events: List[MutableMapping[str, object]] = field(default_factory=list)

    def emit(self, event: str, payload: Mapping[str, object]) -> None:
        entry: MutableMapping[str, object] = {"event": event}
        entry.update(payload)
        self.events.append(entry)


class TraceEventEmitter:
    """Formats structured events for budget charges and breaches."""

    def __init__(self, writer: TraceWriter):
        self._writer = writer

    def emit_run_event(self, event: str, *, flow_id: str, run_id: str, **extra: object) -> None:
        payload = {"flow_id": flow_id, "run_id": run_id}
        if extra:
            payload.update(extra)
        self._writer.emit(event, payload)

    def emit_budget_charge(
        self,
        *,
        outcome: BudgetChargeOutcome,
        context: Mapping[str, object],
    ) -> None:
        payload = {**context, **outcome.to_trace_payload()}
        self._writer.emit("budget_charge", payload)

    def emit_budget_breach(
        self,
        *,
        outcome: BudgetChargeOutcome,
        context: Mapping[str, object],
    ) -> None:
        payload = {**context, **outcome.to_trace_payload()}
        self._writer.emit("budget_breach", payload)
