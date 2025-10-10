"""Trace emission utilities for FlowRunner budget integration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, Mapping

from . import budget


class TraceWriter:
    """Minimal writer interface expected by :class:`TraceEventEmitter`."""

    def emit(self, event: str, payload: Mapping[str, object]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class TraceEventEmitter:
    """Emit immutable budget trace payloads via a ``TraceWriter`` instance."""

    writer: TraceWriter
    clock: Callable[[], float]

    def _base_payload(
        self,
        *,
        run_id: str,
        node_id: str,
        loop_iteration: int | None,
        charge: Mapping[str, float],
        status: budget.ScopeBudgetStatus,
        stop_reason: str | None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "run_id": run_id,
            "node_id": node_id,
            "scope": status.scope,
            "loop_iteration": loop_iteration,
            "timestamp": self.clock(),
            "action": status.action.value,
            "charge": MappingProxyType(dict(charge)),
            "remaining": status.remaining,
            "overages": status.overages,
            "stop_reason": stop_reason,
        }
        return payload

    def record_budget(
        self,
        *,
        run_id: str,
        node_id: str,
        loop_iteration: int | None,
        check: budget.BudgetCheck,
        stop_reason: str | None = None,
    ) -> None:
        """Emit ``budget_charge`` for all scopes and ``budget_breach`` for violations."""

        for status in check.scope_statuses:
            payload = self._base_payload(
                run_id=run_id,
                node_id=node_id,
                loop_iteration=loop_iteration,
                charge=check.charge,
                status=status,
                stop_reason=stop_reason,
            )
            self.writer.emit("budget_charge", payload)
            if status.breached:
                self.writer.emit("budget_breach", dict(payload))

    def record_outcome(
        self,
        *,
        run_id: str,
        node_id: str,
        loop_iteration: int | None,
        outcome: budget.BudgetChargeOutcome,
    ) -> None:
        """Emit outcome information after a successful commit."""

        for status in outcome.scope_statuses:
            payload = self._base_payload(
                run_id=run_id,
                node_id=node_id,
                loop_iteration=loop_iteration,
                charge=outcome.charge,
                status=status,
                stop_reason=None,
            )
            self.writer.emit("budget_commit", payload)
