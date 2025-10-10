"""BudgetManager orchestrates scope registration and spend tracking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import MutableMapping

from ..trace.emitter import TraceEventEmitter
from .meter import BudgetMeter
from .models import BudgetChargeOutcome, BudgetSpec, CostSnapshot

__all__ = ["BudgetManager"]


class BudgetManager:
    """Manage budget meters for run/node/loop scopes."""

    def __init__(self, *, emitter: TraceEventEmitter | None = None) -> None:
        self._emitter = emitter
        self._meters: dict[tuple[str, str], BudgetMeter] = {}

    # ------------------------------------------------------------------
    # Scope registration
    # ------------------------------------------------------------------
    def register_scope(
        self,
        *,
        scope_type: str,
        scope_id: str,
        spec: BudgetSpec | Mapping[str, object] | None,
    ) -> None:
        key = (scope_type, scope_id)
        if spec is None:
            self._meters.pop(key, None)
            return
        if isinstance(spec, Mapping):
            budget_spec = BudgetSpec.from_mapping(scope=scope_type, scope_id=scope_id, data=spec)
        elif isinstance(spec, BudgetSpec):
            budget_spec = spec
        else:  # pragma: no cover - defensive guard
            raise TypeError("spec must be a BudgetSpec, mapping, or None")
        self._meters[key] = BudgetMeter(scope_type=scope_type, scope_id=scope_id, spec=budget_spec)

    def ensure_scope(self, scope_type: str, scope_id: str) -> BudgetMeter | None:
        return self._meters.get((scope_type, scope_id))

    # ------------------------------------------------------------------
    # Charging API
    # ------------------------------------------------------------------
    def preflight(
        self,
        scope_type: str,
        scope_id: str,
        cost: CostSnapshot | Mapping[str, object],
        *,
        event_context: Mapping[str, object] | MutableMapping[str, object] | None = None,
    ) -> BudgetChargeOutcome | None:
        meter = self.ensure_scope(scope_type, scope_id)
        if meter is None:
            return None
        normalized_cost = self._normalize_cost(cost)
        outcome = meter.preview(normalized_cost)
        self._emit("budget_preflight", meter, outcome, event_context)
        return outcome

    def commit(
        self,
        scope_type: str,
        scope_id: str,
        cost: CostSnapshot | Mapping[str, object],
        *,
        event_context: Mapping[str, object] | MutableMapping[str, object] | None = None,
    ) -> BudgetChargeOutcome | None:
        meter = self.ensure_scope(scope_type, scope_id)
        if meter is None:
            return None
        normalized_cost = self._normalize_cost(cost)
        outcome = meter.commit(normalized_cost)
        self._emit("budget_charge", meter, outcome, event_context)
        if outcome.breached:
            self._emit("budget_breach", meter, outcome, event_context)
        return outcome

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_cost(cost: CostSnapshot | Mapping[str, object]) -> CostSnapshot:
        if isinstance(cost, CostSnapshot):
            return cost
        if isinstance(cost, Mapping):
            return CostSnapshot.from_mapping(cost)
        raise TypeError("cost must be a CostSnapshot or mapping")

    def _emit(
        self,
        event: str,
        meter: BudgetMeter,
        outcome: BudgetChargeOutcome,
        context: Mapping[str, object] | MutableMapping[str, object] | None,
    ) -> None:
        if self._emitter is None:
            return
        payload = outcome.to_trace_payload(context=context)
        self._emitter.emit(event, meter.scope_type, meter.scope_id, payload)
