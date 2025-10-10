"""Scope-aware budget manager coordinating trace emission."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping

from .budget import (
    BudgetBreachError,
    BudgetChargeOutcome,
    BudgetMeter,
    BudgetMode,
    BudgetSpec,
    BudgetStopSignal,
)
from .costs import normalize_cost


class BudgetManager:
    """Manage multiple budget scopes and emit structured events."""

    def __init__(self, trace_writer) -> None:
        self._trace_writer = trace_writer
        self._meters: Dict[str, BudgetMeter] = {}
        self._warnings: MutableMapping[str, list[str]] = {}

    def register_scope(self, scope_id: str, spec: BudgetSpec | Mapping[str, object]) -> None:
        if scope_id in self._meters:
            raise ValueError(f"Budget scope already registered: {scope_id}")
        budget_spec = self._coerce_spec(spec)
        self._meters[scope_id] = BudgetMeter(scope_id=scope_id, spec=budget_spec)

    def scopes(self) -> Iterable[str]:
        return self._meters.keys()

    def get_remaining(self, scope_id: str) -> int:
        meter = self._meters[scope_id]
        return meter.snapshot().remaining

    def charge(self, scope_id: str, cost: Mapping[str, int | float]) -> BudgetChargeOutcome:
        meter = self._meters[scope_id]
        normalized = normalize_cost(cost)
        try:
            outcome = meter.charge(normalized)
        except BudgetStopSignal as stop:
            self._emit("budget_breach", stop.outcome)
            raise
        except BudgetBreachError as breach:
            self._emit("budget_breach", breach.outcome)
            raise
        else:
            self._emit("budget_charge", outcome)
            if outcome.breached and outcome.action == "warn":
                self._emit("budget_breach", outcome)
                self._warnings.setdefault(scope_id, []).append(
                    f"{scope_id} exceeded {outcome.spec.metric} by {outcome.overage}"
                )
            return outcome

    def warnings_for(self, scope_id: str) -> list[str]:
        return list(self._warnings.get(scope_id, []))

    def consume_warnings(self, scope_id: str) -> list[str]:
        return self._warnings.pop(scope_id, [])

    def _emit(self, event: str, outcome: BudgetChargeOutcome) -> None:
        payload = dict(outcome.as_mapping())
        payload["scope"] = outcome.scope_id
        payload["action"] = outcome.action
        self._trace_writer.write(event, payload)

    @staticmethod
    def _coerce_spec(spec: BudgetSpec | Mapping[str, object]) -> BudgetSpec:
        if isinstance(spec, BudgetSpec):
            return spec
        metric = spec.get("metric")
        limit = spec.get("limit")
        action = spec.get("breach_action", "error")
        mode_raw = spec.get("mode", BudgetMode.HARD.value)
        mode = BudgetMode(mode_raw)
        return BudgetSpec(metric=metric, limit=limit, breach_action=str(action), mode=mode)
