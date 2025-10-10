from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

try:  # pragma: no cover - import flexibility for dynamic loading
    from .budget_models import (
        BudgetBreachError,
        BudgetChargeResult,
        BudgetMeter,
        BudgetSpec,
        CostSnapshot,
    )
except ImportError:  # pragma: no cover
    from budget_models import (  # type: ignore[F401]
        BudgetBreachError,
        BudgetChargeResult,
        BudgetMeter,
        BudgetSpec,
        CostSnapshot,
    )

TraceEmitter = Callable[[str, str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    scope_id: str
    charges: tuple[BudgetChargeResult, ...]
    breaches: tuple[BudgetChargeResult, ...]
    stop_requested: bool

    @property
    def breached(self) -> bool:
        return bool(self.breaches)


class BudgetManager:
    def __init__(self, *, trace_emitter: TraceEmitter | None = None) -> None:
        self._trace_emitter = trace_emitter
        self._meters: dict[str, BudgetMeter] = {}
        self._parents: dict[str, str | None] = {}

    def register_scope(
        self, scope_id: str, spec: BudgetSpec, *, parent: str | None = None
    ) -> None:
        if scope_id in self._meters:
            raise ValueError(f"Scope '{scope_id}' already registered")
        if parent is not None and parent not in self._meters:
            raise ValueError(f"Parent scope '{parent}' must be registered before children")
        self._meters[scope_id] = BudgetMeter(spec)
        self._parents[scope_id] = parent

    def preflight(
        self, scope_id: str, cost: CostSnapshot, *, label: str = "preflight"
    ) -> BudgetDecision:
        self._ensure_scope(scope_id)
        charges = [
            self._meters[current].preview(cost, label=label)
            for current in self._iter_chain(scope_id)
        ]
        return self._decision(scope_id, charges)

    def commit(
        self, scope_id: str, cost: CostSnapshot, *, label: str
    ) -> BudgetDecision:
        self._ensure_scope(scope_id)
        charges: list[BudgetChargeResult] = []
        errors: list[BudgetBreachError] = []
        for current in self._iter_chain(scope_id):
            result = self._meters[current].charge(cost, label=label)
            charges.append(result)
            self._emit_charge(result)
            if result.breached:
                action = (result.breach_action or "").lower()
                if result.breach_kind is not None and result.breach_kind.value == "hard":
                    errors.append(BudgetBreachError(result))
                elif action == "error":
                    errors.append(BudgetBreachError(result))
        decision = self._decision(scope_id, charges)
        if errors:
            raise errors[0]
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_scope(self, scope_id: str) -> None:
        if scope_id not in self._meters:
            raise KeyError(f"Unknown budget scope '{scope_id}'")

    def _iter_chain(self, scope_id: str) -> Iterable[str]:
        current: str | None = scope_id
        while current is not None:
            yield current
            current = self._parents.get(current)

    def _decision(
        self, scope_id: str, charges: Iterable[BudgetChargeResult]
    ) -> BudgetDecision:
        charge_tuple = tuple(charges)
        breaches = tuple(result for result in charge_tuple if result.breached)
        stop_requested = any(result.should_stop for result in charge_tuple)
        return BudgetDecision(
            scope_id=scope_id,
            charges=charge_tuple,
            breaches=breaches,
            stop_requested=stop_requested,
        )

    def _emit_charge(self, charge: BudgetChargeResult) -> None:
        if self._trace_emitter is None:
            return
        payload: dict[str, object] = {
            "label": charge.label,
            "cost": charge.cost.as_mapping(),
            "remaining": charge.remaining,
            "overages": charge.overages,
            "breached": charge.breached,
        }
        self._trace_emitter("budget_charge", charge.scope, payload)
        if charge.breached:
            breach_payload = {
                "label": charge.label,
                "overages": charge.overages,
                "remaining": charge.remaining,
                "breach_action": charge.breach_action,
            }
            self._trace_emitter("budget_breach", charge.scope, breach_payload)
