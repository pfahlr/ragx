"""Budget manager orchestrating preview and commit operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from .budget_models import (
    BudgetBreach,
    BudgetBreachError,
    BudgetCharge,
    BudgetCheck,
    BudgetDecision,
    BudgetSpec,
    CostAmount,
    CostSnapshot,
    LoopSummary,
)
from .trace_emitter import TraceEventEmitter


@dataclass
class _BudgetState:
    spec: BudgetSpec
    spent: Decimal = Decimal("0")
    warnings: List[BudgetBreach] = field(default_factory=list)


class BudgetManager:
    """Coordinates budget preview/commit flows across scopes."""

    def __init__(self, *, emitter: Optional[TraceEventEmitter] = None) -> None:
        self._states: Dict[str, _BudgetState] = {}
        self.emitter = emitter or TraceEventEmitter()

    def register_scope(self, scope_id: str, spec: BudgetSpec) -> None:
        if scope_id in self._states:
            raise ValueError(f"Scope '{scope_id}' already registered")
        self._states[scope_id] = _BudgetState(spec=spec)

    def preview(self, scope_id: str, estimated_cost: CostAmount, *, node_id: Optional[str] = None) -> BudgetCheck:
        state = self._states.get(scope_id)
        if state is None:
            raise KeyError(scope_id)
        if estimated_cost.value < 0:
            raise ValueError("Estimated cost cannot be negative")

        projected = state.spent + estimated_cost.value
        snapshot = CostSnapshot.from_values(limit=state.spec.limit, spent=state.spent, projected=projected)
        decision, breach = state.spec.resolve_decision(snapshot)

        check = BudgetCheck(scope_id=scope_id, snapshot=snapshot, decision=decision, breach=breach)
        self._emit(
            "budget_preview",
            scope_id,
            {
                "scope_id": scope_id,
                "node_id": node_id,
                "decision": decision.value,
                "estimated": str(estimated_cost.value),
                "remaining": str(snapshot.remaining),
                "overage": str(snapshot.overage),
            },
        )

        if decision is BudgetDecision.STOP:
            self._emit(
                "budget_stop",
                scope_id,
                {
                    "scope_id": scope_id,
                    "node_id": node_id,
                    "reason": breach.reason if breach else "limit_exceeded",
                },
            )
        elif decision is BudgetDecision.WARN:
            self._emit(
                "budget_warning",
                scope_id,
                {
                    "scope_id": scope_id,
                    "node_id": node_id,
                    "remaining": str(snapshot.remaining),
                },
            )
        return check

    def commit(self, scope_id: str, actual_cost: CostAmount, *, node_id: Optional[str] = None) -> BudgetCharge:
        state = self._states.get(scope_id)
        if state is None:
            raise KeyError(scope_id)
        if actual_cost.value < 0:
            raise ValueError("Actual cost cannot be negative")

        projected = state.spent + actual_cost.value
        snapshot = CostSnapshot.from_values(limit=state.spec.limit, spent=projected, projected=projected)
        decision, breach = state.spec.resolve_decision(snapshot)

        if decision is BudgetDecision.STOP:
            self._emit(
                "budget_stop",
                scope_id,
                {
                    "scope_id": scope_id,
                    "node_id": node_id,
                    "reason": breach.reason if breach else "limit_exceeded",
                },
            )
            raise BudgetBreachError(f"Budget for scope '{scope_id}' exceeded")

        state.spent = projected
        if decision is BudgetDecision.WARN and breach is not None:
            state.warnings.append(breach)
            self._emit(
                "budget_warning",
                scope_id,
                {
                    "scope_id": scope_id,
                    "node_id": node_id,
                    "remaining": str(snapshot.remaining),
                },
            )

        charge = BudgetCharge(scope_id=scope_id, snapshot=snapshot, decision=decision, breach=breach)

        self._emit(
            "budget_commit",
            scope_id,
            {
                "scope_id": scope_id,
                "node_id": node_id,
                "decision": decision.value,
                "spent": str(snapshot.spent),
                "remaining": str(snapshot.remaining),
            },
        )
        return charge

    def summary(self, scope_id: str) -> LoopSummary:
        state = self._states.get(scope_id)
        if state is None:
            raise KeyError(scope_id)
        limit = state.spec.limit.value
        remaining = max(limit - state.spent, Decimal("0"))
        overage = max(state.spent - limit, Decimal("0"))
        return LoopSummary(
            scope_id=scope_id,
            total_spent=state.spent,
            total_remaining=remaining,
            total_overage=overage,
        )

    def warnings(self, scope_id: str) -> List[BudgetBreach]:
        state = self._states.get(scope_id)
        if state is None:
            raise KeyError(scope_id)
        return list(state.warnings)

    def _emit(self, event: str, scope_id: str, payload: dict) -> None:
        self.emitter.emit(event, scope_type="budget", scope_id=scope_id, payload=payload)


__all__ = ["BudgetManager"]
