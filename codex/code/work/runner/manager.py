"""BudgetManager orchestrating preflight and commit phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .budgeting import (
    BudgetBreach,
    BudgetChargeOutcome,
    BudgetMode,
    BudgetScope,
    BudgetSpec,
    CostSnapshot,
)
from .trace import TraceEventEmitter

__all__ = [
    "BudgetManager",
    "BudgetPreflightDecision",
]


@dataclass(slots=True)
class _BudgetState:
    spec: BudgetSpec
    spent: CostSnapshot = field(default_factory=CostSnapshot.zero)


@dataclass(frozen=True, slots=True)
class BudgetPreflightDecision:
    """Result from BudgetManager.preflight."""

    blocked: bool
    hard_breaches: tuple[BudgetBreach, ...]
    soft_breaches: tuple[BudgetBreach, ...]

    @property
    def has_warnings(self) -> bool:
        return bool(self.soft_breaches)


class BudgetManager:
    """Coordinate budget scopes for the FlowRunner."""

    def __init__(self, *, trace_emitter: TraceEventEmitter | None = None) -> None:
        self._trace = trace_emitter or TraceEventEmitter()
        self._states: dict[BudgetScope, _BudgetState] = {}

    # ------------------------------------------------------------------
    # Registration and lookup
    # ------------------------------------------------------------------
    def register(self, spec: BudgetSpec) -> None:
        if spec.scope not in self._states:
            self._states[spec.scope] = _BudgetState(spec=spec)

    def spec_for(self, scope: BudgetScope) -> BudgetSpec:
        return self._states[scope].spec

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def preflight(
        self,
        scopes: Sequence[BudgetScope],
        estimate: CostSnapshot,
    ) -> BudgetPreflightDecision:
        hard: list[BudgetBreach] = []
        soft: list[BudgetBreach] = []
        for scope in scopes:
            state = self._states.get(scope)
            if state is None:
                continue
            projected = state.spent.add(estimate)
            if not projected.exceeds(state.spec.limit):
                continue
            overage = projected.overage(state.spec.limit)
            breach = BudgetBreach(
                scope=scope,
                mode=state.spec.mode,
                action=state.spec.breach_action,
                overage=overage,
            )
            if state.spec.mode is BudgetMode.HARD:
                hard.append(breach)
                self._trace.emit_budget_event(
                    "budget_breach",
                    scope,
                    {
                        "action": state.spec.breach_action,
                        "overage": dict(overage.to_dict()),
                        "phase": "preflight",
                    },
                )
            else:
                soft.append(breach)
                self._trace.emit_budget_event(
                    "budget_warning",
                    scope,
                    {
                        "action": state.spec.breach_action,
                        "overage": dict(overage.to_dict()),
                        "phase": "preflight",
                    },
                )
        return BudgetPreflightDecision(
            blocked=bool(hard),
            hard_breaches=tuple(hard),
            soft_breaches=tuple(soft),
        )

    def commit(
        self,
        scopes: Sequence[BudgetScope],
        charge: CostSnapshot,
    ) -> list[BudgetChargeOutcome]:
        if not scopes:
            return []
        outcomes: list[BudgetChargeOutcome] = []
        for scope in scopes:
            state = self._states[scope]
            total = state.spent.add(charge)
            overage = total.overage(state.spec.limit)
            remaining = state.spec.limit.subtract(total)
            state.spent = total
            outcome = BudgetChargeOutcome(
                scope=scope,
                spent=charge,
                total_spent=total,
                remaining=remaining,
                overage=overage,
                mode=state.spec.mode,
                action=state.spec.breach_action,
            )
            self._trace.emit_budget_event(
                "budget_charge",
                scope,
                {
                    "spent": dict(charge.to_dict()),
                    "total_spent": dict(total.to_dict()),
                },
            )
            self._trace.emit_budget_event(
                "budget_remaining",
                scope,
                {
                    "remaining": dict(remaining.to_dict()),
                },
            )
            if overage.milliseconds or overage.tokens_in or overage.tokens_out or overage.calls:
                event = "budget_breach" if state.spec.mode is BudgetMode.HARD else "budget_warning"
                self._trace.emit_budget_event(
                    event,
                    scope,
                    {
                        "action": state.spec.breach_action,
                        "overage": dict(overage.to_dict()),
                        "phase": "commit",
                    },
                )
            outcomes.append(outcome)
        return outcomes

    @property
    def trace(self) -> TraceEventEmitter:
        return self._trace
