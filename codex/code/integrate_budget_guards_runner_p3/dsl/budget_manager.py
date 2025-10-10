"""Scope-aware budget manager coordinating previews, commits, and traces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from . import budget_models as bm
from .trace import TraceEventEmitter

__all__ = ["BudgetManager", "BudgetBreachError", "BudgetError"]


class BudgetError(RuntimeError):
    """Base error for budget management."""


class BudgetBreachError(BudgetError):
    """Raised when a budget breach requires execution to stop."""

    def __init__(self, scope: bm.ScopeKey, outcome: bm.BudgetChargeOutcome) -> None:
        super().__init__(
            f"Budget '{outcome.spec.name}' breached at {scope.scope_type}:{scope.scope_id}"
        )
        self.scope = scope
        self.outcome = outcome


@dataclass(slots=True)
class _ScopeState:
    scope: bm.ScopeKey
    spent: dict[str, bm.CostSnapshot]


class BudgetManager:
    """Coordinate budget previews and commits across scopes."""

    def __init__(
        self,
        *,
        specs: Iterable[bm.BudgetSpec],
        trace: TraceEventEmitter | None = None,
    ) -> None:
        self._trace = trace or TraceEventEmitter()
        self._specs_by_scope: dict[str, list[bm.BudgetSpec]] = defaultdict(list)
        for spec in specs:
            self._specs_by_scope[spec.scope_type].append(spec)
        self._scopes: dict[bm.ScopeKey, _ScopeState] = {}
        self._history: dict[bm.ScopeKey, _ScopeState] = {}

    # ------------------------------------------------------------------
    # Scope management
    # ------------------------------------------------------------------
    def enter_scope(self, scope: bm.ScopeKey) -> None:
        if scope in self._scopes:
            raise KeyError(f"scope already active: {scope}")
        self._history.pop(scope, None)
        specs = self._specs_by_scope.get(scope.scope_type, [])
        spent = {spec.name: bm.CostSnapshot.zero() for spec in specs}
        self._scopes[scope] = _ScopeState(scope=scope, spent=spent)

    def exit_scope(self, scope: bm.ScopeKey) -> None:
        if scope not in self._scopes:
            raise KeyError(f"scope not active: {scope}")
        state = self._scopes.pop(scope)
        self._history[scope] = state

    # ------------------------------------------------------------------
    # Charging
    # ------------------------------------------------------------------
    def preview_charge(
        self,
        scope: bm.ScopeKey,
        cost: bm.CostSnapshot,
    ) -> bm.BudgetDecision:
        state = self._scopes.get(scope)
        if state is None:
            raise KeyError(f"scope not active: {scope}")
        specs = self._specs_by_scope.get(scope.scope_type, [])
        outcomes: list[bm.BudgetChargeOutcome] = []
        for spec in specs:
            prior = state.spent[spec.name]
            outcome = bm.BudgetChargeOutcome.compute(spec=spec, prior=prior, cost=cost)
            outcomes.append(outcome)
        decision = bm.BudgetDecision.make(scope=scope, cost=cost, outcomes=outcomes)
        return decision

    def commit_charge(self, decision: bm.BudgetDecision) -> None:
        state = self._scopes.get(decision.scope)
        if state is None:
            raise KeyError(f"scope not active: {decision.scope}")
        if decision.should_stop:
            blocking = decision.blocking
            if blocking is None:  # pragma: no cover - defensive guard
                raise BudgetError("blocking outcome missing for stop decision")
            raise BudgetBreachError(decision.scope, blocking)
        for outcome in decision.outcomes:
            state.spent[outcome.spec.name] = outcome.charge.new_total
            self._trace.emit(
                "budget_charge",
                scope_type=decision.scope.scope_type,
                scope_id=decision.scope.scope_id,
                payload=outcome.to_trace_payload(
                    scope_type=decision.scope.scope_type,
                    scope_id=decision.scope.scope_id,
                ),
            )

    def record_breach(self, decision: bm.BudgetDecision) -> None:
        state = self._scopes.get(decision.scope)
        if state is None:
            raise KeyError(f"scope not active: {decision.scope}")
        for outcome in decision.outcomes:
            if outcome.breached:
                self._trace.emit(
                    "budget_breach",
                    scope_type=decision.scope.scope_type,
                    scope_id=decision.scope.scope_id,
                    payload=outcome.to_trace_payload(
                        scope_type=decision.scope.scope_type,
                        scope_id=decision.scope.scope_id,
                    ),
                )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def spent(self, scope: bm.ScopeKey, spec_name: str) -> bm.CostSnapshot:
        state = self._scopes.get(scope)
        if state is None:
            state = self._history.get(scope)
            if state is None:
                raise KeyError(f"scope not active: {scope}")
        return state.spent[spec_name]

    @property
    def trace(self) -> TraceEventEmitter:
        return self._trace
