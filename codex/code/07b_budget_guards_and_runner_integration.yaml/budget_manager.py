"""BudgetManager orchestrates scoped budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .budget_models import (
    BreachAction,
    BudgetBreach,
    BudgetDecision,
    BudgetSpec,
    CostSnapshot,
    ScopeKey,
    ScopeSnapshot,
)


@dataclass
class _ScopeState:
    spec: BudgetSpec
    spent: CostSnapshot = CostSnapshot.zero()


class BudgetBreachError(RuntimeError):
    """Raised when a hard budget breach occurs during commit."""

    def __init__(self, decision: BudgetDecision):
        message = (
            f"Budget breach for {decision.scope.category}:{decision.scope.identifier}"
        )
        super().__init__(message)
        self.decision = decision


class BudgetManager:
    """Coordinates budget evaluations across scopes."""

    def __init__(self) -> None:
        self._scopes: Dict[Tuple[str, str], _ScopeState] = {}
        self._warnings: List[str] = []
        self._warned_scopes: set[Tuple[str, str]] = set()

    def register_scope(self, spec: BudgetSpec) -> None:
        key = spec.scope.as_tuple()
        if key in self._scopes:
            raise ValueError(f"Scope already registered: {key}")
        self._scopes[key] = _ScopeState(spec=spec)

    def has_scope(self, scope: ScopeKey) -> bool:
        return scope.as_tuple() in self._scopes

    def preflight(self, scope: ScopeKey, attempted: CostSnapshot) -> BudgetDecision:
        return self._evaluate(scope, attempted, stage="preflight", mutate=False)

    def commit(self, scope: ScopeKey, attempted: CostSnapshot) -> BudgetDecision:
        decision = self._evaluate(scope, attempted, stage="commit", mutate=True)
        if not decision.allowed and decision.action == BreachAction.STOP:
            raise BudgetBreachError(decision)
        return decision

    def snapshot(self, scope: ScopeKey) -> ScopeSnapshot:
        state = self._require_scope(scope)
        return ScopeSnapshot(scope=scope, limit_ms=state.spec.limit_ms, spent=state.spent)

    def drain_warnings(self) -> List[str]:
        warnings = list(self._warnings)
        self._warnings.clear()
        return warnings

    def spec_for(self, scope: ScopeKey) -> BudgetSpec:
        return self._require_scope(scope).spec

    def _require_scope(self, scope: ScopeKey) -> _ScopeState:
        key = scope.as_tuple()
        if key not in self._scopes:
            raise KeyError(f"Scope not registered: {key}")
        return self._scopes[key]

    def _evaluate(
        self,
        scope: ScopeKey,
        attempted: CostSnapshot,
        *,
        stage: str,
        mutate: bool,
    ) -> BudgetDecision:
        state = self._require_scope(scope)
        remaining_ms = max(0.0, state.spec.limit_ms - state.spent.milliseconds)
        remaining_snapshot = CostSnapshot(remaining_ms)
        breach: Optional[BudgetBreach] = None
        allowed = True

        if attempted.milliseconds > remaining_ms:
            breach = BudgetBreach(
                scope=scope,
                attempted=attempted,
                limit_ms=state.spec.limit_ms,
                remaining=remaining_snapshot,
                action=state.spec.breach_action,
            )
            allowed = state.spec.breach_action == BreachAction.WARN

        if mutate and allowed:
            new_spent = state.spent + attempted
            self._scopes[scope.as_tuple()] = _ScopeState(spec=state.spec, spent=new_spent)
            if breach is not None:
                self._append_warning(scope, breach)
            remaining_snapshot = CostSnapshot(
                max(0.0, state.spec.limit_ms - new_spent.milliseconds)
            )
        elif mutate and not allowed:
            # Preserve existing spent totals and produce decision for caller.
            pass

        decision = BudgetDecision(
            scope=scope,
            stage=stage,
            attempted=attempted,
            remaining=remaining_snapshot if mutate and allowed else remaining_snapshot,
            allowed=allowed,
            action=state.spec.breach_action,
            breach=breach,
        )
        return decision

    def _append_warning(self, scope: ScopeKey, breach: BudgetBreach) -> None:
        key = scope.as_tuple()
        if key in self._warned_scopes:
            return
        message = (
            f"{scope.category}:{scope.identifier} warning: attempted {breach.attempted.milliseconds:.1f}ms "
            f"limit {breach.limit_ms:.1f}ms"
        )
        self._warnings.append(message)
        self._warned_scopes.add(key)
