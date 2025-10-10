"""Budget domain models and BudgetManager implementation for Phase 3 sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class BudgetMode(str, Enum):
    """Enumeration describing how a budget should be enforced."""

    HARD = "hard"
    SOFT = "soft"


class BreachAction(str, Enum):
    """Actions to take when a budget is exceeded."""

    STOP = "stop"
    WARN = "warn"


@dataclass(frozen=True)
class BudgetSpec:
    """Immutable description of a single budget scope."""

    scope_type: str
    limit_ms: float
    mode: BudgetMode = BudgetMode.HARD
    breach_action: BreachAction = BreachAction.STOP


@dataclass(frozen=True)
class CostSnapshot:
    """Represents a normalised cost measurement in milliseconds."""

    milliseconds: float

    @classmethod
    def from_seconds(cls, seconds: float) -> "CostSnapshot":
        """Create a snapshot from seconds, converting to milliseconds."""

        return cls(milliseconds=seconds * 1000.0)

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(milliseconds=self.milliseconds + other.milliseconds)


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome from preflight or commit phases."""

    scope_id: str
    should_stop: bool
    action: BreachAction
    remaining_ms: float
    overage_ms: float

    @property
    def is_breached(self) -> bool:
        return self.overage_ms > 0


@dataclass
class _ScopeState:
    spec: BudgetSpec
    spent: CostSnapshot


class BudgetManager:
    """Tracks remaining budget per scope and exposes preflight/commit APIs."""

    def __init__(self, specs: Optional[Dict[Tuple[str, str], BudgetSpec]] = None) -> None:
        self._specs = specs or {}
        self._scopes: Dict[Tuple[str, str], _ScopeState] = {}

    def _get_spec(self, scope_type: str, scope_id: str, explicit: Optional[BudgetSpec]) -> BudgetSpec:
        if explicit is not None:
            return explicit
        try:
            return self._specs[(scope_type, scope_id)]
        except KeyError as exc:
            raise KeyError(f"No budget spec for scope {(scope_type, scope_id)}") from exc

    def enter_scope(self, scope_type: str, scope_id: str, spec: Optional[BudgetSpec] = None) -> None:
        """Register a new scope with the provided spec."""

        key = (scope_type, scope_id)
        if key in self._scopes:
            raise ValueError(f"Scope {key} already registered")
        resolved_spec = self._get_spec(scope_type, scope_id, spec)
        self._scopes[key] = _ScopeState(spec=resolved_spec, spent=CostSnapshot(milliseconds=0.0))

    def exit_scope(self, scope_type: str, scope_id: str) -> None:
        key = (scope_type, scope_id)
        try:
            del self._scopes[key]
        except KeyError as exc:
            raise KeyError(f"Scope {key} not registered") from exc

    def _ensure_scope(self, scope_type: str, scope_id: str) -> _ScopeState:
        try:
            return self._scopes[(scope_type, scope_id)]
        except KeyError as exc:
            raise KeyError(f"Scope {(scope_type, scope_id)} not registered") from exc

    def _make_decision(self, scope_id: str, spec: BudgetSpec, projected: float) -> BudgetDecision:
        overage = max(projected - spec.limit_ms, 0.0)
        remaining = max(spec.limit_ms - projected, 0.0)
        if overage > 0:
            action = spec.breach_action
            should_stop = action == BreachAction.STOP
        else:
            action = BreachAction.WARN if spec.mode == BudgetMode.SOFT else BreachAction.STOP
            should_stop = False
        return BudgetDecision(
            scope_id=scope_id,
            should_stop=should_stop,
            action=action,
            remaining_ms=remaining,
            overage_ms=overage,
        )

    def preflight(self, scope_type: str, scope_id: str, estimate: CostSnapshot) -> BudgetDecision:
        state = self._ensure_scope(scope_type, scope_id)
        projected = state.spent.milliseconds + estimate.milliseconds
        return self._make_decision(scope_id, state.spec, projected)

    def commit(self, scope_type: str, scope_id: str, actual: CostSnapshot) -> BudgetDecision:
        state = self._ensure_scope(scope_type, scope_id)
        new_spent = state.spent.add(actual)
        state.spent = new_spent  # type: ignore[misc]
        projected = new_spent.milliseconds
        return self._make_decision(scope_id, state.spec, projected)
