"""Budget domain models and manager for Phase 3 sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, MutableMapping


class BudgetMode(str, Enum):
    """Budget enforcement modes."""

    HARD = "hard"
    SOFT = "soft"


class BreachAction(str, Enum):
    """Actions to take when a breach occurs."""

    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class ScopeKey:
    """Identifies a budget scope by type and identifier."""

    scope_type: str
    scope_id: str

    def __str__(self) -> str:  # pragma: no cover - convenience for debugging
        return f"{self.scope_type}:{self.scope_id}"


@dataclass(frozen=True)
class Cost:
    """Normalized cost snapshot in milliseconds."""

    milliseconds: float

    def __post_init__(self) -> None:
        normalized = round(float(self.milliseconds), 6)
        if abs(normalized) < 1e-3:
            normalized = 0.0
        object.__setattr__(self, "milliseconds", max(0.0, normalized))

    @classmethod
    def from_seconds(cls, seconds: float) -> "Cost":
        return cls(milliseconds=seconds * 1000.0)


@dataclass(frozen=True)
class BudgetSpec:
    """Configuration for a budget scope."""

    scope_type: str
    scope_id: str
    limit_ms: float
    mode: BudgetMode
    breach_action: BreachAction

    def __post_init__(self) -> None:
        if self.limit_ms < 0:
            raise ValueError("Budget limit must be non-negative")


@dataclass(frozen=True)
class BudgetBreach:
    """Details about a budget breach."""

    scope: ScopeKey
    limit_ms: float
    attempted_ms: float
    mode: BudgetMode
    action: BreachAction

    @property
    def breach_kind(self) -> str:
        return "soft" if self.mode is BudgetMode.SOFT else "hard"


class BudgetDecisionStatus(str, Enum):
    """Decision results produced during budget evaluation."""

    ALLOW = "allow"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class BudgetDecision:
    """Immutable decision returned during preflight or commit."""

    status: BudgetDecisionStatus
    breach: BudgetBreach | None = None

    @property
    def requires_stop(self) -> bool:
        return self.status is BudgetDecisionStatus.STOP


@dataclass(frozen=True)
class BudgetCharge:
    """Charge snapshot describing spend, remaining, and overages."""

    scope: ScopeKey
    spent_ms: float
    remaining_ms: float
    overage_ms: float

    def as_payload(self) -> Mapping[str, float]:
        payload = {
            "spent_ms": self.spent_ms,
            "remaining_ms": self.remaining_ms,
            "overage_ms": self.overage_ms,
        }
        return MappingProxyType(payload)


@dataclass(frozen=True)
class BudgetChargeOutcome:
    """Result returned from BudgetManager.commit."""

    decision: BudgetDecision
    charge: BudgetCharge
    breach: BudgetBreach | None

    @property
    def requires_stop(self) -> bool:
        return self.decision.requires_stop


class BudgetManager:
    """Coordinates budget scopes and charges."""

    def __init__(self, specs: Mapping[ScopeKey, BudgetSpec]) -> None:
        self._specs: Mapping[ScopeKey, BudgetSpec] = MappingProxyType(dict(specs))
        self._spent: MutableMapping[ScopeKey, float] = {key: 0.0 for key in self._specs}

    def spec_for(self, scope: ScopeKey) -> BudgetSpec:
        try:
            return self._specs[scope]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Unknown scope {scope}") from exc

    def preflight(self, scope: ScopeKey, cost: Cost) -> BudgetDecision:
        spec = self._specs[scope]
        spent = self._spent[scope]
        attempted = spent + cost.milliseconds
        overage = max(0.0, attempted - spec.limit_ms)
        if overage <= 0.0:
            return BudgetDecision(status=BudgetDecisionStatus.ALLOW)

        breach = BudgetBreach(
            scope=scope,
            limit_ms=spec.limit_ms,
            attempted_ms=attempted,
            mode=spec.mode,
            action=spec.breach_action,
        )
        if spec.breach_action is BreachAction.WARN and spec.mode is BudgetMode.SOFT:
            status = BudgetDecisionStatus.WARN
        else:
            status = BudgetDecisionStatus.STOP
        return BudgetDecision(status=status, breach=breach)

    def commit(self, scope: ScopeKey, cost: Cost) -> BudgetChargeOutcome:
        decision = self.preflight(scope, cost)
        spec = self._specs[scope]
        previous_spent = self._spent[scope]
        attempted = previous_spent + cost.milliseconds
        remaining_after_attempt = max(0.0, spec.limit_ms - min(spec.limit_ms, attempted))
        overage = max(0.0, attempted - spec.limit_ms)
        charge = BudgetCharge(
            scope=scope,
            spent_ms=attempted,
            remaining_ms=remaining_after_attempt,
            overage_ms=overage,
        )
        # Clamp stored spend to the limit to avoid infinite overage accumulation while preserving totals.
        self._spent[scope] = min(spec.limit_ms, attempted)
        return BudgetChargeOutcome(decision=decision, charge=charge, breach=decision.breach)

    def remaining(self, scope: ScopeKey) -> float:
        spec = self._specs[scope]
        spent = self._spent[scope]
        return max(0.0, spec.limit_ms - spent)

    def scopes(self) -> Iterable[ScopeKey]:  # pragma: no cover - utility
        return self._specs.keys()

    @property
    def specs(self) -> Mapping[ScopeKey, BudgetSpec]:
        return self._specs
