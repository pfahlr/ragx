"""Budget domain models and exceptions for FlowRunner integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Dict, Mapping

from .costs import accumulate_cost


class BudgetMode(str, Enum):
    """Enumerates budget enforcement intensity."""

    SOFT = "soft"
    HARD = "hard"


@dataclass(frozen=True)
class BudgetSpec:
    """Configuration for a single budget meter."""

    metric: str
    limit: int | float
    breach_action: str = "error"
    mode: BudgetMode = BudgetMode.HARD

    def normalized_limit(self) -> int:
        return int(round(float(self.limit)))


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable snapshot of spent and remaining budget."""

    metric: str
    spent: int
    remaining: int
    limit: int


@dataclass(frozen=True)
class BudgetChargeOutcome:
    """Result of a budget charge application."""

    scope_id: str
    action: str
    breached: bool
    cost: Mapping[str, int]
    remaining: int
    overage: int
    spec: BudgetSpec

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "scope_id": self.scope_id,
                "action": self.action,
                "breached": self.breached,
                "cost": dict(self.cost),
                "remaining": self.remaining,
                "overage": self.overage,
                "limit": self.spec.normalized_limit(),
                "metric": self.spec.metric,
            }
        )


class BudgetError(RuntimeError):
    """Base exception for budget enforcement."""

    def __init__(self, outcome: BudgetChargeOutcome):
        super().__init__(
            f"Budget breach in scope {outcome.scope_id}: action={outcome.action} overage={outcome.overage}"
        )
        self.outcome = outcome


class BudgetBreachError(BudgetError):
    """Raised when a hard budget is breached and execution must halt."""


class BudgetStopSignal(BudgetError):
    """Raised to request a controlled stop, typically for loop budgets."""


class BudgetMeter:
    """Tracks spend against a single budget specification."""

    def __init__(self, scope_id: str, spec: BudgetSpec) -> None:
        self.scope_id = scope_id
        self.spec = spec
        self._spent: Dict[str, int] = {}

    def snapshot(self) -> CostSnapshot:
        spent = int(self._spent.get(self.spec.metric, 0))
        limit = self.spec.normalized_limit()
        remaining = max(limit - spent, 0)
        return CostSnapshot(metric=self.spec.metric, spent=spent, remaining=remaining, limit=limit)

    def charge(self, cost: Mapping[str, int]) -> BudgetChargeOutcome:
        limit = self.spec.normalized_limit()
        metric_value = int(cost.get(self.spec.metric, 0))
        prior = int(self._spent.get(self.spec.metric, 0))
        new_spent = prior + metric_value
        remaining = max(limit - new_spent, 0)
        overage = max(new_spent - limit, 0)

        if metric_value:
            self._spent[self.spec.metric] = new_spent

        breached = new_spent > limit
        outcome = BudgetChargeOutcome(
            scope_id=self.scope_id,
            action=self.spec.breach_action,
            breached=breached,
            cost=MappingProxyType(dict(cost)),
            remaining=remaining,
            overage=overage,
            spec=self.spec,
        )

        if breached:
            if self.spec.breach_action == "warn" and self.spec.mode == BudgetMode.SOFT:
                return outcome
            if self.spec.breach_action == "stop":
                raise BudgetStopSignal(outcome)
            raise BudgetBreachError(outcome)

        return outcome

    def accumulate(self, additional_cost: Mapping[str, int]) -> None:
        self._spent = accumulate_cost(self._spent, additional_cost)
