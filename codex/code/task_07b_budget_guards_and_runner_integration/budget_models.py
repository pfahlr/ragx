"""Budget domain models for FlowRunner integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional


class BudgetMode(str, Enum):
    """Supported budget enforcement modes."""

    HARD = "hard"
    SOFT = "soft"


class BreachAction(str, Enum):
    """Action to take when a budget breach is detected."""

    STOP = "stop"
    WARN = "warn"


class BudgetDecision(str, Enum):
    """Outcome of a budget check."""

    ALLOW = "allow"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class CostAmount:
    """Immutable representation of a normalized cost amount."""

    value: Decimal
    unit: str = "credits"

    @classmethod
    def of(cls, value: str | int | float | Decimal, unit: str = "credits") -> "CostAmount":
        decimal_value = Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if decimal_value < 0:
            raise ValueError("CostAmount cannot be negative")
        return cls(value=decimal_value, unit=unit)

    def add(self, other: "CostAmount") -> "CostAmount":
        self._ensure_compatible(other)
        return CostAmount.of(self.value + other.value, unit=self.unit)

    def subtract(self, other: "CostAmount") -> "CostAmount":
        self._ensure_compatible(other)
        new_value = self.value - other.value
        return CostAmount.of(max(new_value, Decimal("0")), unit=self.unit)

    def _ensure_compatible(self, other: "CostAmount") -> None:
        if self.unit != other.unit:
            raise ValueError("CostAmount units must match")


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable snapshot of a scope's budget state after a check or charge."""

    limit: Decimal
    spent: Decimal
    remaining: Decimal
    overage: Decimal

    @classmethod
    def from_values(
        cls, *, limit: CostAmount, spent: Decimal, projected: Decimal
    ) -> "CostSnapshot":
        remaining = max(limit.value - projected, Decimal("0"))
        overage = max(projected - limit.value, Decimal("0"))
        return cls(limit=limit.value, spent=spent, remaining=remaining, overage=overage)


@dataclass(frozen=True)
class BudgetBreach:
    """Metadata about a budget breach."""

    reason: str
    scope_id: str
    decision: BudgetDecision


class BudgetBreachError(RuntimeError):
    """Raised when a budget breach prevents commit."""


@dataclass(frozen=True)
class BudgetSpec:
    """Configuration describing a budget scope."""

    scope_id: str
    limit: CostAmount
    mode: BudgetMode = BudgetMode.HARD
    breach_action: BreachAction = BreachAction.STOP

    def resolve_decision(self, snapshot: CostSnapshot) -> tuple[BudgetDecision, Optional[BudgetBreach]]:
        if snapshot.overage <= 0:
            return BudgetDecision.ALLOW, None

        if self.mode is BudgetMode.HARD or self.breach_action is BreachAction.STOP:
            return BudgetDecision.STOP, BudgetBreach(
                reason="limit_exceeded",
                scope_id=self.scope_id,
                decision=BudgetDecision.STOP,
            )
        return BudgetDecision.WARN, BudgetBreach(
            reason="limit_near",
            scope_id=self.scope_id,
            decision=BudgetDecision.WARN,
        )


@dataclass(frozen=True)
class BudgetCheck:
    """Result of previewing a charge against the current budget."""

    scope_id: str
    snapshot: CostSnapshot
    decision: BudgetDecision
    breach: Optional[BudgetBreach] = None


@dataclass(frozen=True)
class BudgetCharge:
    """Result of committing a cost to a budget."""

    scope_id: str
    snapshot: CostSnapshot
    decision: BudgetDecision
    breach: Optional[BudgetBreach] = None


@dataclass(frozen=True)
class LoopSummary:
    """Aggregate metrics for a scope after execution."""

    scope_id: str
    total_spent: Decimal
    total_remaining: Decimal
    total_overage: Decimal


__all__ = [
    "BreachAction",
    "BudgetBreach",
    "BudgetBreachError",
    "BudgetCharge",
    "BudgetCheck",
    "BudgetDecision",
    "BudgetMode",
    "BudgetSpec",
    "CostAmount",
    "CostSnapshot",
    "LoopSummary",
]
