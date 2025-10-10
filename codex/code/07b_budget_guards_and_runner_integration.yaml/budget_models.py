"""Typed budget value objects for FlowRunner integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Dict, Mapping, Optional


class BreachAction(str, Enum):
    """Enumerates the supported breach actions."""

    STOP = "stop"
    WARN = "warn"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "BreachAction":
        if value is None:
            return cls.STOP
        value_normalized = value.lower()
        for member in cls:
            if member.value == value_normalized:
                return member
        raise ValueError(f"Unsupported breach_action: {value}")


@dataclass(frozen=True)
class ScopeKey:
    """Identifies a budget scope (run, node, spec, loop)."""

    category: str
    identifier: str

    VALID_CATEGORIES = {"run", "node", "spec", "loop"}

    def __post_init__(self) -> None:
        if self.category not in self.VALID_CATEGORIES:
            raise ValueError(f"Unsupported scope category: {self.category}")
        if not self.identifier:
            raise ValueError("Scope identifier must be provided")

    def as_tuple(self) -> tuple[str, str]:
        return self.category, self.identifier


@dataclass(frozen=True)
class CostSnapshot:
    """Represents a normalized cost in milliseconds."""

    milliseconds: float

    def __post_init__(self) -> None:
        if self.milliseconds < 0:
            raise ValueError("Cost cannot be negative")

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls(0.0)

    @classmethod
    def from_inputs(
        cls,
        *,
        duration_ms: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        payload: Optional[Mapping[str, float]] = None,
    ) -> "CostSnapshot":
        total_ms = 0.0
        if payload is not None:
            if "duration_ms" in payload:
                duration_ms = payload["duration_ms"]
            elif "duration_seconds" in payload:
                duration_seconds = payload["duration_seconds"]
        if duration_ms is not None:
            total_ms += float(duration_ms)
        if duration_seconds is not None:
            total_ms += float(duration_seconds) * 1000.0
        if payload is None and duration_ms is None and duration_seconds is None:
            raise ValueError("At least one cost input must be supplied")
        if total_ms < 0.0:
            raise ValueError("Cost cannot be negative")
        return cls(total_ms)

    def __add__(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(self.milliseconds + other.milliseconds)

    def __sub__(self, other: "CostSnapshot") -> "CostSnapshot":
        result = self.milliseconds - other.milliseconds
        if result < 0:
            result = 0.0
        return CostSnapshot(result)

    def to_payload(self) -> Mapping[str, float]:
        return MappingProxyType({"duration_ms": self.milliseconds})


@dataclass(frozen=True)
class BudgetSpec:
    """Immutable budget specification for a scope."""

    scope: ScopeKey
    limit_ms: float
    breach_action: BreachAction

    @classmethod
    def from_config(cls, scope: ScopeKey, config: Mapping[str, float]) -> "BudgetSpec":
        if "limit_ms" in config:
            limit_ms = float(config["limit_ms"])
        elif "limit_seconds" in config:
            limit_ms = float(config["limit_seconds"]) * 1000.0
        else:
            raise ValueError("Budget config must provide limit_ms or limit_seconds")
        if limit_ms <= 0:
            raise ValueError("Budget limit must be positive")
        action_value = config.get("breach_action") if hasattr(config, "get") else None
        action = BreachAction.from_value(action_value)
        return cls(scope=scope, limit_ms=limit_ms, breach_action=action)


@dataclass(frozen=True)
class BudgetBreach:
    """Describes an attempted spend that exceeded the limit."""

    scope: ScopeKey
    attempted: CostSnapshot
    limit_ms: float
    remaining: CostSnapshot
    action: BreachAction

    def to_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "scope": self.scope.category,
                "scope_id": self.scope.identifier,
                "attempted_ms": self.attempted.milliseconds,
                "limit_ms": self.limit_ms,
                "remaining_ms": self.remaining.milliseconds,
                "action": self.action.value,
            }
        )


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome of a budget evaluation stage."""

    scope: ScopeKey
    stage: str
    attempted: CostSnapshot
    remaining: CostSnapshot
    allowed: bool
    action: BreachAction
    breach: Optional[BudgetBreach] = None

    def to_payload(self) -> Mapping[str, object]:
        payload: Dict[str, object] = {
            "scope": self.scope.category,
            "scope_id": self.scope.identifier,
            "stage": self.stage,
            "attempted_ms": self.attempted.milliseconds,
            "remaining_ms": self.remaining.milliseconds,
            "allowed": self.allowed,
            "action": self.action.value,
        }
        if self.breach is not None:
            payload["breach"] = self.breach.to_payload()
        return MappingProxyType(payload)


@dataclass(frozen=True)
class ScopeSnapshot:
    """Represents an immutable view of scope spend."""

    scope: ScopeKey
    limit_ms: float
    spent: CostSnapshot

    def to_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "scope": self.scope.category,
                "scope_id": self.scope.identifier,
                "limit_ms": self.limit_ms,
                "spent_ms": self.spent.milliseconds,
            }
        )
