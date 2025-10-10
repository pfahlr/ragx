"""Immutable budget domain models and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "BudgetMode",
    "BudgetScope",
    "BudgetSpec",
    "BudgetBreach",
    "BudgetChargeOutcome",
    "CostSnapshot",
]


class BudgetMode(str, Enum):
    """Enumeration of supported budget enforcement modes."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class BudgetScope:
    """Identify a budget scope using type + identifier."""

    scope_type: str
    identifier: str

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.scope_type}:{self.identifier}"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Normalized, immutable representation of spend or limits."""

    milliseconds: int
    tokens_in: int
    tokens_out: int
    calls: int

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "milliseconds", int(self.milliseconds))
        object.__setattr__(self, "tokens_in", int(self.tokens_in))
        object.__setattr__(self, "tokens_out", int(self.tokens_out))
        object.__setattr__(self, "calls", int(self.calls))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls(milliseconds=0, tokens_in=0, tokens_out=0, calls=0)

    @classmethod
    def from_seconds(
        cls,
        *,
        seconds: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        calls: int = 0,
    ) -> "CostSnapshot":
        milliseconds = int(round(seconds * 1000))
        return cls(
            milliseconds=milliseconds,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            calls=calls,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "CostSnapshot":
        if not data:
            return cls.zero()
        milliseconds = data.get("milliseconds")
        seconds = data.get("seconds")
        if milliseconds is None and seconds is None:
            milliseconds_value = 0
        elif milliseconds is not None:
            milliseconds_value = int(milliseconds)
        else:
            milliseconds_value = int(round(float(seconds) * 1000))
        return cls(
            milliseconds=milliseconds_value,
            tokens_in=int(data.get("tokens_in", 0)),
            tokens_out=int(data.get("tokens_out", 0)),
            calls=int(data.get("calls", 0)),
        )

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------
    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            milliseconds=self.milliseconds + other.milliseconds,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            calls=self.calls + other.calls,
        )

    def subtract(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            milliseconds=max(0, self.milliseconds - other.milliseconds),
            tokens_in=max(0, self.tokens_in - other.tokens_in),
            tokens_out=max(0, self.tokens_out - other.tokens_out),
            calls=max(0, self.calls - other.calls),
        )

    def exceeds(self, limit: "CostSnapshot") -> bool:
        return (
            self.milliseconds > limit.milliseconds
            or (limit.tokens_in > 0 and self.tokens_in > limit.tokens_in)
            or (limit.tokens_out > 0 and self.tokens_out > limit.tokens_out)
            or (limit.calls > 0 and self.calls > limit.calls)
        )

    def overage(self, limit: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            milliseconds=max(0, self.milliseconds - limit.milliseconds),
            tokens_in=max(0, self.tokens_in - limit.tokens_in) if limit.tokens_in > 0 else 0,
            tokens_out=max(0, self.tokens_out - limit.tokens_out) if limit.tokens_out > 0 else 0,
            calls=max(0, self.calls - limit.calls) if limit.calls > 0 else 0,
        )

    def to_dict(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "milliseconds": self.milliseconds,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "calls": self.calls,
            }
        )

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def __add__(self, other: "CostSnapshot") -> "CostSnapshot":  # pragma: no cover - thin wrapper
        return self.add(other)

    def __sub__(self, other: "CostSnapshot") -> "CostSnapshot":  # pragma: no cover - thin wrapper
        return self.subtract(other)


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Immutable budget specification for a scope."""

    scope: BudgetScope
    limit: CostSnapshot
    mode: BudgetMode
    breach_action: str

    @classmethod
    def from_dict(cls, scope: BudgetScope, data: Mapping[str, object]) -> "BudgetSpec":
        if "limit" not in data:
            raise ValueError("budget spec requires 'limit'")
        limit = CostSnapshot.from_mapping(data["limit"])
        mode = BudgetMode(data.get("mode", BudgetMode.HARD))
        breach_action = str(data.get("breach_action", "stop"))
        return cls(scope=scope, limit=limit, mode=mode, breach_action=breach_action)


@dataclass(frozen=True, slots=True)
class BudgetBreach:
    """Outcome when a budget exceeded its limit."""

    scope: BudgetScope
    mode: BudgetMode
    action: str
    overage: CostSnapshot


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    """Result of applying a charge to a budget scope."""

    scope: BudgetScope
    spent: CostSnapshot
    total_spent: CostSnapshot
    remaining: CostSnapshot
    overage: CostSnapshot
    mode: BudgetMode
    action: str

    def to_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "scope_type": self.scope.scope_type,
                "scope_id": self.scope.identifier,
                "spent": dict(self.spent.to_dict()),
                "total_spent": dict(self.total_spent.to_dict()),
                "remaining": dict(self.remaining.to_dict()),
                "overage": dict(self.overage.to_dict()),
                "mode": self.mode.value,
                "action": self.action,
            }
        )
