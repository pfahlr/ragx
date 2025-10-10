"""Canonical budget domain objects shared across FlowRunner integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable

__all__ = [
    "BudgetMode",
    "BudgetSpec",
    "CostSnapshot",
    "BudgetCharge",
    "BudgetPreview",
    "mapping_proxy",
]


def mapping_proxy(data: Mapping[str, float] | None = None) -> Mapping[str, float]:
    """Return an immutable mapping for trace payload safety."""

    return MappingProxyType(dict(data or {}))


class BudgetMode(str, Enum):
    """Budget breach handling modes."""

    WARN = "warn"
    STOP = "stop"
    UNLIMITED = "unlimited"


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Immutable specification describing cost limits and breach mode."""

    limits: Mapping[str, float] = field(default_factory=dict)
    mode: BudgetMode = BudgetMode.STOP

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        normalized = {str(key): float(value) for key, value in self.limits.items()}
        object.__setattr__(self, "limits", mapping_proxy(normalized))
        if not isinstance(self.mode, BudgetMode):
            object.__setattr__(self, "mode", BudgetMode(str(self.mode)))


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable representation of normalized cost metrics."""

    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        normalized = {str(key): float(value) for key, value in self.metrics.items()}
        object.__setattr__(self, "metrics", mapping_proxy(normalized))

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls(metrics={})

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        keys: set[str] = set(self.metrics) | set(other.metrics)
        combined = {key: self.metrics.get(key, 0.0) + other.metrics.get(key, 0.0) for key in keys}
        return CostSnapshot(metrics=combined)


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    """Outcome of applying cost against a specific budget scope."""

    scope_id: str
    scope_type: str
    mode: BudgetMode
    cost: CostSnapshot
    spent: CostSnapshot
    remaining: Mapping[str, float]
    overages: Mapping[str, float]
    breached: bool

    @property
    def hard_breach(self) -> bool:
        return self.mode == BudgetMode.STOP and self.breached


@dataclass(frozen=True, slots=True)
class BudgetPreview:
    """Preview of a cost charge across a scope chain."""

    scope_id: str
    charges: tuple[BudgetCharge, ...]
    cost: CostSnapshot

    @property
    def has_breach(self) -> bool:
        return any(charge.breached for charge in self.charges)

    @property
    def hard_breach(self) -> bool:
        return any(charge.hard_breach for charge in self.charges)

    @property
    def scope_ids(self) -> Iterable[str]:
        return (charge.scope_id for charge in self.charges)

