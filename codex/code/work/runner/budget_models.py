"""Budget domain models for FlowRunner integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

__all__ = [
    "BudgetMode",
    "BudgetScope",
    "BudgetSpec",
    "BudgetChargeOutcome",
    "CostSnapshot",
]


class BudgetMode(str, Enum):
    """Modes supported by budget specifications."""

    HARD = "hard"
    SOFT = "soft"


_LIMIT_KEYS = {
    "usd": "max_usd",
    "tokens": "max_tokens",
    "calls": "max_calls",
}


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        return Decimal(value)
    raise TypeError(f"Unsupported numeric value: {value!r}")


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable representation of metered costs."""

    _values: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: Dict[str, Decimal] = {metric: _to_decimal(amount) for metric, amount in self._values.items()}
        for key in _LIMIT_KEYS:
            normalized.setdefault(key, Decimal("0"))
        object.__setattr__(self, "_values", MappingProxyType(normalized))

    @classmethod
    def from_costs(cls, costs: Mapping[str, object]) -> "CostSnapshot":
        return cls(costs)

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls({})

    def as_dict(self) -> Mapping[str, Decimal]:
        return self._values

    def get(self, metric: str) -> Decimal:
        return self._values.get(metric, Decimal("0"))

    @property
    def usd(self) -> Decimal:
        return self.get("usd")

    @property
    def tokens(self) -> Decimal:
        return self.get("tokens")

    @property
    def calls(self) -> Decimal:
        return self.get("calls")

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        merged: Dict[str, Decimal] = {}
        for key in set(self._values) | set(other._values):
            merged[key] = self.get(key) + other.get(key)
        return CostSnapshot(merged)


@dataclass(frozen=True)
class BudgetScope:
    """Identifies the logical scope of a budget."""

    level: str
    identifier: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Mapping[str, object]:
        return {"level": self.level, "identifier": list(self.identifier)}

    @classmethod
    def run(cls, run_id: str) -> "BudgetScope":
        return cls(level="run", identifier=(run_id,))

    @classmethod
    def loop(cls, loop_id: str) -> "BudgetScope":
        return cls(level="loop", identifier=(loop_id,))

    @classmethod
    def node(cls, node_id: str) -> "BudgetScope":
        return cls(level="node", identifier=(node_id,))

    @classmethod
    def spec(cls, node_id: str, spec_name: str) -> "BudgetScope":
        return cls(level="spec", identifier=(node_id, spec_name))

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.level}:{'/'.join(self.identifier)}"


@dataclass(frozen=True)
class BudgetSpec:
    """Configuration for a budget scope."""

    mode: BudgetMode
    limits: Mapping[str, Decimal]
    breach_action: str
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BudgetSpec":
        mode = BudgetMode(str(data.get("mode", BudgetMode.HARD.value)))
        breach_action = str(data.get("breach_action", "warn" if mode is BudgetMode.SOFT else "error"))
        limits: Dict[str, Decimal] = {}
        for metric, key in _LIMIT_KEYS.items():
            if key in data and data[key] is not None:
                limits[metric] = _to_decimal(data[key])
        return cls(mode=mode, limits=MappingProxyType(limits), breach_action=breach_action, name=str(data.get("name")) if data.get("name") else None)

    def has_limit(self, metric: str) -> bool:
        return metric in self.limits

    def limit_for(self, metric: str) -> Optional[Decimal]:
        return self.limits.get(metric)


@dataclass(frozen=True)
class BudgetChargeOutcome:
    """Result of charging a budget scope."""

    scope: BudgetScope
    spec: BudgetSpec
    cost: CostSnapshot
    spent: CostSnapshot
    remaining: Mapping[str, Decimal]
    overages: Mapping[str, Decimal]
    breached: bool
    action: str

    @classmethod
    def build(
        cls,
        *,
        scope: BudgetScope,
        spec: BudgetSpec,
        cost: CostSnapshot,
        spent: CostSnapshot,
    ) -> "BudgetChargeOutcome":
        remaining: MutableMapping[str, Decimal] = {}
        overages: MutableMapping[str, Decimal] = {}
        breached = False
        for metric in _LIMIT_KEYS:
            limit = spec.limit_for(metric)
            value = spent.get(metric)
            if limit is None:
                continue
            remaining_value = limit - value
            remaining[metric] = remaining_value
            breach_here = remaining_value < 0
            if spec.breach_action == "stop" and remaining_value <= 0:
                breach_here = True
            if breach_here:
                breached = True
                overages[metric] = -remaining_value if remaining_value < 0 else Decimal("0")
            else:
                overages[metric] = Decimal("0")
        outcome_action = spec.breach_action if breached else "none"
        return cls(
            scope=scope,
            spec=spec,
            cost=cost,
            spent=spent,
            remaining=MappingProxyType(dict(remaining)),
            overages=MappingProxyType(dict(overages)),
            breached=breached,
            action=outcome_action,
        )

    def to_trace_payload(self) -> Mapping[str, object]:
        return {
            "scope": self.scope.as_dict(),
            "cost": {k: str(v) for k, v in self.cost.as_dict().items()},
            "spent": {k: str(v) for k, v in self.spent.as_dict().items()},
            "remaining": {k: str(v) for k, v in self.remaining.items()},
            "overages": {k: str(v) for k, v in self.overages.items()},
            "breached": self.breached,
            "action": self.action,
            "mode": self.spec.mode.value,
            "name": self.spec.name,
        }
