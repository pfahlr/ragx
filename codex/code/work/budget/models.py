"""Immutable budget and cost data models used by the FlowRunner."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, MutableMapping, Sequence

_METRIC_FIELDS = ("calls", "tokens_in", "tokens_out", "seconds")

__all__ = [
    "CostSnapshot",
    "BudgetSpec",
    "BudgetCharge",
    "BudgetChargeOutcome",
]


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable cost metrics for a tool invocation."""

    calls: float = 0.0
    tokens_in: float = 0.0
    tokens_out: float = 0.0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        for field in _METRIC_FIELDS:
            value = float(getattr(self, field))
            if value < 0:
                raise ValueError(f"{field} cannot be negative")
            object.__setattr__(self, field, value)

    @classmethod
    def zero(cls) -> CostSnapshot:
        return cls()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> CostSnapshot:
        values = {field: float(mapping.get(field, 0.0) or 0.0) for field in _METRIC_FIELDS}
        return cls(**values)  # type: ignore[arg-type]

    def as_mapping(self) -> Mapping[str, float]:
        return MappingProxyType({field: float(getattr(self, field)) for field in _METRIC_FIELDS})

    def add(self, other: CostSnapshot) -> CostSnapshot:
        return CostSnapshot(
            **{field: getattr(self, field) + getattr(other, field) for field in _METRIC_FIELDS}
        )

    def subtract(self, other: CostSnapshot) -> CostSnapshot:
        values: dict[str, float] = {}
        for field in _METRIC_FIELDS:
            result = getattr(self, field) - getattr(other, field)
            if result < 0:
                raise ValueError(f"{field} would become negative")
            values[field] = result
        return CostSnapshot(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Configuration for a scoped budget meter."""

    scope: str
    scope_id: str
    limit: CostSnapshot
    mode: str
    breach_action: str

    @classmethod
    def from_mapping(cls, *, scope: str, scope_id: str, data: Mapping[str, object]) -> BudgetSpec:
        try:
            limit_data = data["limit"]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError("budget spec requires a limit") from exc
        if not isinstance(limit_data, Mapping):
            raise ValueError("budget limit must be a mapping")
        limit = CostSnapshot.from_mapping(limit_data)
        mode = str(data.get("mode", "soft"))
        if mode not in {"soft", "hard"}:
            raise ValueError("mode must be 'soft' or 'hard'")
        breach_action = str(data.get("breach_action", "warn"))
        if breach_action not in {"warn", "stop"}:
            raise ValueError("breach_action must be 'warn' or 'stop'")
        return cls(
            scope=scope,
            scope_id=scope_id,
            limit=limit,
            mode=mode,
            breach_action=breach_action,
        )

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "scope": self.scope,
                "scope_id": self.scope_id,
                "mode": self.mode,
                "breach_action": self.breach_action,
                "limit": dict(self.limit.as_mapping()),
            }
        )


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    """Resulting charge for a budget meter interaction."""

    scope_type: str
    scope_id: str
    spec: BudgetSpec
    cost: CostSnapshot
    spent_before: CostSnapshot
    spent_after: CostSnapshot
    remaining: Mapping[str, float]
    overage: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    """Envelope returned by BudgetMeter and BudgetManager operations."""

    charge: BudgetCharge
    breached: bool
    stop: bool
    reasons: Sequence[str] = ()

    def to_trace_payload(
        self,
        *,
        context: Mapping[str, object] | MutableMapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "scope_type": self.charge.scope_type,
            "scope_id": self.charge.scope_id,
            "spec": self.charge.spec.as_mapping(),
            "cost": dict(self.charge.cost.as_mapping()),
            "spent": dict(self.charge.spent_after.as_mapping()),
            "remaining": dict(self.charge.remaining),
            "overage": dict(self.charge.overage),
            "breached": self.breached,
            "stop": self.stop,
            "reasons": tuple(self.reasons),
        }
        if context is not None:
            payload["context"] = dict(context)
        return MappingProxyType(payload)


def remaining_after(limit: CostSnapshot, spent: CostSnapshot) -> Mapping[str, float]:
    data: dict[str, float] = {}
    for field in _METRIC_FIELDS:
        data[field] = max(0.0, getattr(limit, field) - getattr(spent, field))
    return MappingProxyType(data)


def overage_after(limit: CostSnapshot, spent: CostSnapshot) -> Mapping[str, float]:
    data: dict[str, float] = {}
    for field in _METRIC_FIELDS:
        data[field] = max(0.0, getattr(spent, field) - getattr(limit, field))
    return MappingProxyType(data)
