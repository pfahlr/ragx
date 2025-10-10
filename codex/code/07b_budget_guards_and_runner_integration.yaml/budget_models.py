from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, MutableMapping


__all__ = [
    "BudgetMode",
    "BudgetSpec",
    "CostSnapshot",
    "BudgetChargeResult",
    "BudgetMeter",
    "BudgetBreachError",
]


class BudgetMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    usd: float
    calls: int
    tokens: int
    time_ms: int

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls(usd=0.0, calls=0, tokens=0, time_ms=0)

    @classmethod
    def from_values(
        cls,
        *,
        usd: float | int = 0.0,
        calls: int = 0,
        tokens: int = 0,
        time_ms: int | None = None,
        time_seconds: float | int | None = None,
    ) -> "CostSnapshot":
        if time_ms is not None and time_seconds is not None:
            raise ValueError("Provide either time_ms or time_seconds, not both")
        if time_ms is None and time_seconds is not None:
            time_ms = int(round(float(time_seconds) * 1000))
        if time_ms is None:
            time_ms = 0
        return cls(
            usd=float(usd),
            calls=int(calls),
            tokens=int(tokens),
            time_ms=int(time_ms),
        )

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            usd=self.usd + other.usd,
            calls=self.calls + other.calls,
            tokens=self.tokens + other.tokens,
            time_ms=self.time_ms + other.time_ms,
        )

    def as_mapping(self) -> Mapping[str, float | int]:
        return MappingProxyType(
            {
                "usd": self.usd,
                "calls": self.calls,
                "tokens": self.tokens,
                "time_ms": self.time_ms,
            }
        )


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    scope: str
    mode: BudgetMode
    limits: Mapping[str, float | int]
    breach_action: str = "error"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        normalized: MutableMapping[str, float | int] = {}
        for raw_key, raw_value in (self.limits or {}).items():
            key, value = self._normalize_limit(raw_key, raw_value)
            if value < 0:
                value = 0
            normalized[key] = value
        object.__setattr__(self, "limits", MappingProxyType(dict(normalized)))
        object.__setattr__(self, "breach_action", str(self.breach_action or "warn"))

    @staticmethod
    def _normalize_limit(key: str, value: float | int) -> tuple[str, float | int]:
        key = key.lower()
        mapping = {
            "max_usd": "usd",
            "usd": "usd",
            "max_calls": "calls",
            "calls": "calls",
            "max_tokens": "tokens",
            "tokens": "tokens",
            "time_ms": "time_ms",
            "time_limit_ms": "time_ms",
            "time_limit_sec": "time_ms",
            "time_seconds": "time_ms",
        }
        if key not in mapping:
            raise ValueError(f"Unsupported budget limit key: {key}")
        normalized_key = mapping[key]
        if normalized_key == "usd":
            normalized_value = float(value)
        elif normalized_key in {"calls", "tokens"}:
            normalized_value = int(value)
        else:  # time_ms
            if key in {"time_limit_sec", "time_seconds"}:
                normalized_value = int(round(float(value) * 1000))
            else:
                normalized_value = int(value)
        return normalized_key, normalized_value


@dataclass(frozen=True, slots=True)
class BudgetChargeResult:
    scope: str
    label: str
    cost: CostSnapshot
    total: CostSnapshot
    remaining: Mapping[str, float | int]
    overages: Mapping[str, float | int]
    breached: bool
    breach_kind: BudgetMode | None
    breach_action: str | None

    @property
    def should_stop(self) -> bool:
        if not self.breached:
            return False
        if self.breach_kind is BudgetMode.HARD:
            return True
        if not self.breach_action:
            return False
        return self.breach_action.lower() in {"stop", "error"}


class BudgetBreachError(RuntimeError):
    def __init__(self, result: BudgetChargeResult):
        message = (
            f"Budget breach on {result.scope}: action={result.breach_action} "
            f"remaining={dict(result.remaining)} overages={dict(result.overages)}"
        )
        super().__init__(message)
        self.result = result


class BudgetMeter:
    def __init__(self, spec: BudgetSpec) -> None:
        self.spec = spec
        self._spend = CostSnapshot.zero()

    def preview(self, cost: CostSnapshot, *, label: str = "preview") -> BudgetChargeResult:
        return self._calculate(cost, label, mutate=False)

    def charge(self, cost: CostSnapshot, *, label: str) -> BudgetChargeResult:
        return self._calculate(cost, label, mutate=True)

    def _calculate(self, cost: CostSnapshot, label: str, *, mutate: bool) -> BudgetChargeResult:
        total = self._spend.add(cost)
        if mutate:
            self._spend = total

        remaining: MutableMapping[str, float | int] = {}
        overages: MutableMapping[str, float | int] = {}
        breached = False
        for key, limit in self.spec.limits.items():
            spent_value = getattr(total, key)
            remaining_value: float | int
            overage_value: float | int
            if limit == 0:
                remaining_value = 0
                overage_value = max(spent_value - limit, 0)
            else:
                remaining_value = max(limit - spent_value, 0)
                overage_value = max(spent_value - limit, 0)
            remaining[key] = remaining_value
            overages[key] = overage_value
            if overage_value > 0:
                breached = True

        return BudgetChargeResult(
            scope=self.spec.scope,
            label=label,
            cost=cost,
            total=total,
            remaining=MappingProxyType(dict(remaining)),
            overages=MappingProxyType(dict(overages)),
            breached=breached,
            breach_kind=self.spec.mode if breached else None,
            breach_action=self.spec.breach_action if breached else None,
        )
