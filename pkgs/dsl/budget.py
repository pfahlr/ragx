from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

__all__ = [
    "BudgetMode",
    "BudgetExceededError",
    "BudgetCheck",
    "BudgetCharge",
    "Cost",
    "CostBreakdown",
    "BudgetMeter",
]


@dataclass(frozen=True, slots=True)
class Cost:
    usd: float = 0.0
    tokens: int = 0
    calls: int = 0
    time_sec: float = 0.0

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "usd", float(self.usd or 0.0))
        object.__setattr__(self, "tokens", int(self.tokens or 0))
        object.__setattr__(self, "calls", int(self.calls or 0))
        object.__setattr__(self, "time_sec", float(self.time_sec or 0.0))

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            usd=math.fsum((self.usd, other.usd)),
            tokens=self.tokens + other.tokens,
            calls=self.calls + other.calls,
            time_sec=math.fsum((self.time_sec, other.time_sec)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "usd": self.usd,
            "tokens": self.tokens,
            "calls": self.calls,
            "time_sec": self.time_sec,
        }


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    max_usd: float
    max_tokens: float
    max_calls: float
    time_limit_sec: float
    total_spent: Cost

    def to_dict(self) -> dict[str, float | int | Mapping[str, float | int]]:
        return {
            "max_usd": self.max_usd,
            "max_tokens": self.max_tokens,
            "max_calls": self.max_calls,
            "time_limit_sec": self.time_limit_sec,
            "total_spent": self.total_spent.to_dict(),
        }


class BudgetMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"

    @classmethod
    def from_value(cls, value: BudgetMode | str | None) -> BudgetMode:
        if isinstance(value, BudgetMode):
            return value
        if value is None:
            return cls.HARD
        try:
            return cls(value.lower())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid budget mode: {value!r}") from exc


class BudgetExceededError(RuntimeError):
    def __init__(
        self,
        *,
        scope: str,
        metric: str,
        limit: float,
        spent: float,
        attempted: float,
        mode: BudgetMode,
    ) -> None:
        super().__init__(
            f"Budget exceeded for {scope}:{metric} (limit={limit}, attempted={attempted})"
        )
        self.scope = scope
        self.metric = metric
        self.limit = float(limit)
        self.spent = float(spent)
        self.attempted = float(attempted)
        self.mode = mode


@dataclass(frozen=True, slots=True)
class BudgetCheck:
    allowed: bool
    breach_kind: str | None
    metric: str | None
    limit: float | None
    attempted: float | None
    remaining: CostBreakdown


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    cost: Cost
    breached: bool
    breach_kind: str | None
    metric: str | None
    limit: float | None
    attempted: float | None
    remaining: CostBreakdown


@dataclass(slots=True)
class _BudgetLimits:
    max_usd: float | None
    max_tokens: int | None
    max_calls: int | None
    time_limit_sec: float | None

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> _BudgetLimits:
        return cls(
            max_usd=_normalize_float(spec.get("max_usd")),
            max_tokens=_normalize_int(spec.get("max_tokens")),
            max_calls=_normalize_int(spec.get("max_calls")),
            time_limit_sec=_normalize_float(spec.get("time_limit_sec")),
        )

    def remaining(self, spent: Cost) -> CostBreakdown:
        return CostBreakdown(
            max_usd=_remaining(self.max_usd, spent.usd),
            max_tokens=_remaining(self.max_tokens, spent.tokens),
            max_calls=_remaining(self.max_calls, spent.calls),
            time_limit_sec=_remaining(self.time_limit_sec, spent.time_sec),
            total_spent=spent,
        )


def _normalize_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return None if float(value) <= 0 else float(value)
    raise TypeError(f"expected numeric limit, got {type(value)!r}")


def _normalize_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return None if value <= 0 else value
    if isinstance(value, float):
        normalized = int(value)
        return None if normalized <= 0 else normalized
    raise TypeError(f"expected integer limit, got {type(value)!r}")


def _remaining(limit: float | int | None, spent: float | int) -> float:
    if limit is None:
        return math.inf
    return float(limit) - float(spent)


class BudgetMeter:
    def __init__(
        self,
        *,
        scope: str,
        label: str,
        limits: Mapping[str, object] | None = None,
        mode: BudgetMode = BudgetMode.HARD,
        breach_action: str | None = None,
    ) -> None:
        self._scope = scope
        self._label = label
        self._mode = BudgetMode.from_value(mode)
        self._limits = _BudgetLimits.from_spec(limits or {})
        self._spent = Cost()
        self._breach_action = breach_action or "error"
        self._epsilon = 1e-9

    @property
    def label(self) -> str:
        return self._label

    @property
    def mode(self) -> BudgetMode:
        return self._mode

    @property
    def breach_action(self) -> str:
        return self._breach_action

    @property
    def spent(self) -> Cost:
        return self._spent

    @property
    def limits(self) -> CostBreakdown:
        return self._limits.remaining(Cost())

    def snapshot(self) -> CostBreakdown:
        return self._limits.remaining(self._spent)

    def preview(self, cost: Cost | Mapping[str, object] | None) -> BudgetCheck:
        normalized = _normalize_cost(cost)
        projected = self._spent + normalized
        metric, limit_value, attempted_value, kind = self._first_breach(projected)
        allowed = True
        if kind == "hard":
            allowed = False
        elif kind == "soft":
            allowed = True
        remaining = self._limits.remaining(projected if allowed else self._spent)
        return BudgetCheck(
            allowed=allowed,
            breach_kind=kind,
            metric=metric,
            limit=limit_value,
            attempted=attempted_value,
            remaining=remaining,
        )

    def can_spend(self, cost: Cost | Mapping[str, object] | None) -> bool:
        check = self.preview(cost)
        return check.allowed

    def charge(self, cost: Cost | Mapping[str, object] | None) -> BudgetCharge:
        normalized = _normalize_cost(cost)
        projected = self._spent + normalized
        metric, limit_value, attempted_value, kind = self._first_breach(projected)
        if kind == "hard":
            raise BudgetExceededError(
                scope=self._label,
                metric=metric or "usd",
                limit=limit_value or 0.0,
                spent=_resolve_spent(self._spent, metric),
                attempted=attempted_value or 0.0,
                mode=self._mode,
            )
        self._spent = projected
        remaining = self._limits.remaining(self._spent)
        breached = kind == "soft"
        return BudgetCharge(
            cost=normalized,
            breached=breached,
            breach_kind="soft" if breached else None,
            metric=metric,
            limit=limit_value,
            attempted=attempted_value,
            remaining=remaining,
        )

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, object] | None,
        *,
        scope: str,
        label: str,
        default_mode: BudgetMode | str = BudgetMode.HARD,
        breach_action: str | None = None,
    ) -> BudgetMeter:
        data = dict(spec or {})
        mode_value = cast(BudgetMode | str | None, data.get("mode", default_mode))
        limits = {
            key: value
            for key, value in data.items()
            if key in {"max_usd", "max_tokens", "max_calls", "time_limit_sec"}
        }
        return cls(
            scope=scope,
            label=label,
            limits=limits,
            mode=BudgetMode.from_value(mode_value),
            breach_action=cast(str | None, data.get("breach_action", breach_action)),
        )

    def remaining(self) -> CostBreakdown:
        return self._limits.remaining(self._spent)

    def _first_breach(
        self, projected: Cost
    ) -> tuple[str | None, float | None, float | None, str | None]:
        checks = [
            ("usd", self._limits.max_usd, projected.usd),
            ("tokens", self._limits.max_tokens, float(projected.tokens)),
            ("calls", self._limits.max_calls, float(projected.calls)),
            ("time_sec", self._limits.time_limit_sec, projected.time_sec),
        ]
        for metric, limit_value, attempted in checks:
            if limit_value is None:
                continue
            if attempted - float(limit_value) > self._epsilon:
                if self._mode is BudgetMode.SOFT:
                    return metric, float(limit_value), attempted, "soft"
                return metric, float(limit_value), attempted, "hard"
        return None, None, None, None


def _resolve_spent(spent: Cost, metric: str | None) -> float:
    if metric == "tokens":
        return float(spent.tokens)
    if metric == "calls":
        return float(spent.calls)
    if metric == "time_sec":
        return float(spent.time_sec)
    return float(spent.usd)


def _normalize_cost(cost: Cost | Mapping[str, object] | None) -> Cost:
    if cost is None:
        return Cost()
    if isinstance(cost, Cost):
        return cost
    mapping = cast(Mapping[str, Any], cost)
    return Cost(
        usd=_coerce_float(mapping.get("usd")),
        tokens=_coerce_int(mapping.get("tokens")),
        calls=_coerce_int(mapping.get("calls")),
        time_sec=_coerce_float(mapping.get("time_sec")),
    )


def _coerce_float(value: object | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"expected numeric value, got {type(value)!r}")


def _coerce_int(value: object | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"expected integer value, got {type(value)!r}")
