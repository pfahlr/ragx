from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import SupportsFloat

__all__ = [
    "BudgetError",
    "BudgetBreachError",
    "BudgetWarning",
    "BudgetChargeResult",
    "BudgetMeter",
]


_METRIC_KEYS = ("usd", "tokens", "calls", "time_ms")


def _coerce_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, SupportsFloat):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Budget value must be numeric, got {type(value)!r}")


class BudgetError(RuntimeError):
    """Base error for budget enforcement issues."""


class BudgetBreachError(BudgetError):
    """Raised when a hard budget cap would be exceeded."""

    def __init__(
        self,
        *,
        scope: str,
        metric: str,
        limit: float,
        attempted: float,
    ) -> None:
        super().__init__(
            f"{scope} budget hard cap exceeded for {metric}: "
            f"attempted {attempted:.4f} against limit {limit:.4f}"
        )
        self.scope = scope
        self.metric = metric
        self.limit = limit
        self.attempted = attempted


@dataclass(frozen=True, slots=True)
class BudgetWarning:
    """Details for a soft budget overage."""

    scope: str
    over: Mapping[str, float]
    mode: str


@dataclass(frozen=True, slots=True)
class BudgetChargeResult:
    """Outcome of a budget charge operation."""

    scope: str
    cost: Mapping[str, float]
    spent: Mapping[str, float]
    limits: Mapping[str, float | None]
    warning: BudgetWarning | None = None


class BudgetMeter:
    """Track spend against configured hard/soft budget caps."""

    def __init__(
        self,
        *,
        name: str,
        scope: str,
        config: Mapping[str, object] | None,
    ) -> None:
        self.name = name
        self.scope = scope
        normalized = dict(config or {})
        self.mode = str(normalized.get("mode", "hard" if scope != "spec" else "soft"))
        self.breach_action = normalized.get("breach_action")
        self._limits = self._extract_limits(normalized)
        self._spent = {key: 0.0 for key in _METRIC_KEYS}
        self._warnings: list[BudgetWarning] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_limits(config: Mapping[str, object]) -> dict[str, float | None]:
        limits: dict[str, float | None] = {key: None for key in _METRIC_KEYS}
        if "max_usd" in config:
            limits["usd"] = _coerce_float(config["max_usd"])
        if "max_tokens" in config:
            limits["tokens"] = _coerce_float(config["max_tokens"])
        if "max_calls" in config:
            limits["calls"] = _coerce_float(config["max_calls"])
        if "max_time_ms" in config:
            limits["time_ms"] = _coerce_float(config["max_time_ms"])
        if "time_limit_ms" in config:
            limits["time_ms"] = _coerce_float(config["time_limit_ms"])
        if "time_limit_sec" in config:
            limits["time_ms"] = _coerce_float(config["time_limit_sec"]) * 1000.0
        return limits

    @staticmethod
    def _normalize_cost(cost: Mapping[str, object]) -> dict[str, float]:
        normalized = {key: 0.0 for key in _METRIC_KEYS}
        for key in _METRIC_KEYS:
            value = cost.get(key)
            if value is None:
                continue
            normalized[key] = _coerce_float(value)
        # Accept shorthand keys for compatibility.
        if "time_limit_ms" in cost:
            normalized["time_ms"] = _coerce_float(cost["time_limit_ms"])
        if "time_limit_sec" in cost:
            normalized["time_ms"] = _coerce_float(cost["time_limit_sec"]) * 1000.0
        if "max_calls" in cost:
            normalized["calls"] = _coerce_float(cost["max_calls"])
        if "max_tokens" in cost:
            normalized["tokens"] = _coerce_float(cost["max_tokens"])
        if "max_usd" in cost:
            normalized["usd"] = _coerce_float(cost["max_usd"])
        return normalized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def can_spend(self, cost: Mapping[str, object]) -> bool:
        """Return ``True`` when the meter can accommodate the cost."""

        if self.mode == "soft":
            return True
        for metric, amount in self._normalize_cost(cost).items():
            limit = self._limits[metric]
            if limit is None:
                continue
            if self._spent[metric] + amount - limit > 1e-9:
                return False
        return True

    def charge(self, cost: Mapping[str, object]) -> BudgetChargeResult:
        """Charge the meter, raising on hard overages."""

        normalized = self._normalize_cost(cost)
        warning_over: dict[str, float] = {}
        for metric, amount in normalized.items():
            if amount == 0:
                continue
            limit = self._limits[metric]
            if limit is None:
                continue
            projected = self._spent[metric] + amount
            if projected - limit > 1e-9:
                if self.mode == "hard":
                    raise BudgetBreachError(
                        scope=self.full_scope,
                        metric=metric,
                        limit=limit,
                        attempted=projected,
                    )
                warning_over[metric] = projected - limit
        for metric, amount in normalized.items():
            self._spent[metric] += amount
        warning_obj: BudgetWarning | None = None
        if warning_over:
            warning_obj = BudgetWarning(
                scope=self.full_scope,
                over=MappingProxyType(dict(warning_over)),
                mode=self.mode,
            )
            self._warnings.append(warning_obj)
        return BudgetChargeResult(
            scope=self.full_scope,
            cost=MappingProxyType({k: v for k, v in normalized.items() if v}),
            spent=MappingProxyType(self._spent.copy()),
            limits=MappingProxyType(self._limits.copy()),
            warning=warning_obj,
        )

    @property
    def full_scope(self) -> str:
        return f"{self.scope}:{self.name}" if self.scope else self.name

    @property
    def is_exhausted(self) -> bool:
        for metric, limit in self._limits.items():
            if limit is None:
                continue
            if limit <= 0 and self._spent[metric] > 0:
                return True
            if limit > 0 and self._spent[metric] >= limit - 1e-9:
                return True
        return False

    def remaining(self) -> Mapping[str, float]:
        remaining: dict[str, float] = {}
        for metric, limit in self._limits.items():
            if limit is None:
                continue
            remaining[metric] = max(0.0, limit - self._spent[metric])
        return MappingProxyType(remaining)


    @property
    def limits(self) -> Mapping[str, float | None]:
        return MappingProxyType(self._limits.copy())

    @property
    def spent(self) -> Mapping[str, float]:
        return MappingProxyType(self._spent.copy())
    @property
    def warnings(self) -> tuple[BudgetWarning, ...]:
        return tuple(self._warnings)
