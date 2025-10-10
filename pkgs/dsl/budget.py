"""Budget accounting utilities for the FlowRunner."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, replace

__all__ = [
    "Cost",
    "BudgetDecision",
    "BudgetExceededError",
    "BudgetMeter",
]


_EPSILON = 1e-9


def _coerce_optional_number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


@dataclass(frozen=True, slots=True)
class Cost:
    """Normalized cost payload tracked across budgets."""

    usd: float = 0.0
    tokens: float = 0.0
    calls: float = 0.0
    time_ms: float = 0.0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            usd=self.usd + other.usd,
            tokens=self.tokens + other.tokens,
            calls=self.calls + other.calls,
            time_ms=self.time_ms + other.time_ms,
        )

    def __sub__(self, other: Cost) -> Cost:
        return Cost(
            usd=self.usd - other.usd,
            tokens=self.tokens - other.tokens,
            calls=self.calls - other.calls,
            time_ms=self.time_ms - other.time_ms,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "usd": float(self.usd),
            "tokens": float(self.tokens),
            "calls": float(self.calls),
            "time_ms": float(self.time_ms),
        }


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Result of applying or previewing a budget charge."""

    scope: str
    allowed: bool
    breached: tuple[str, ...]
    soft_breach: bool
    remaining: Cost
    spent: Cost

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "allowed": self.allowed,
            "breached": self.breached,
            "soft_breach": self.soft_breach,
            "remaining": self.remaining.as_dict(),
            "spent": self.spent.as_dict(),
        }


class BudgetExceededError(RuntimeError):
    """Raised when a hard budget refuses a charge."""

    def __init__(self, decision: BudgetDecision) -> None:
        message = "Budget exceeded"
        if decision.breached:
            message = (
                f"Budget exceeded for {decision.scope}: "
                + ", ".join(decision.breached)
            )
        super().__init__(message)
        self.decision = decision


class BudgetMeter:
    """Track spend and enforce hard/soft budgets for a scope."""

    def __init__(
        self,
        *,
        scope: str,
        limits: Mapping[str, float | int | None] | None = None,
        mode: str | None = None,
    ) -> None:
        self._scope = scope
        self._mode = (mode or "hard").lower()
        if self._mode not in {"hard", "soft"}:
            raise ValueError(f"Unsupported budget mode: {mode!r}")
        normalized = self._normalize_limits(limits or {})
        self._limits: MutableMapping[str, float | None] = normalized
        self._spent = Cost()
        self._last_decision: BudgetDecision | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_budget(
        cls, budget: Mapping[str, object] | None, *, scope: str
    ) -> BudgetMeter:
        raw = dict(budget) if isinstance(budget, Mapping) else {}
        mode_value = raw.get("mode")
        mode = mode_value if isinstance(mode_value, str) else None
        limits = {
            "max_usd": _coerce_optional_number(raw.get("max_usd")),
            "max_tokens": _coerce_optional_number(raw.get("max_tokens")),
            "max_calls": _coerce_optional_number(raw.get("max_calls")),
            "time_limit_sec": _coerce_optional_number(raw.get("time_limit_sec")),
            "max_time_ms": _coerce_optional_number(raw.get("max_time_ms")),
        }
        return cls(scope=scope, limits=limits, mode=mode)

    @staticmethod
    def _normalize_limits(
        limits: Mapping[str, float | int | None]
    ) -> MutableMapping[str, float | None]:
        def _norm_number(value: float | int | None) -> float | None:
            if value is None:
                return None
            number = float(value)
            if number <= 0:
                return None
            return number

        normalized: MutableMapping[str, float | None] = {
            "usd": _norm_number(limits.get("max_usd")),
            "tokens": _norm_number(limits.get("max_tokens")),
            "calls": _norm_number(limits.get("max_calls")),
            "time_ms": None,
        }
        if "time_limit_sec" in limits and limits["time_limit_sec"] is not None:
            seconds = float(limits["time_limit_sec"])
            normalized["time_ms"] = seconds * 1000.0 if seconds > 0 else None
        elif "max_time_ms" in limits and limits["max_time_ms"] is not None:
            millis = float(limits["max_time_ms"])
            normalized["time_ms"] = millis if millis > 0 else None
        return normalized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def scope(self) -> str:
        return self._scope

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def spent(self) -> Cost:
        return self._spent

    @property
    def remaining(self) -> Cost:
        return Cost(
            usd=self._remaining_for("usd"),
            tokens=self._remaining_for("tokens"),
            calls=self._remaining_for("calls"),
            time_ms=self._remaining_for("time_ms"),
        )

    @property
    def last_decision(self) -> BudgetDecision | None:
        return self._last_decision

    def can_spend(self, cost: Cost) -> bool:
        decision = self._evaluate(cost)
        self._last_decision = decision
        return decision.allowed

    def charge(self, cost: Cost) -> BudgetDecision:
        decision = self._evaluate(cost)
        if not decision.allowed and self._mode == "hard":
            self._last_decision = decision
            raise BudgetExceededError(decision)
        self._spent = self._spent + cost
        applied = replace(decision, spent=self._spent)
        self._last_decision = applied
        return applied

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _remaining_for(self, key: str) -> float:
        limit = self._limits[key]
        if limit is None:
            return math.inf
        value = getattr(self._spent, key)
        remaining = limit - value
        # Ensure we do not surface negative zeros.
        if abs(remaining) < _EPSILON:
            return 0.0
        return remaining

    def _evaluate(self, cost: Cost) -> BudgetDecision:
        breaches: list[str] = []
        future_spent = self._spent + cost
        remaining_values: dict[str, float] = {}

        for key, limit in self._limits.items():
            future_value = getattr(future_spent, key)
            if limit is None:
                remaining_values[key] = math.inf
                continue
            remaining = limit - future_value
            if remaining < 0 and abs(remaining) < _EPSILON:
                remaining = 0.0
            remaining_values[key] = remaining
            if future_value - limit > _EPSILON:
                breaches.append(key)

        allowed = not breaches or self._mode == "soft"
        decision = BudgetDecision(
            scope=self._scope,
            allowed=allowed,
            breached=tuple(breaches),
            soft_breach=bool(breaches) and self._mode == "soft",
            remaining=Cost(**remaining_values),
            spent=future_spent,
        )
        return decision
