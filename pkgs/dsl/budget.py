"""Budget metering primitives for the DSL runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "BudgetError",
    "BudgetBreachHard",
    "BudgetSpec",
    "BudgetRemaining",
    "BudgetChargeOutcome",
    "CostSnapshot",
    "BudgetMeter",
]


class BudgetError(RuntimeError):
    """Base error for budget evaluation failures."""


class BudgetBreachHard(BudgetError):
    """Raised when a hard budget cap is exceeded."""

    def __init__(self, scope: str, *, limit: str, amount: float | int) -> None:
        super().__init__(f"Budget hard cap exceeded for {scope}: {limit}={amount}")
        self.scope = scope
        self.limit = limit
        self.amount = amount


@dataclass(slots=True, frozen=True)
class BudgetSpec:
    """Canonical representation of a budget configuration."""

    mode: str | None = None
    scope: str | None = None
    max_usd: float | None = None
    max_tokens: int | None = None
    max_calls: int | None = None
    time_limit_sec: float | None = None
    token_rate: Mapping[str, object] | None = None
    breach_action: str | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        scope: str | None = None,
    ) -> BudgetSpec:
        if not data:
            return cls(scope=scope)
        raw_mode = data.get("mode")
        mode = raw_mode if isinstance(raw_mode, str) else None
        raw_breach = data.get("breach_action")
        breach_action = raw_breach if isinstance(raw_breach, str) else None
        raw_token_rate = data.get("token_rate")
        def _coerce_number(value: object) -> float | None:
            if value is None:
                return None
            if isinstance(value, int | float):
                return float(value)
            return None

        def _coerce_int(value: object) -> int | None:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return None

        return cls(
            mode=mode,
            scope=scope,
            max_usd=_coerce_number(data.get("max_usd")),
            max_tokens=_coerce_int(data.get("max_tokens")),
            max_calls=_coerce_int(data.get("max_calls")),
            time_limit_sec=_coerce_number(data.get("time_limit_sec")),
            token_rate=raw_token_rate if isinstance(raw_token_rate, Mapping) else None,
            breach_action=breach_action,
        )


@dataclass(slots=True, frozen=True)
class CostSnapshot:
    """Concrete spend payload for budget accounting."""

    usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass(slots=True, frozen=True)
class BudgetRemaining:
    """Remaining headroom for each tracked metric."""

    usd: float | None
    calls: int | None
    tokens: int | None
    seconds: float | None


@dataclass(slots=True, frozen=True)
class BudgetChargeOutcome:
    """Result metadata returned from :meth:`BudgetMeter.charge`."""

    soft_breach: bool
    breach_kind: str | None
    remaining: BudgetRemaining


class BudgetMeter:
    """Accumulates spend against configured budget caps."""

    def __init__(
        self,
        spec: BudgetSpec,
        *,
        scope: str,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._spec = spec
        self._scope = scope
        self._clock = clock or (lambda: 0.0)
        self._usd_spent = 0.0
        self._tokens_spent = 0
        self._calls_spent = 0
        self._seconds_spent = 0.0
        self._soft_exceeded = False
        self._hard_exceeded = False
        self._defer_hard_stop = (
            (spec.scope == "loop" or scope == "loop")
            and spec.breach_action == "stop"
        )

    @classmethod
    def from_spec(
        cls,
        config: BudgetSpec | Mapping[str, object] | None,
        *,
        scope: str,
        clock: Callable[[], float] | None = None,
    ) -> BudgetMeter:
        if isinstance(config, BudgetSpec):
            spec = config
        else:
            spec = BudgetSpec.from_mapping(config, scope=scope)
        return cls(spec, scope=scope, clock=clock)

    @classmethod
    def unlimited(cls, *, scope: str) -> BudgetMeter:
        return cls(BudgetSpec(scope=scope), scope=scope)

    @property
    def mode(self) -> str:
        if self._spec.mode:
            return self._spec.mode
        if self._spec.breach_action:
            return "hard"
        return "hard"

    @property
    def stop_behavior(self) -> str:
        return self._spec.breach_action or "error"

    @property
    def exceeded(self) -> bool:
        return self._soft_exceeded or self._hard_exceeded

    def can_spend(self, cost: CostSnapshot) -> bool:
        totals = self._project_totals(cost)
        for _name, limit, total in totals:
            if limit is None or limit <= 0:
                continue
            if total > limit + 1e-9:
                if self.mode == "hard" and not self._defer_hard_stop:
                    return False
        return True

    def charge(self, cost: CostSnapshot) -> BudgetChargeOutcome:
        breach_kind: str | None = None
        breach_limit: str | None = None
        totals = self._project_totals(cost)
        for name, limit, total in totals:
            if limit is None or limit <= 0:
                continue
            if total > limit + 1e-9:
                breach_kind = "hard" if self.mode == "hard" else "soft"
                breach_limit = name
                break
        self._usd_spent += cost.usd
        self._calls_spent += cost.calls
        self._tokens_spent += cost.tokens_total
        self._seconds_spent += cost.seconds
        for _name, limit, total in totals:
            if limit is None or limit <= 0:
                continue
            if total >= limit - 1e-9:
                if self.mode == "hard":
                    self._hard_exceeded = True
                else:
                    self._soft_exceeded = True
        if breach_kind == "hard":
            if not self._defer_hard_stop:
                raise BudgetBreachHard(
                    self._scope,
                    limit=breach_limit or "budget",
                    amount=self._current_amount(breach_limit or "budget", cost),
                )
        elif breach_kind == "soft":
            self._soft_exceeded = True
        return BudgetChargeOutcome(
            soft_breach=breach_kind == "soft",
            breach_kind=breach_kind,
            remaining=self.remaining(),
        )

    def remaining(self) -> BudgetRemaining:
        return BudgetRemaining(
            usd=self._remaining_float(self._spec.max_usd, self._usd_spent),
            calls=self._remaining_int(self._spec.max_calls, self._calls_spent),
            tokens=self._remaining_int(self._spec.max_tokens, self._tokens_spent),
            seconds=self._remaining_float(self._spec.time_limit_sec, self._seconds_spent),
        )

    def _project_totals(
        self, cost: CostSnapshot
    ) -> tuple[tuple[str, float | int | None, float | int], ...]:
        return (
            ("usd", self._spec.max_usd, self._usd_spent + cost.usd),
            ("calls", self._spec.max_calls, self._calls_spent + cost.calls),
            ("tokens", self._spec.max_tokens, self._tokens_spent + cost.tokens_total),
            ("seconds", self._spec.time_limit_sec, self._seconds_spent + cost.seconds),
        )

    def _current_amount(self, limit: str, cost: CostSnapshot) -> float | int:
        if limit == "usd":
            return self._usd_spent + cost.usd
        if limit == "calls":
            return self._calls_spent + cost.calls
        if limit == "tokens":
            return self._tokens_spent + cost.tokens_total
        if limit == "seconds":
            return self._seconds_spent + cost.seconds
        return 0

    @staticmethod
    def _remaining_float(limit: float | None, spent: float) -> float | None:
        if limit is None or limit <= 0:
            return None
        return max(limit - spent, 0.0)

    @staticmethod
    def _remaining_int(limit: int | None, spent: int) -> int | None:
        if limit is None or limit <= 0:
            return None
        remaining = limit - spent
        return remaining if remaining >= 0 else 0

    def snapshot(self) -> Mapping[str, float | int | None]:
        return MappingProxyType(
            {
                "usd_spent": self._usd_spent,
                "calls_spent": self._calls_spent,
                "tokens_spent": self._tokens_spent,
                "seconds_spent": self._seconds_spent,
                "remaining_usd": self._remaining_float(self._spec.max_usd, self._usd_spent),
                "remaining_calls": self._remaining_int(self._spec.max_calls, self._calls_spent),
                "remaining_tokens": self._remaining_int(
                    self._spec.max_tokens, self._tokens_spent
                ),
                "remaining_seconds": self._remaining_float(
                    self._spec.time_limit_sec, self._seconds_spent
                ),
            }
        )
