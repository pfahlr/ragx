"""Budget tracking utilities for the DSL runner."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType

Number = float | int
CostMapping = Mapping[str, Number]

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class BudgetBreach:
    """Metadata describing a budget breach event."""

    scope: str
    metric: str
    level: str
    limit: Number | None
    attempted: Number
    spent_before: Number

    @property
    def remaining_before(self) -> Number | None:
        if self.limit is None:
            return None
        remaining = self.limit - self.spent_before
        return remaining if remaining > 0 else 0


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    """Result of a successful budget charge."""

    scope: str
    cost: Mapping[str, Number]
    spent: Mapping[str, Number]
    breaches: tuple[BudgetBreach, ...] = ()


@dataclass(frozen=True, slots=True)
class BudgetCheck:
    """Outcome of a preflight budget check."""

    allowed: bool
    breach: BudgetBreach | None = None


class BudgetExceededError(RuntimeError):
    """Raised when attempting to charge past a hard budget limit."""

    def __init__(
        self,
        *,
        scope: str,
        metric: str,
        limit: Number | None,
        attempted: Number,
    ) -> None:
        message = (
            f"Budget exceeded for {scope} ({metric}): attempted {attempted} "
            f"with limit {limit}"
        )
        super().__init__(message)
        self.scope = scope
        self.metric = metric
        self.limit = limit
        self.attempted = attempted


class BudgetMeter:
    """Track spend across currencies/tokens/time with hard or soft limits."""

    def __init__(
        self,
        *,
        scope: str,
        config: Mapping[str, object] | None = None,
        default_mode: str = "hard",
    ) -> None:
        self.scope = scope
        normalized = dict(config or {})
        self.mode = str(normalized.get("mode", default_mode)).lower()
        if self.mode not in {"hard", "soft"}:
            raise ValueError(f"Unsupported budget mode: {self.mode!r}")

        self.breach_action = normalized.get("breach_action")

        self._limits: dict[str, Number | None] = {
            "usd": _normalize_limit(normalized.get("max_usd")),
            "tokens": _normalize_limit(normalized.get("max_tokens")),
            "calls": _normalize_limit(normalized.get("max_calls")),
            "time_ms": _normalize_time_limit(normalized.get("time_limit_sec")),
        }
        self._spent: dict[str, Number] = {key: 0 for key in self._limits}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def can_spend(self, cost: CostMapping | None) -> BudgetCheck:
        """Return whether the provided cost can be applied without breach."""

        if not cost:
            return BudgetCheck(allowed=True, breach=None)

        normalized_cost = _normalize_cost(cost)
        soft_breach: BudgetBreach | None = None
        for metric, limit in self._limits.items():
            if limit is None:
                continue
            projected = self._spent[metric] + normalized_cost[metric]
            if projected - limit > _EPSILON:
                breach = BudgetBreach(
                    scope=self.scope,
                    metric=metric,
                    level=self.mode,
                    limit=limit,
                    attempted=projected,
                    spent_before=self._spent[metric],
                )
                if self.mode == "hard":
                    return BudgetCheck(allowed=False, breach=breach)
                soft_breach = soft_breach or breach
        return BudgetCheck(allowed=True, breach=soft_breach)

    def charge(self, cost: CostMapping | None) -> BudgetCharge:
        """Apply a cost to the meter, raising if hard budgets are exceeded."""

        if not cost:
            return BudgetCharge(
                scope=self.scope,
                cost=MappingProxyType({}),
                spent=self.spent_snapshot(),
                breaches=(),
            )

        normalized_cost = _normalize_cost(cost)
        breaches: list[BudgetBreach] = []
        for metric, limit in self._limits.items():
            if limit is None:
                continue
            projected = self._spent[metric] + normalized_cost[metric]
            if projected - limit > _EPSILON:
                breach = BudgetBreach(
                    scope=self.scope,
                    metric=metric,
                    level=self.mode,
                    limit=limit,
                    attempted=projected,
                    spent_before=self._spent[metric],
                )
                if self.mode == "hard":
                    raise BudgetExceededError(
                        scope=self.scope,
                        metric=metric,
                        limit=limit,
                        attempted=projected,
                    )
                breaches.append(breach)

        for metric, value in normalized_cost.items():
            self._spent[metric] += value

        return BudgetCharge(
            scope=self.scope,
            cost=_freeze(normalized_cost),
            spent=self.spent_snapshot(),
            breaches=tuple(breaches),
        )

    def remaining(self) -> Mapping[str, Number | None]:
        """Return remaining budget for each metric (None if unlimited)."""

        remaining: dict[str, Number | None] = {}
        for metric, limit in self._limits.items():
            if limit is None:
                remaining[metric] = None
            else:
                remaining[metric] = max(limit - self._spent[metric], 0)
        return MappingProxyType(remaining)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def spent_snapshot(self) -> Mapping[str, Number]:
        return MappingProxyType(dict(self._spent))


def _normalize_limit(raw: object | None) -> Number | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        value = float(raw)
    else:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(f"Budget limit must be numeric or None, got {raw!r}") from None
    if value <= 0:
        return None
    return value


def _normalize_time_limit(raw: object | None) -> Number | None:
    limit = _normalize_limit(raw)
    if limit is None:
        return None
    return limit * 1000.0


def _normalize_cost(cost: CostMapping) -> dict[str, Number]:
    normalized: dict[str, Number] = {"usd": 0.0, "tokens": 0.0, "calls": 0.0, "time_ms": 0.0}
    for key, value in cost.items():
        if value is None:
            continue
        if key not in normalized:
            raise ValueError(f"Unsupported cost metric: {key!r}")
        normalized[key] = float(value)
    return normalized


def _freeze(cost: MutableMapping[str, Number] | Mapping[str, Number]) -> Mapping[str, Number]:
    return MappingProxyType(dict(cost))

