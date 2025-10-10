"""Budget domain models and helpers for the FlowRunner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

__all__ = [
    "BudgetBreachError",
    "BudgetDecision",
    "BudgetMeter",
    "BudgetMode",
    "BudgetSpec",
    "CostSnapshot",
]


class BudgetMode(Enum):
    """Budget enforcement mode."""

    HARD = "hard"
    SOFT = "soft"

    @classmethod
    def from_value(cls, value: object | None) -> "BudgetMode":
        if isinstance(value, BudgetMode):
            return value
        if isinstance(value, str):
            normalized = value.lower().strip()
            if normalized == "soft":
                return cls.SOFT
        return cls.HARD


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable cost bookkeeping snapshot."""

    usd: float = 0.0
    calls: int = 0
    tokens: int = 0
    elapsed_ms: int = 0

    def __add__(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            usd=self.usd + other.usd,
            calls=self.calls + other.calls,
            tokens=self.tokens + other.tokens,
            elapsed_ms=self.elapsed_ms + other.elapsed_ms,
        )

    def __sub__(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            usd=max(self.usd - other.usd, 0.0),
            calls=max(self.calls - other.calls, 0),
            tokens=max(self.tokens - other.tokens, 0),
            elapsed_ms=max(self.elapsed_ms - other.elapsed_ms, 0),
        )

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls()


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Declarative description of a budget."""

    max_usd: float | None = None
    max_calls: int | None = None
    max_tokens: int | None = None
    time_limit_ms: int | None = None
    mode: BudgetMode = BudgetMode.HARD
    breach_action: str = "error"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "BudgetSpec":
        if data is None:
            return cls()
        mapping = dict(data)
        mode = BudgetMode.from_value(mapping.get("mode"))
        breach_action = str(mapping.get("breach_action", "warn" if mode is BudgetMode.SOFT else "error"))
        max_usd = cls._coerce_float(mapping.get("max_usd"))
        max_calls = cls._coerce_int(mapping.get("max_calls"))
        max_tokens = cls._coerce_int(mapping.get("max_tokens"))
        time_limit_sec = cls._coerce_float(mapping.get("time_limit_sec"))
        time_limit_ms = cls._coerce_int(mapping.get("time_limit_ms"))
        if time_limit_ms is None and time_limit_sec is not None:
            time_limit_ms = int(round(time_limit_sec * 1000))
        return cls(
            max_usd=max_usd,
            max_calls=max_calls,
            max_tokens=max_tokens,
            time_limit_ms=time_limit_ms,
            mode=mode,
            breach_action=breach_action,
        )

    @staticmethod
    def _coerce_float(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid float value {value!r}") from exc
        return result

    @staticmethod
    def _coerce_int(value: object | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid int value {value!r}") from exc

    def has_limits(self) -> bool:
        return any(
            limit is not None
            for limit in (self.max_usd, self.max_calls, self.max_tokens, self.time_limit_ms)
        )


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Outcome of a budget charge for a scope."""

    scope_type: str
    scope_id: str
    spec: BudgetSpec
    cost: CostSnapshot
    spent: CostSnapshot
    remaining: CostSnapshot
    overage: CostSnapshot
    stage: str
    breached: bool
    mode: BudgetMode
    breach_action: str
    should_stop: bool
    warnings: tuple[str, ...]


class BudgetBreachError(RuntimeError):
    """Raised when a hard budget is exceeded."""

    def __init__(self, decision: BudgetDecision) -> None:
        action = decision.breach_action or "error"
        message = (
            f"Budget breach on {decision.scope_type}:{decision.scope_id} action={action}"
        )
        super().__init__(message)
        self.decision = decision


class BudgetMeter:
    """Track budget consumption for a single scope."""

    def __init__(self, *, scope_type: str, scope_id: str, spec: BudgetSpec) -> None:
        self._scope_type = scope_type
        self._scope_id = scope_id
        self._spec = spec
        self._spent = CostSnapshot.zero()

    @property
    def spec(self) -> BudgetSpec:
        return self._spec

    @property
    def spent(self) -> CostSnapshot:
        return self._spent

    def preview(self, cost: CostSnapshot) -> BudgetDecision:
        return self._evaluate(cost, stage="preflight", mutate=False)

    def commit(self, cost: CostSnapshot) -> BudgetDecision:
        return self._evaluate(cost, stage="commit", mutate=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate(self, cost: CostSnapshot, *, stage: str, mutate: bool) -> BudgetDecision:
        projected = self._spent + cost
        spec = self._spec
        remaining = self._compute_remaining(spec, projected)
        overage = self._compute_overage(spec, projected)
        breached = any(
            value > 0
            for value in (overage.usd, overage.calls, overage.tokens, overage.elapsed_ms)
        )
        exhausted = self._is_exhausted(spec, projected)
        should_stop = spec.breach_action == "stop" and (exhausted or breached)
        warnings: tuple[str, ...] = ()
        if breached and spec.mode is BudgetMode.SOFT:
            warnings = ("budget_soft_limit_exceeded",)

        decision = BudgetDecision(
            scope_type=self._scope_type,
            scope_id=self._scope_id,
            spec=spec,
            cost=cost,
            spent=projected,
            remaining=remaining,
            overage=overage,
            stage=stage,
            breached=breached,
            mode=spec.mode,
            breach_action=spec.breach_action,
            should_stop=should_stop,
            warnings=warnings,
        )

        if mutate:
            if breached and spec.mode is BudgetMode.HARD and spec.breach_action != "stop":
                raise BudgetBreachError(decision)
            self._spent = projected
        return decision

    @staticmethod
    def _compute_remaining(spec: BudgetSpec, spent: CostSnapshot) -> CostSnapshot:
        return CostSnapshot(
            usd=max((spec.max_usd or float("inf")) - spent.usd, 0.0)
            if spec.max_usd is not None
            else float("inf"),
            calls=max((spec.max_calls or float("inf")) - spent.calls, 0)
            if spec.max_calls is not None
            else float("inf"),
            tokens=max((spec.max_tokens or float("inf")) - spent.tokens, 0)
            if spec.max_tokens is not None
            else float("inf"),
            elapsed_ms=max((spec.time_limit_ms or float("inf")) - spent.elapsed_ms, 0)
            if spec.time_limit_ms is not None
            else float("inf"),
        )

    @staticmethod
    def _compute_overage(spec: BudgetSpec, spent: CostSnapshot) -> CostSnapshot:
        usd_over = max(spent.usd - spec.max_usd, 0.0) if spec.max_usd is not None else 0.0
        call_over = max(spent.calls - spec.max_calls, 0) if spec.max_calls is not None else 0
        token_over = max(spent.tokens - spec.max_tokens, 0) if spec.max_tokens is not None else 0
        time_over = (
            max(spent.elapsed_ms - spec.time_limit_ms, 0)
            if spec.time_limit_ms is not None
            else 0
        )
        return CostSnapshot(
            usd=usd_over,
            calls=call_over,
            tokens=token_over,
            elapsed_ms=time_over,
        )

    @staticmethod
    def _is_exhausted(spec: BudgetSpec, spent: CostSnapshot) -> bool:
        return any(
            (
                spec.max_usd is not None and spent.usd >= spec.max_usd,
                spec.max_calls is not None and spent.calls >= spec.max_calls,
                spec.max_tokens is not None and spent.tokens >= spec.max_tokens,
                spec.time_limit_ms is not None and spent.elapsed_ms >= spec.time_limit_ms,
            )
        )
