"""Canonical budget models reused by manager and FlowRunner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .trace import TraceEventEmitter

__all__ = [
    "ScopeKey",
    "CostSnapshot",
    "BudgetSpec",
    "BudgetCharge",
    "BudgetChargeOutcome",
    "BudgetDecision",
]


@dataclass(frozen=True, slots=True)
class ScopeKey:
    """Identifies a budget scope (run/node/loop/etc.)."""

    scope_type: str
    scope_id: str


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Normalized cost snapshot in milliseconds and tokens."""

    time_ms: float = 0.0
    tokens: int = 0

    def __add__(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            time_ms=self.time_ms + other.time_ms,
            tokens=self.tokens + other.tokens,
        )

    def __sub__(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(
            time_ms=max(0.0, self.time_ms - other.time_ms),
            tokens=max(0, self.tokens - other.tokens),
        )

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls()

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> "CostSnapshot":
        if raw is None:
            return cls.zero()
        time_ms = float(raw.get("time_ms", 0.0))
        if "time_s" in raw:
            time_ms += float(raw["time_s"]) * 1000.0
        tokens = int(raw.get("tokens", 0))
        return cls(time_ms=time_ms, tokens=tokens)

    def has_positive(self) -> bool:
        return self.time_ms > 0.0 or self.tokens > 0

    def to_payload(self) -> dict[str, float | int]:
        return {"time_ms": float(self.time_ms), "tokens": int(self.tokens)}


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Declarative definition of a budget."""

    name: str
    scope_type: str
    limit: CostSnapshot
    mode: str = "hard"  # "hard" or "soft"
    breach_action: str = "stop"  # "stop" or "warn"

    def __post_init__(self) -> None:  # pragma: no cover - validation
        mode = self.mode.lower()
        action = self.breach_action.lower()
        if mode not in {"hard", "soft"}:
            raise ValueError("mode must be 'hard' or 'soft'")
        if action not in {"stop", "warn"}:
            raise ValueError("breach_action must be 'stop' or 'warn'")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "breach_action", action)

    def should_stop(self, breached: bool) -> bool:
        if not breached:
            return False
        if self.mode == "hard":
            return True
        return self.breach_action == "stop"


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    """Computed charge against a spec."""

    spec: BudgetSpec
    cost: CostSnapshot
    prior: CostSnapshot
    new_total: CostSnapshot
    remaining: CostSnapshot
    overage: CostSnapshot


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    """Outcome of applying a cost to a budget spec."""

    spec: BudgetSpec
    charge: BudgetCharge
    breached: bool

    @classmethod
    def compute(
        cls,
        *,
        spec: BudgetSpec,
        prior: CostSnapshot,
        cost: CostSnapshot,
    ) -> "BudgetChargeOutcome":
        new_total = prior + cost
        remaining_time = spec.limit.time_ms - new_total.time_ms
        remaining_tokens = spec.limit.tokens - new_total.tokens
        remaining = CostSnapshot(time_ms=remaining_time, tokens=remaining_tokens)
        overage = CostSnapshot(
            time_ms=max(0.0, -remaining_time),
            tokens=max(0, -remaining_tokens),
        )
        breached = overage.has_positive()
        charge = BudgetCharge(
            spec=spec,
            cost=cost,
            prior=prior,
            new_total=new_total,
            remaining=remaining,
            overage=overage,
        )
        return cls(spec=spec, charge=charge, breached=breached)

    @property
    def should_stop(self) -> bool:
        return self.spec.should_stop(self.breached)

    def to_trace_payload(self, *, scope_type: str, scope_id: str) -> Mapping[str, Any]:
        from types import MappingProxyType

        payload = {
            "scope_type": scope_type,
            "scope_id": scope_id,
            "spec_name": self.spec.name,
            "mode": self.spec.mode,
            "breach_action": self.spec.breach_action,
            "breached": self.breached,
            "should_stop": self.should_stop,
            "cost": self.charge.cost.to_payload(),
            "prior": self.charge.prior.to_payload(),
            "new_total": self.charge.new_total.to_payload(),
            "remaining": self.charge.remaining.to_payload(),
            "overage": self.charge.overage.to_payload(),
        }
        return MappingProxyType(payload)


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Aggregate of multiple budget outcomes for a scope."""

    scope: ScopeKey
    cost: CostSnapshot
    outcomes: tuple[BudgetChargeOutcome, ...]
    blocking: BudgetChargeOutcome | None

    @classmethod
    def make(
        cls,
        *,
        scope: ScopeKey,
        cost: CostSnapshot,
        outcomes: Iterable[BudgetChargeOutcome],
    ) -> "BudgetDecision":
        materialized = tuple(outcomes)
        blocking = next((out for out in materialized if out.should_stop), None)
        return cls(scope=scope, cost=cost, outcomes=materialized, blocking=blocking)

    @property
    def breached(self) -> bool:
        return any(outcome.breached for outcome in self.outcomes)

    @property
    def should_stop(self) -> bool:
        return self.blocking is not None

    def to_trace_records(
        self,
        *,
        emitter: TraceEventEmitter,
        event: str,
    ) -> None:
        for outcome in self.outcomes:
            emitter.emit(
                event,
                scope_type=self.scope.scope_type,
                scope_id=self.scope.scope_id,
                payload=outcome.to_trace_payload(
                    scope_type=self.scope.scope_type,
                    scope_id=self.scope.scope_id,
                ),
            )
