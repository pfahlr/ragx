from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


__all__ = [
    "CostSnapshot",
    "BudgetMode",
    "BudgetSpec",
    "BudgetBreach",
    "BudgetChargeOutcome",
    "BudgetChargeResult",
    "BudgetManager",
]


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable view over cost metrics used by budgeting logic."""

    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        normalized = {key: float(value) for key, value in self.metrics.items()}
        object.__setattr__(self, "metrics", MappingProxyType(normalized))

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(_combine_metrics(self.metrics, other.metrics, op="add"))

    def subtract(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot(_combine_metrics(self.metrics, other.metrics, op="subtract"))

    def clamp_nonnegative(self) -> "CostSnapshot":
        return CostSnapshot({key: max(value, 0.0) for key, value in self.metrics.items()})

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls({})


class BudgetMode(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    scope_type: str
    scope_id: str
    limit: CostSnapshot
    mode: BudgetMode
    breach_action: str = "stop"


@dataclass(frozen=True, slots=True)
class BudgetBreach:
    scope_type: str
    scope_id: str
    kind: str
    action: str
    overages: CostSnapshot
    stop_reason: str


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    scope_type: str
    scope_id: str
    cost: CostSnapshot
    spent: CostSnapshot
    remaining: CostSnapshot
    overages: CostSnapshot
    spec: BudgetSpec | None
    breach: BudgetBreach | None


@dataclass(frozen=True, slots=True)
class BudgetChargeResult:
    outcomes: tuple[BudgetChargeOutcome, ...]
    breached: BudgetBreach | None

    @property
    def should_stop(self) -> bool:
        return self.breached is not None and self.breached.action == "stop"

    def outcome_for(self, scope_type: str, scope_id: str) -> BudgetChargeOutcome:
        for outcome in self.outcomes:
            if outcome.scope_type == scope_type and outcome.scope_id == scope_id:
                return outcome
        raise KeyError(f"no outcome for {scope_type}:{scope_id}")


class BudgetManager:
    """Manage nested budget scopes and compute charge outcomes."""

    def __init__(self) -> None:
        self._stack: list[_ScopeContext] = []

    def open_scope(self, spec: BudgetSpec | None):
        context = _ScopeContext(self, spec)
        return context

    def _register(self, context: "_ScopeContext") -> None:
        parent = self._stack[-1] if self._stack else None
        context.attach(parent)
        self._stack.append(context)

    def _unregister(self, context: "_ScopeContext") -> None:
        if not self._stack or self._stack[-1] is not context:
            raise RuntimeError("budget scopes must unwind in LIFO order")
        self._stack.pop()

    def _charge(self, context: "_ScopeContext", cost: CostSnapshot) -> BudgetChargeResult:
        outcomes: list[BudgetChargeOutcome] = []
        breach_to_report: BudgetBreach | None = None
        for scope in context.iter_scopes():
            outcome = scope.apply_cost(cost)
            outcomes.append(outcome)
            if outcome.breach is not None and breach_to_report is None:
                breach_to_report = outcome.breach
        return BudgetChargeResult(tuple(outcomes), breach_to_report)


@dataclass(slots=True)
class _BudgetState:
    spec: BudgetSpec | None
    spent: dict[str, float]

    def snapshot(self) -> CostSnapshot:
        return CostSnapshot(self.spent)


class _ScopeContext:
    def __init__(self, manager: BudgetManager, spec: BudgetSpec | None) -> None:
        self._manager = manager
        self.spec = spec
        self._parent: _ScopeContext | None = None
        self._state = _BudgetState(spec=spec, spent={})

    def attach(self, parent: "_ScopeContext" | None) -> None:
        self._parent = parent

    def __enter__(self) -> "_ScopeContext":
        self._manager._register(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple teardown
        self._manager._unregister(self)

    def charge(self, cost: CostSnapshot) -> BudgetChargeResult:
        return self._manager._charge(self, cost)

    def iter_scopes(self) -> Iterator["_ScopeContext"]:
        current: _ScopeContext | None = self
        while current is not None:
            yield current
            current = current._parent

    def apply_cost(self, cost: CostSnapshot) -> BudgetChargeOutcome:
        for key, value in cost.metrics.items():
            self._state.spent[key] = self._state.spent.get(key, 0.0) + value
        spent_snapshot = self._state.snapshot()
        spec = self.spec
        if spec is None:
            remaining = CostSnapshot.zero()
            overages = CostSnapshot.zero()
            breach = None
            scope_type = "unbounded"
            scope_id = "unbounded"
        else:
            remaining_map = _combine_metrics(spec.limit.metrics, spent_snapshot.metrics, op="remaining")
            remaining = CostSnapshot(remaining_map)
            over_map = _combine_metrics(spent_snapshot.metrics, spec.limit.metrics, op="subtract")
            overages = CostSnapshot({k: max(v, 0.0) for k, v in over_map.items()})
            breach = None
            if any(value > 0 for value in overages.metrics.values()):
                kind = "hard" if spec.mode is BudgetMode.HARD else "soft"
                action = spec.breach_action
                reason = f"{spec.scope_type} budget exceeded"
                breach = BudgetBreach(
                    scope_type=spec.scope_type,
                    scope_id=spec.scope_id,
                    kind=kind,
                    action=action,
                    overages=overages,
                    stop_reason=reason,
                )
                if spec.mode is BudgetMode.SOFT and action == "stop":
                    breach = BudgetBreach(
                        scope_type=spec.scope_type,
                        scope_id=spec.scope_id,
                        kind="soft",
                        action=action,
                        overages=overages,
                        stop_reason=reason,
                    )
            scope_type = spec.scope_type
            scope_id = spec.scope_id
        return BudgetChargeOutcome(
            scope_type=scope_type,
            scope_id=scope_id,
            cost=cost,
            spent=spent_snapshot,
            remaining=remaining,
            overages=overages,
            spec=spec,
            breach=breach,
        )


def _combine_metrics(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    op: str,
) -> dict[str, float]:
    keys = set(left) | set(right)
    combined: dict[str, float] = {}
    for key in keys:
        l_val = float(left.get(key, 0.0))
        r_val = float(right.get(key, 0.0))
        if op == "add":
            combined[key] = l_val + r_val
        elif op == "subtract":
            combined[key] = l_val - r_val
        elif op == "remaining":
            combined[key] = max(l_val - r_val, 0.0)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown operation {op}")
    return combined
