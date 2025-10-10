from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Dict

from .trace import TraceEventEmitter

__all__ = [
    "BreachAction",
    "BudgetChargeOutcome",
    "BudgetChargeResult",
    "BudgetContext",
    "BudgetManager",
    "BudgetMode",
    "BudgetSpec",
    "CostSnapshot",
    "ScopeKey",
    "ScopeType",
]


class ScopeType(str, Enum):
    RUN = "run"
    LOOP = "loop"
    NODE = "node"
    SPEC = "spec"


@dataclass(frozen=True, slots=True)
class ScopeKey:
    scope_type: ScopeType
    identifier: str


class BudgetMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class BreachAction(str, Enum):
    STOP = "stop"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: Dict[str, float] = {}
        for key, value in dict(self.metrics).items():
            normalized[key] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(normalized))

    @classmethod
    def from_raw(cls, metrics: Mapping[str, float] | None = None) -> "CostSnapshot":
        normalized: Dict[str, float] = {}
        for key, value in dict(metrics or {}).items():
            if key.endswith("_s"):
                normalized[f"{key[:-2]}_ms"] = float(value) * 1000.0
            else:
                normalized[key] = float(value)
        return cls(normalized)

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls({})

    def add(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot({key: self.metrics.get(key, 0.0) + other.metrics.get(key, 0.0) for key in self._keys(other)})

    def subtract(self, other: "CostSnapshot") -> "CostSnapshot":
        return CostSnapshot({key: self.metrics.get(key, 0.0) - other.metrics.get(key, 0.0) for key in self._keys(other)})

    def clamp_non_negative(self) -> "CostSnapshot":
        return CostSnapshot({key: max(value, 0.0) for key, value in self.metrics.items()})

    def any_positive(self) -> bool:
        return any(value > 0.0 for value in self.metrics.values())

    def as_dict(self) -> dict[str, float]:
        return dict(self.metrics)

    def _keys(self, other: "CostSnapshot") -> set[str]:
        keys = set(self.metrics.keys())
        keys.update(other.metrics.keys())
        return keys


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    name: str
    limits: CostSnapshot
    mode: BudgetMode = BudgetMode.HARD
    breach_action: BreachAction = BreachAction.STOP


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    scope: ScopeKey
    spec: BudgetSpec
    spent: CostSnapshot
    remaining: CostSnapshot
    overages: CostSnapshot
    breached: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BudgetChargeResult:
    outcomes: Mapping[ScopeKey, BudgetChargeOutcome]
    should_stop: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BudgetContext:
    run: ScopeKey | None = None
    loop: ScopeKey | None = None
    node: ScopeKey | None = None
    spec: ScopeKey | None = None

    def scopes(self) -> tuple[ScopeKey, ...]:
        ordered = []
        for scope in (self.run, self.loop, self.node, self.spec):
            if scope is not None and scope not in ordered:
                ordered.append(scope)
        return tuple(ordered)


@dataclass(slots=True)
class _ScopeState:
    spec: BudgetSpec | None
    spent: CostSnapshot
    parent: ScopeKey | None


class BudgetManager:
    """Manage budget scopes and emit deterministic trace events."""

    def __init__(self, *, trace_emitter: TraceEventEmitter | None = None) -> None:
        self._states: dict[ScopeKey, _ScopeState] = {}
        self._trace = trace_emitter

    def enter_scope(
        self,
        scope: ScopeKey,
        spec: BudgetSpec | None,
        *,
        parent: ScopeKey | None = None,
    ) -> None:
        if scope in self._states:
            raise ValueError(f"scope {scope!r} already active")
        self._states[scope] = _ScopeState(spec=spec, spent=CostSnapshot.zero(), parent=parent)

    def exit_scope(self, scope: ScopeKey) -> None:
        self._states.pop(scope, None)

    def preview(self, context: BudgetContext, cost: CostSnapshot, *, label: str) -> BudgetChargeResult:
        return self._apply_charge(context, cost, label=label, commit=False)

    def commit(self, context: BudgetContext, cost: CostSnapshot, *, label: str) -> BudgetChargeResult:
        return self._apply_charge(context, cost, label=label, commit=True)

    def snapshot(self, scope: ScopeKey) -> BudgetChargeOutcome | None:
        state = self._states.get(scope)
        if state is None or state.spec is None:
            return None
        return self._build_outcome(scope, state, CostSnapshot.zero())

    def _apply_charge(
        self,
        context: BudgetContext,
        cost: CostSnapshot,
        *,
        label: str,
        commit: bool,
    ) -> BudgetChargeResult:
        outcomes: dict[ScopeKey, BudgetChargeOutcome] = {}
        should_stop = False
        warnings: list[str] = []

        for scope in context.scopes():
            state = self._states.get(scope)
            if state is None or state.spec is None:
                continue

            outcome = self._build_outcome(scope, state, cost)
            outcomes[scope] = outcome
            warnings.extend(outcome.warnings)
            if outcome.breached and self._enforces_stop(state.spec):
                should_stop = True

            if self._trace is not None:
                payload = {
                    "label": label,
                    "scope": scope.identifier,
                    "scope_type": scope.scope_type.value,
                    "spent": outcome.spent.as_dict(),
                    "remaining": outcome.remaining.as_dict(),
                    "overages": outcome.overages.as_dict(),
                    "breached": outcome.breached,
                    "mode": outcome.spec.mode.value,
                    "breach_action": outcome.spec.breach_action.value,
                }
                self._trace.emit("budget_charge", scope=scope.identifier, payload=payload)
                if outcome.breached:
                    self._trace.emit(
                        "budget_breach",
                        scope=scope.identifier,
                        payload={
                            "scope": scope.identifier,
                            "scope_type": scope.scope_type.value,
                            "overages": outcome.overages.as_dict(),
                            "mode": outcome.spec.mode.value,
                            "breach_action": outcome.spec.breach_action.value,
                            "label": label,
                        },
                    )

            if commit:
                new_spent = outcome.spent
                self._states[scope] = _ScopeState(spec=state.spec, spent=new_spent, parent=state.parent)

        return BudgetChargeResult(outcomes=MappingProxyType(outcomes), should_stop=should_stop, warnings=tuple(warnings))

    def _build_outcome(
        self,
        scope: ScopeKey,
        state: _ScopeState,
        cost: CostSnapshot,
    ) -> BudgetChargeOutcome:
        assert state.spec is not None
        new_spent = state.spent.add(cost)
        remaining_raw = state.spec.limits.subtract(new_spent)
        overages_raw = new_spent.subtract(state.spec.limits)
        remaining = CostSnapshot({key: max(value, 0.0) for key, value in remaining_raw.metrics.items()})
        overages = CostSnapshot({key: max(value, 0.0) for key, value in overages_raw.metrics.items()})
        breached = overages.any_positive()
        warnings: tuple[str, ...]
        if breached and state.spec.mode is BudgetMode.SOFT:
            warnings = (
                f"scope {scope.identifier} breached budget {state.spec.name} by {overages.as_dict()}",
            )
        elif breached:
            warnings = (
                f"scope {scope.identifier} exceeded budget {state.spec.name}",
            )
        else:
            warnings = ()
        return BudgetChargeOutcome(
            scope=scope,
            spec=state.spec,
            spent=new_spent,
            remaining=remaining,
            overages=overages,
            breached=breached,
            warnings=warnings,
        )

    def _enforces_stop(self, spec: BudgetSpec) -> bool:
        if spec.mode is BudgetMode.HARD:
            return True
        return spec.breach_action is BreachAction.STOP
