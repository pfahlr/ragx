"""Budget domain model and manager integration for the sandbox FlowRunner.

The implementation threads together:
* Immutable budget snapshots from `codex/integrate-budget-guards-with-runner-zwi2ny`.
* Structured overage accounting from `codex/integrate-budget-guards-with-runner-pbdel9`.
* Budget mode semantics originating in `codex/implement-budget-guards-with-test-first-approach-qhq0jq`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Dict

from .costs import normalize_cost
from .trace import TraceEventEmitter


class BudgetMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    metrics: Mapping[str, float]

    @classmethod
    def from_raw(cls, raw: Mapping[str, float | int]) -> "CostSnapshot":
        return cls(metrics=normalize_cost(raw))

    def as_dict(self) -> dict[str, float]:
        return dict(self.metrics)


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    scope_id: str
    limits: Mapping[str, float]
    mode: BudgetMode = BudgetMode.HARD
    breach_action: str = "stop"

    def __post_init__(self) -> None:
        normalised = normalize_cost(self.limits)
        object.__setattr__(self, "limits", normalised)
        action = self.breach_action.lower()
        if action not in {"warn", "stop"}:
            raise ValueError("breach_action must be either 'warn' or 'stop'")
        object.__setattr__(self, "breach_action", action)


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    run_id: str
    node_id: str
    scope: str
    scope_type: str
    spec: BudgetSpec
    cost: CostSnapshot
    spent: Mapping[str, float]
    projected_spend: Mapping[str, float]
    remaining: Mapping[str, float]
    overages: Mapping[str, float]
    breach_kind: str
    should_stop: bool


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome(BudgetDecision):
    """Outcome returned after a successful commit."""


class BudgetHardStop(RuntimeError):
    """Raised when a hard budget requires termination."""

    def __init__(self, scope: str, overages: Mapping[str, float]) -> None:
        formatted = ", ".join(f"{metric}:+{value:.2f}" for metric, value in overages.items()) or "no overages"
        super().__init__(f"Budget exceeded for {scope}: {formatted}")
        self.scope = scope
        self.overages = MappingProxyType(dict(overages))


@dataclass
class _BudgetAccount:
    spec: BudgetSpec
    spent: Dict[str, float]


class BudgetManager:
    """Coordinates budget preflight and commit operations while emitting traces."""

    def __init__(self, emitter: TraceEventEmitter) -> None:
        self._emitter = emitter
        self._accounts: dict[str, _BudgetAccount] = {}

    def preflight(
        self,
        run_id: str,
        scope: str,
        scope_type: str,
        node_id: str,
        spec: BudgetSpec,
        cost: CostSnapshot,
    ) -> BudgetDecision:
        account = self._accounts.get(scope)
        if account is None:
            account = _BudgetAccount(spec=spec, spent={})
            self._accounts[scope] = account
        else:
            # Allow updated limits but keep cumulative spend.
            account.spec = spec

        spent = dict(account.spent)
        projected = _add_costs(spent, cost.metrics)

        remaining = _remaining(spec.limits, projected)
        overages = _overages(spec.limits, projected)
        has_overage = any(value > 0 for value in overages.values())

        if not has_overage:
            breach_kind = "none"
            should_stop = False
        elif spec.mode is BudgetMode.SOFT and spec.breach_action != "stop":
            breach_kind = "soft"
            should_stop = False
        else:
            breach_kind = "hard"
            should_stop = True

        return BudgetDecision(
            run_id=run_id,
            node_id=node_id,
            scope=scope,
            scope_type=scope_type,
            spec=spec,
            cost=cost,
            spent=_freeze_mapping(spent),
            projected_spend=_freeze_mapping(projected),
            remaining=_freeze_mapping(remaining),
            overages=_freeze_mapping({k: v for k, v in overages.items() if v > 0}),
            breach_kind=breach_kind,
            should_stop=should_stop,
        )

    def commit(self, decision: BudgetDecision) -> BudgetChargeOutcome:
        account = self._accounts[decision.scope]

        self._emitter.budget_charge(
            scope=decision.scope,
            scope_type=decision.scope_type,
            run_id=decision.run_id,
            node_id=decision.node_id,
            cost=decision.cost.metrics,
            remaining=decision.remaining,
        )

        if decision.breach_kind != "none":
            self._emitter.budget_breach(
                scope=decision.scope,
                scope_type=decision.scope_type,
                run_id=decision.run_id,
                node_id=decision.node_id,
                overages=decision.overages,
                remaining=decision.remaining,
                severity="hard" if decision.should_stop else "soft",
            )

        if decision.should_stop:
            raise BudgetHardStop(decision.scope, decision.overages)

        account.spent = dict(decision.projected_spend)
        payload = {field.name: getattr(decision, field.name) for field in fields(BudgetDecision)}
        return BudgetChargeOutcome(**payload)


def _add_costs(spent: Mapping[str, float], cost: Mapping[str, float]) -> Dict[str, float]:
    combined: Dict[str, float] = dict(spent)
    for key, value in cost.items():
        combined[key] = combined.get(key, 0.0) + float(value)
    return combined


def _remaining(limits: Mapping[str, float], projected: Mapping[str, float]) -> Dict[str, float]:
    remaining: Dict[str, float] = {}
    for metric, limit in limits.items():
        remaining[metric] = float(limit) - float(projected.get(metric, 0.0))
    return remaining


def _overages(limits: Mapping[str, float], projected: Mapping[str, float]) -> Dict[str, float]:
    over: Dict[str, float] = {}
    for metric, limit in limits.items():
        projected_value = float(projected.get(metric, 0.0))
        if projected_value > float(limit):
            over[metric] = projected_value - float(limit)
        else:
            over[metric] = 0.0
    return over


def _freeze_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({key: float(amount) for key, amount in values.items()})
