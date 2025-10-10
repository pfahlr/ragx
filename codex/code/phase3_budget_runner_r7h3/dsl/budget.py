"""Budget domain objects and orchestration for FlowRunner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

from . import costs


class BudgetBreachError(RuntimeError):
    """Raised when a budget breach prevents execution."""

    def __init__(self, check: "BudgetCheck", message: str | None = None) -> None:
        self.check = check
        super().__init__(message or f"Budget breach: {check.action.value}")


class BudgetAction(Enum):
    """Represents the dominant action to take after a budget evaluation."""

    ALLOW = "allow"
    WARN = "warn"
    STOP = "stop"
    ERROR = "error"

    @classmethod
    def from_breach_action(cls, breach_action: str) -> "BudgetAction":
        mapping = {
            "warn": cls.WARN,
            "stop": cls.STOP,
            "error": cls.ERROR,
        }
        try:
            return mapping[breach_action]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported breach_action: {breach_action}") from exc


@dataclass(frozen=True)
class BudgetSpec:
    """Immutable definition of a budget for a given scope."""

    scope: str
    limits: Mapping[str, float]
    breach_action: str = "error"

    def __post_init__(self) -> None:
        normalized = costs.normalize_cost(self.limits)
        object.__setattr__(self, "limits", MappingProxyType(dict(normalized)))
        if self.breach_action not in {"warn", "stop", "error"}:
            raise ValueError(f"Unsupported breach_action: {self.breach_action}")


@dataclass(frozen=True)
class ScopeBudgetStatus:
    """Diagnostic information for a single scope during preview/commit."""

    scope: str
    breached: bool
    action: BudgetAction
    remaining: MappingProxyType[str, float]
    overages: MappingProxyType[str, float]


@dataclass(frozen=True)
class BudgetCheck:
    """Result of previewing a cost against one or more budget scopes."""

    allowed: bool
    action: BudgetAction
    charge: MappingProxyType[str, float]
    scope_statuses: Tuple[ScopeBudgetStatus, ...]


@dataclass(frozen=True)
class BudgetChargeOutcome:
    """Result after successfully committing a cost to the manager."""

    allowed: bool
    action: BudgetAction
    charge: MappingProxyType[str, float]
    remaining: MappingProxyType[str, float]
    overages: MappingProxyType[str, float]
    scope_statuses: Tuple[ScopeBudgetStatus, ...]


class BudgetManager:
    """Coordinates budget scopes and exposes preview/commit APIs."""

    def __init__(self, specs: Mapping[str, BudgetSpec]) -> None:
        self._specs: Dict[str, BudgetSpec] = dict(specs)
        self._spent: Dict[str, Dict[str, float]] = {scope: {} for scope in self._specs}

    def preview(self, scopes: Sequence[str], cost: Mapping[str, float | int]) -> BudgetCheck:
        normalized = costs.normalize_cost(cost)
        charge_proxy = MappingProxyType(dict(normalized))
        statuses = []
        dominant_action = BudgetAction.ALLOW
        severity = {BudgetAction.ALLOW: 0, BudgetAction.WARN: 1, BudgetAction.STOP: 2, BudgetAction.ERROR: 3}

        for scope in scopes:
            spec = self._specs.get(scope)
            if spec is None:
                status = ScopeBudgetStatus(
                    scope=scope,
                    breached=False,
                    action=BudgetAction.ALLOW,
                    remaining=MappingProxyType({}),
                    overages=MappingProxyType({}),
                )
                statuses.append(status)
                continue

            spent = self._spent.setdefault(scope, {})
            new_totals = costs.combine_costs([spent, normalized])
            remaining: Dict[str, float] = {}
            overages: Dict[str, float] = {}
            breached = False
            for metric, limit in spec.limits.items():
                value = new_totals.get(metric, 0.0)
                remaining_value = max(limit - value, 0.0)
                remaining[metric] = remaining_value
                overage_value = max(value - limit, 0.0)
                if overage_value > 0:
                    breached = True
                    overages[metric] = overage_value
            action = BudgetAction.from_breach_action(spec.breach_action) if breached else BudgetAction.ALLOW
            statuses.append(
                ScopeBudgetStatus(
                    scope=scope,
                    breached=breached,
                    action=action,
                    remaining=MappingProxyType(dict(remaining)),
                    overages=MappingProxyType(dict(overages)),
                )
            )
            if action is not BudgetAction.ALLOW and severity[action] > severity[dominant_action]:
                dominant_action = action

        ordered_statuses = tuple(sorted(statuses, key=lambda s: (-severity[s.action], s.scope)))
        allowed = dominant_action in {BudgetAction.ALLOW, BudgetAction.WARN}
        return BudgetCheck(
            allowed=allowed,
            action=dominant_action,
            charge=charge_proxy,
            scope_statuses=ordered_statuses,
        )

    def commit(self, scopes: Sequence[str], cost: Mapping[str, float | int]) -> BudgetChargeOutcome:
        check = self.preview(scopes, cost)
        if check.action in {BudgetAction.ERROR, BudgetAction.STOP}:
            raise BudgetBreachError(check)

        normalized = dict(check.charge)
        for scope in scopes:
            spec = self._specs.get(scope)
            if spec is None:
                continue
            spent = self._spent.setdefault(scope, {})
            for metric, value in normalized.items():
                spent[metric] = spent.get(metric, 0.0) + value

        dominant_status = check.scope_statuses[0] if check.scope_statuses else ScopeBudgetStatus(
            scope="run",
            breached=False,
            action=BudgetAction.ALLOW,
            remaining=MappingProxyType({}),
            overages=MappingProxyType({}),
        )
        return BudgetChargeOutcome(
            allowed=True,
            action=check.action,
            charge=check.charge,
            remaining=dominant_status.remaining,
            overages=dominant_status.overages,
            scope_statuses=check.scope_statuses,
        )
