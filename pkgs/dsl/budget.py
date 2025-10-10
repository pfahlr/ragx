"""Budget enforcement primitives for the FlowRunner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .models import mapping_proxy

__all__ = [
    "BudgetMode",
    "BudgetSpec",
    "BudgetScope",
    "BudgetCharge",
    "BudgetOutcome",
    "BudgetManager",
]


class BudgetMode(str, Enum):
    """Supported budget enforcement modes."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Immutable specification describing a single budget scope."""

    name: str
    limit: Mapping[str, float]
    mode: BudgetMode
    breach_action: str

    def __post_init__(self) -> None:  # pragma: no cover - dataclass post init
        object.__setattr__(self, "limit", mapping_proxy(self.limit))
        object.__setattr__(self, "breach_action", self.breach_action.lower())

    @staticmethod
    def from_mapping(mapping: Mapping[str, object], *, name: str | None = None) -> BudgetSpec:
        """Normalise a mapping describing a budget specification."""

        provided_name = name or str(mapping.get("name") or "")
        raw_limit = mapping.get("limit")
        if not isinstance(raw_limit, Mapping) or not raw_limit:
            raise ValueError("budget limit must be a non-empty mapping")
        limit: dict[str, float] = {}
        for metric, value in raw_limit.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"invalid numeric value for metric {metric!r}") from exc
            if numeric < 0:
                raise ValueError("budget limits must be non-negative")
            limit[str(metric)] = numeric

        mode_value = str(mapping.get("mode", BudgetMode.HARD.value)).lower()
        try:
            mode = BudgetMode(mode_value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"unsupported budget mode: {mode_value}") from exc

        breach_action = str(mapping.get("breach_action", "stop")).lower()
        if breach_action not in {"stop", "warn"}:
            raise ValueError(f"unsupported breach action: {breach_action}")

        return BudgetSpec(
            name=provided_name,
            limit=limit,
            mode=mode,
            breach_action=breach_action,
        )


@dataclass(frozen=True, slots=True)
class BudgetScope:
    """Identify a budget scope tracked by the manager."""

    kind: str
    identifier: str

    @classmethod
    def run(cls, identifier: str) -> BudgetScope:
        return cls("run", identifier)

    @classmethod
    def loop(cls, identifier: str) -> BudgetScope:
        return cls("loop", identifier)

    @classmethod
    def node(cls, identifier: str) -> BudgetScope:
        return cls("node", identifier)

    @classmethod
    def spec(cls, identifier: str) -> BudgetScope:
        return cls("spec", identifier)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.kind}:{self.identifier}"


@dataclass(frozen=True, slots=True)
class BudgetCharge:
    """Result of charging a budget scope."""

    scope: BudgetScope
    spec: BudgetSpec
    cost: Mapping[str, float]
    remaining: Mapping[str, float]
    overages: Mapping[str, float]
    breached: bool

    def __post_init__(self) -> None:  # pragma: no cover - dataclass post init
        object.__setattr__(self, "cost", mapping_proxy(self.cost))
        object.__setattr__(self, "remaining", mapping_proxy(self.remaining))
        object.__setattr__(self, "overages", mapping_proxy(self.overages))

    @property
    def action(self) -> str:
        """Return the breach action associated with the budget specification."""

        return self.spec.breach_action


@dataclass(frozen=True, slots=True)
class BudgetOutcome:
    """Aggregate preview/commit outcome across multiple scopes."""

    charges: tuple[BudgetCharge, ...]
    warnings: tuple[BudgetCharge, ...]
    should_stop: bool

    @staticmethod
    def from_charges(charges: Sequence[BudgetCharge]) -> BudgetOutcome:
        warning_charges = tuple(
            charge for charge in charges if charge.breached and charge.action == "warn"
        )
        should_stop = any(
            charge.breached and charge.action == "stop" for charge in charges
        )
        return BudgetOutcome(
            charges=tuple(charges),
            warnings=warning_charges,
            should_stop=should_stop,
        )


class _BudgetMeter:
    """Track spend for a particular scope."""

    def __init__(self, spec: BudgetSpec) -> None:
        self.spec = spec
        self._spent: dict[str, float] = {}

    def evaluate(
        self, cost: Mapping[str, float], scope: BudgetScope, *, mutate: bool
    ) -> BudgetCharge:
        totals = self._spent.copy()
        for metric, value in cost.items():
            totals[metric] = totals.get(metric, 0.0) + value

        remaining: dict[str, float] = {}
        overages: dict[str, float] = {}
        breached = False
        for metric, limit in self.spec.limit.items():
            current = totals.get(metric, 0.0)
            remaining_value = max(limit - current, 0.0)
            remaining[metric] = remaining_value
            overage_value = max(current - limit, 0.0)
            if overage_value > 0:
                breached = True
                overages[metric] = overage_value

        if mutate:
            self._spent = totals

        return BudgetCharge(
            scope=scope,
            spec=self.spec,
            cost=dict(cost),
            remaining=remaining,
            overages=overages,
            breached=breached,
        )


class BudgetManager:
    """Coordinate budget meters across run/node/loop/spec scopes."""

    def __init__(self) -> None:
        self._meters: dict[BudgetScope, _BudgetMeter] = {}

    def register(self, scope: BudgetScope, spec: BudgetSpec) -> None:
        if scope in self._meters:
            raise ValueError(f"budget scope already registered: {scope}")
        self._meters[scope] = _BudgetMeter(spec)

    def preview(
        self, cost: Mapping[str, float], scopes: Sequence[BudgetScope]
    ) -> BudgetOutcome:
        normalized = _normalize_cost(cost)
        charges = [
            meter.evaluate(normalized, scope, mutate=False)
            for scope in scopes
            if (meter := self._meters.get(scope)) is not None
        ]
        return BudgetOutcome.from_charges(charges)

    def commit(
        self, cost: Mapping[str, float], scopes: Sequence[BudgetScope]
    ) -> BudgetOutcome:
        normalized = _normalize_cost(cost)
        charges = [
            meter.evaluate(normalized, scope, mutate=True)
            for scope in scopes
            if (meter := self._meters.get(scope)) is not None
        ]
        return BudgetOutcome.from_charges(charges)


def _normalize_cost(cost: Mapping[str, float]) -> Mapping[str, float]:
    """Normalize metric keys and ensure floats."""

    normalized: dict[str, float] = {}
    for raw_metric, value in cost.items():
        metric = str(raw_metric)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid cost value for metric {metric!r}") from exc
        if numeric < 0:
            raise ValueError("cost values must be non-negative")
        if metric.endswith("_seconds"):
            metric = metric[: -len("_seconds")] + "_ms"
            numeric *= 1000
        normalized[metric] = normalized.get(metric, 0.0) + numeric
    return normalized
