from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from pkgs.dsl.models import mapping_proxy

from .trace import TraceEventEmitter


class BudgetMode(Enum):
    """Operational mode for a budget scope."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable snapshot describing spend per metric."""

    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {str(metric): float(amount) for metric, amount in self.metrics.items()}
        object.__setattr__(self, "metrics", mapping_proxy(normalized))

    def get(self, metric: str, default: float = 0.0) -> float:
        return float(self.metrics.get(metric, default))


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Configuration describing the limits for a scope."""

    scope: str
    limits: Mapping[str, float]
    mode: BudgetMode = BudgetMode.HARD
    breach_action: str = "stop"

    def __post_init__(self) -> None:
        normalized_limits = {str(metric): float(value) for metric, value in self.limits.items()}
        object.__setattr__(self, "limits", mapping_proxy(normalized_limits))
        if self.breach_action not in {"warn", "stop"}:
            raise ValueError("breach_action must be 'warn' or 'stop'")


@dataclass(frozen=True, slots=True)
class BudgetBreach:
    """Details about a specific metric exceeding its limit."""

    metric: str
    limit: float
    observed: float
    overage: float


@dataclass(frozen=True, slots=True)
class BudgetCheck:
    """Result of calling :meth:`BudgetManager.preflight`."""

    scope: str
    projected: CostSnapshot
    remaining: Mapping[str, float]
    breaches: tuple[BudgetBreach, ...]
    stop_requested: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    """Snapshot returned by :meth:`BudgetManager.commit`."""

    scope: str
    charged: CostSnapshot
    spent: CostSnapshot
    remaining: Mapping[str, float]
    overages: Mapping[str, float]
    breaches: tuple[BudgetBreach, ...]
    warnings: tuple[str, ...]
    stop: bool


class BudgetManager:
    """Co-ordinates budgets across run/loop/node scopes."""

    def __init__(
        self,
        specs: Mapping[str, BudgetSpec],
        *,
        trace_emitter: TraceEventEmitter | None = None,
    ) -> None:
        self._specs = {scope: spec for scope, spec in specs.items()}
        self._spent: dict[str, dict[str, float]] = {
            scope: {metric: 0.0 for metric in spec.limits}
            for scope, spec in self._specs.items()
        }
        self._emitter = trace_emitter

    def preflight(self, scope: str, projected_cost: Mapping[str, float]) -> BudgetCheck:
        spec, spent = self._resolve_scope(scope)
        normalized_cost = self._normalize_cost(projected_cost)
        projected_snapshot = CostSnapshot(normalized_cost)

        remaining: dict[str, float] = {}
        breaches: list[BudgetBreach] = []
        warnings: list[str] = []

        for metric, limit in spec.limits.items():
            projected_total = spent.get(metric, 0.0) + normalized_cost.get(metric, 0.0)
            remaining_value = max(0.0, limit - projected_total)
            remaining[metric] = remaining_value
            if projected_total > limit:
                overage = projected_total - limit
                breaches.append(
                    BudgetBreach(
                        metric=metric,
                        limit=limit,
                        observed=projected_total,
                        overage=overage,
                    )
                )
                if not self._should_stop(spec):
                    warnings.append(
                        f"{scope}:{metric} projected over by {overage:.2f}"
                    )

        stop_requested = self._should_stop(spec) and bool(breaches)
        check = BudgetCheck(
            scope=scope,
            projected=projected_snapshot,
            remaining=mapping_proxy(remaining),
            breaches=tuple(breaches),
            stop_requested=stop_requested,
            warnings=tuple(warnings),
        )
        self._emit_preflight(spec, check)
        return check

    def commit(self, scope: str, actual_cost: Mapping[str, float]) -> BudgetChargeOutcome:
        spec, spent = self._resolve_scope(scope)
        normalized_cost = self._normalize_cost(actual_cost)

        for metric, value in normalized_cost.items():
            spent[metric] = spent.get(metric, 0.0) + value

        charged_snapshot = CostSnapshot(normalized_cost)
        spent_snapshot = CostSnapshot(spent)

        remaining: dict[str, float] = {}
        overages: dict[str, float] = {}
        breaches: list[BudgetBreach] = []

        for metric, limit in spec.limits.items():
            observed = spent.get(metric, 0.0)
            remaining_value = max(0.0, limit - observed)
            remaining[metric] = remaining_value
            if observed > limit:
                overage = observed - limit
                overages[metric] = overage
                breaches.append(
                    BudgetBreach(
                        metric=metric,
                        limit=limit,
                        observed=observed,
                        overage=overage,
                    )
                )
            else:
                overages[metric] = 0.0

        stop = self._should_stop(spec) and bool(breaches)
        warnings = self._build_warnings(scope, spec, breaches, stop)

        outcome = BudgetChargeOutcome(
            scope=scope,
            charged=charged_snapshot,
            spent=spent_snapshot,
            remaining=mapping_proxy(remaining),
            overages=mapping_proxy(overages),
            breaches=tuple(breaches),
            warnings=tuple(warnings),
            stop=stop,
        )
        self._emit_events(spec, outcome)
        return outcome

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_scope(self, scope: str) -> tuple[BudgetSpec, dict[str, float]]:
        if scope not in self._specs:
            raise KeyError(f"Unknown budget scope: {scope}")
        return self._specs[scope], self._spent[scope]

    @staticmethod
    def _normalize_cost(cost: Mapping[str, float]) -> dict[str, float]:
        return {str(metric): float(value) for metric, value in cost.items()}

    @staticmethod
    def _should_stop(spec: BudgetSpec) -> bool:
        if spec.mode is BudgetMode.HARD:
            return True
        return spec.breach_action == "stop"

    def _build_warnings(
        self,
        scope: str,
        spec: BudgetSpec,
        breaches: list[BudgetBreach],
        stop: bool,
    ) -> list[str]:
        if stop or not breaches:
            return []
        return [f"{scope}:{breach.metric} exceeded by {breach.overage:.2f}" for breach in breaches]

    def _emit_events(self, spec: BudgetSpec, outcome: BudgetChargeOutcome) -> None:
        if self._emitter is None:
            return

        payload = {
            "charged": outcome.charged.metrics,
            "spent": outcome.spent.metrics,
            "remaining": outcome.remaining,
            "overages": outcome.overages,
            "warnings": outcome.warnings,
        }
        breach_kind = "hard" if outcome.stop else "soft"
        self._emitter.emit(
            event="budget_charge",
            scope=spec.scope,
            scope_type="budget",
            payload=payload,
            breach_kind=breach_kind,
        )
        if outcome.breaches:
            breach_payload = {
                "breaches": [
                    {
                        "metric": breach.metric,
                        "observed": breach.observed,
                        "limit": breach.limit,
                        "overage": breach.overage,
                    }
                    for breach in outcome.breaches
                ],
                "stop": outcome.stop,
            }
            self._emitter.emit(
                event="budget_breach",
                scope=spec.scope,
                scope_type="budget",
                payload=breach_payload,
                breach_kind=breach_kind,
            )

    def _emit_preflight(self, spec: BudgetSpec, check: BudgetCheck) -> None:
        if self._emitter is None or not check.breaches:
            return
        payload = {
            "breaches": [
                {
                    "metric": breach.metric,
                    "limit": breach.limit,
                    "projected": breach.observed,
                    "overage": breach.overage,
                }
                for breach in check.breaches
            ],
            "warnings": check.warnings,
            "phase": "preflight",
        }
        breach_kind = "hard" if check.stop_requested else "soft"
        self._emitter.emit(
            event="budget_breach",
            scope=spec.scope,
            scope_type="budget",
            payload=payload,
            breach_kind=breach_kind,
        )
