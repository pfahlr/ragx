from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, List


def _mapping_proxy(data: Mapping[str, float] | None = None) -> Mapping[str, float]:
    return MappingProxyType(dict(data or {}))


def _normalize_metrics(data: Mapping[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in data.items():
        if value < 0:
            raise ValueError(f"Budget metrics must be non-negative, got {key}={value}")
        if key == "time_s":
            normalized["time_ms"] = normalized.get("time_ms", 0.0) + float(value) * 1000.0
        else:
            normalized[key] = normalized.get(key, 0.0) + float(value)
    return normalized


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable cost representation with arithmetic helpers."""

    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "metrics", _mapping_proxy(self.metrics))

    @classmethod
    def from_mapping(cls, data: Mapping[str, float] | None = None) -> "CostSnapshot":
        return cls(metrics=_normalize_metrics(data or {}))

    @classmethod
    def zero(cls) -> "CostSnapshot":
        return cls.from_mapping({})

    def to_payload(self) -> Dict[str, float]:
        return dict(self.metrics)

    def __add__(self, other: "CostSnapshot") -> "CostSnapshot":
        keys = set(self.metrics) | set(other.metrics)
        combined = {k: self.metrics.get(k, 0.0) + other.metrics.get(k, 0.0) for k in keys}
        return CostSnapshot.from_mapping(combined)

    def __sub__(self, other: "CostSnapshot") -> "CostSnapshot":
        keys = set(self.metrics) | set(other.metrics)
        combined = {k: self.metrics.get(k, 0.0) - other.metrics.get(k, 0.0) for k in keys}
        return CostSnapshot.from_mapping({k: max(0.0, v) for k, v in combined.items()})

    def __mul__(self, factor: float) -> "CostSnapshot":
        scaled = {k: self.metrics.get(k, 0.0) * factor for k in self.metrics}
        # Allow negative factors during helper usage; clamp at creation
        for key, value in list(scaled.items()):
            if value < 0:
                scaled[key] = value
        return CostSnapshot(metrics=scaled)

    __rmul__ = __mul__

    def clamped_non_negative(self, delta: "CostSnapshot") -> "CostSnapshot":
        keys = set(self.metrics) | set(delta.metrics)
        adjusted: Dict[str, float] = {}
        for key in keys:
            value = self.metrics.get(key, 0.0) + delta.metrics.get(key, 0.0)
            adjusted[key] = max(0.0, value)
        return CostSnapshot.from_mapping(adjusted)


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Configuration describing limits and breach handling for a scope."""

    scope: str
    limits: Mapping[str, float]
    breach_action: str = "stop"

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        normalized = _normalize_metrics(dict(self.limits))
        object.__setattr__(self, "limits", _mapping_proxy(normalized))
        if self.breach_action not in {"stop", "warn"}:
            raise ValueError(f"Unsupported breach_action: {self.breach_action}")


@dataclass(frozen=True, slots=True)
class BudgetBreach:
    """Details about a budget breach for observability."""

    metric: str
    limit: float
    attempted: float
    breach_action: str


@dataclass(frozen=True, slots=True)
class BudgetChargeOutcome:
    """Result of a preflight or commit operation."""

    allowed: bool
    breaches: tuple[BudgetBreach, ...]
    remaining: CostSnapshot
    overage: CostSnapshot
    warnings: tuple[str, ...] = ()

    @property
    def breached(self) -> bool:
        return bool(self.breaches)


class TraceWriter:
    """Interface for trace emission."""

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ListTraceWriter(TraceWriter):
    """In-memory trace writer that enforces schema and chronological ordering."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._sequence: int = 0

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if "scope" not in payload or "data" not in payload:
            raise ValueError("Trace payload must include 'scope' and 'data' keys")
        scope = payload["scope"]
        data = payload["data"]
        if not isinstance(scope, str):
            raise ValueError("scope must be a string")
        if not isinstance(data, Mapping):
            raise ValueError("data must be a mapping")
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        event = {
            "timestamp": timestamp,
            "scope": scope,
            "event": event_type,
            "data": MappingProxyType(dict(data)),
            "sequence": self._sequence,
        }
        self.events.append(event)
        self._sequence += 1


class BudgetManager:
    """Coordinates budget enforcement across scopes."""

    def __init__(self, trace_writer: TraceWriter | None = None) -> None:
        self.trace_writer = trace_writer or ListTraceWriter()
        self._specs: Dict[str, BudgetSpec] = {}
        self._spent: Dict[str, CostSnapshot] = defaultdict(CostSnapshot.zero)
        self._stop_scopes: set[str] = set()

    def preflight(
        self,
        scope: str,
        budget: BudgetSpec,
        attempt: CostSnapshot,
    ) -> BudgetChargeOutcome:
        if scope not in self._specs:
            self._specs[scope] = budget
        return self._evaluate(scope, attempt, phase="budget_preflight")

    def commit(self, scope: str, cost: CostSnapshot) -> BudgetChargeOutcome:
        if scope not in self._specs:
            raise KeyError(f"Scope '{scope}' not registered")
        outcome = self._evaluate(scope, cost, phase="budget_charge")
        if outcome.breached and self._specs[scope].breach_action == "stop":
            self._stop_scopes.add(scope)
        self._spent[scope] = self._spent[scope] + cost
        return outcome

    def should_stop(self, scope: str) -> bool:
        return scope in self._stop_scopes

    def pop_scope(self, scope: str) -> None:
        self._specs.pop(scope, None)
        self._spent.pop(scope, None)
        self._stop_scopes.discard(scope)

    # Internal helpers
    def _evaluate(self, scope: str, delta: CostSnapshot, phase: str) -> BudgetChargeOutcome:
        spec = self._specs[scope]
        spent = self._spent[scope]
        projected = spent + delta

        breaches = self._detect_breaches(spec, projected)
        allowed = not breaches or spec.breach_action == "warn"
        remaining = self._compute_remaining(spec, projected)
        overage = self._compute_overage(spec, projected)
        warnings: tuple[str, ...] = ()
        if breaches and spec.breach_action == "warn":
            warnings = tuple(
                f"{scope} exceeded {breach.metric} by {breach.attempted - breach.limit:.2f}"
                for breach in breaches
            )

        payload = {
            "scope": scope,
            "data": {
                "phase": phase,
                "delta": delta.to_payload(),
                "spent": (spent + delta if phase == "budget_charge" else projected).to_payload(),
                "remaining": remaining.to_payload(),
                "overage": overage.to_payload(),
                "breach_action": spec.breach_action,
            },
        }
        self.trace_writer.emit(phase, payload)
        if breaches:
            breach_payload = {
                "scope": scope,
                "data": {
                    "breaches": [asdict(breach) for breach in breaches],
                    "breach_action": spec.breach_action,
                },
            }
            self.trace_writer.emit("budget_breach", breach_payload)

        return BudgetChargeOutcome(
            allowed=allowed,
            breaches=breaches,
            remaining=remaining,
            overage=overage,
            warnings=warnings,
        )

    @staticmethod
    def _detect_breaches(spec: BudgetSpec, cost: CostSnapshot) -> tuple[BudgetBreach, ...]:
        breaches: List[BudgetBreach] = []
        for metric, limit in spec.limits.items():
            attempted = cost.metrics.get(metric, 0.0)
            if attempted > limit:
                breaches.append(
                    BudgetBreach(
                        metric=metric,
                        limit=limit,
                        attempted=attempted,
                        breach_action=spec.breach_action,
                    )
                )
        return tuple(breaches)

    @staticmethod
    def _compute_remaining(spec: BudgetSpec, cost: CostSnapshot) -> CostSnapshot:
        remaining = {
            metric: max(0.0, limit - cost.metrics.get(metric, 0.0))
            for metric, limit in spec.limits.items()
        }
        return CostSnapshot.from_mapping(remaining)

    @staticmethod
    def _compute_overage(spec: BudgetSpec, cost: CostSnapshot) -> CostSnapshot:
        over = {
            metric: max(0.0, cost.metrics.get(metric, 0.0) - limit)
            for metric, limit in spec.limits.items()
        }
        return CostSnapshot.from_mapping(over)


class FlowRunner:
    """Simplified flow runner demonstrating budget guard integration."""

    def __init__(
        self,
        adapter: Any,
        trace_writer: TraceWriter | None = None,
        manager: BudgetManager | None = None,
    ) -> None:
        self.trace_writer = trace_writer or ListTraceWriter()
        self.manager = manager or BudgetManager(trace_writer=self.trace_writer)
        self.adapter = adapter

    def add_policy_trace(self, event: str, data: Mapping[str, Any]) -> None:
        self.trace_writer.emit(event, {"scope": "policy", "data": dict(data)})

    def get_trace(self) -> Sequence[Mapping[str, Any]]:
        if isinstance(self.trace_writer, ListTraceWriter):
            return tuple(self.trace_writer.events)
        raise AttributeError("Trace writer does not expose events")

    def run(
        self,
        nodes: Sequence[Mapping[str, Any]],
        budgets: Mapping[str, BudgetSpec],
    ) -> List[Any]:
        results: List[Any] = []
        run_spec = budgets.get("run")

        for node in nodes:
            node_id = node.get("id", str(len(results)))
            node_scope = f"node:{node_id}"
            node_spec = budgets.get(node_scope)
            estimate = None
            if node_spec is not None:
                estimate = self.adapter.estimate(node)
                pre = self.manager.preflight(node_scope, node_spec, estimate)
                if not pre.allowed and node_spec.breach_action == "stop":
                    break
            if run_spec is not None:
                if estimate is None:
                    estimate = self.adapter.estimate(node)
                pre_run = self.manager.preflight("run", run_spec, estimate)
                if not pre_run.allowed and run_spec.breach_action == "stop":
                    break

            result, cost = self.adapter.execute(node)
            results.append(result)

            if node_spec is not None:
                node_outcome = self.manager.commit(node_scope, cost)
                if not node_outcome.allowed and node_spec.breach_action == "stop":
                    if run_spec is not None:
                        self.manager.commit("run", cost)
                        self._mark_stop("run")
                    break
            if run_spec is not None:
                run_outcome = self.manager.commit("run", cost)
                if not run_outcome.allowed and run_spec.breach_action == "stop":
                    break
            if run_spec is not None and self.manager.should_stop("run"):
                break
        return results

    def _mark_stop(self, scope: str) -> None:
        self.manager._stop_scopes.add(scope)


__all__ = [
    "BudgetBreach",
    "BudgetChargeOutcome",
    "BudgetManager",
    "BudgetSpec",
    "CostSnapshot",
    "FlowRunner",
    "ListTraceWriter",
    "TraceWriter",
]
