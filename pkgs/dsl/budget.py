"""Budget enforcement primitives for FlowRunner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Iterable

from .trace import RunnerTraceEvent, RunnerTraceRecorder, emit_trace_event

__all__ = [
    "BudgetError",
    "BudgetBreachHard",
    "BudgetWarning",
    "BudgetEvaluation",
    "BudgetChargeResult",
    "BudgetPreflightResult",
    "BudgetCommitResult",
    "LoopIterationOutcome",
    "BudgetMeter",
    "BudgetManager",
]


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable representation of cumulative or incremental spend."""

    usd: float
    tokens: int
    calls: int
    time_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "usd": self.usd,
            "tokens": self.tokens,
            "calls": self.calls,
            "time_ms": self.time_ms,
        }


@dataclass(frozen=True, slots=True)
class BudgetRemaining:
    """Remaining budget per metric; ``None`` represents unlimited."""

    usd: float | None
    tokens: int | None
    calls: int | None
    time_ms: float | None

    def as_dict(self) -> dict[str, object | None]:
        return {
            "usd": self.usd,
            "tokens": self.tokens,
            "calls": self.calls,
            "time_ms": self.time_ms,
        }


@dataclass(frozen=True, slots=True)
class BudgetEvaluation:
    """Result produced by :meth:`BudgetMeter.evaluate`."""

    cost: CostSnapshot
    breached: bool
    breach_kind: str | None
    metrics: tuple[str, ...]
    allowed: bool
    remaining: BudgetRemaining
    breach_action: str


@dataclass(frozen=True, slots=True)
class BudgetChargeResult:
    """Outcome produced when charging a meter."""

    meter: "BudgetMeter"
    cost: CostSnapshot
    spent: CostSnapshot
    remaining: BudgetRemaining
    breached: bool
    breach_kind: str | None
    metrics: tuple[str, ...]
    breach_action: str
    should_stop: bool


@dataclass(frozen=True, slots=True)
class BudgetWarning:
    """Structured warning describing a soft or stop-on-budget breach."""

    scope_type: str
    scope_id: str
    metrics: tuple[str, ...]
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class BudgetPreflightResult:
    """Preflight evaluation result emitted by :meth:`BudgetManager.preflight_node`."""

    warnings: tuple[BudgetWarning, ...]


@dataclass(frozen=True, slots=True)
class BudgetCommitResult:
    """Commit outcome for node execution."""

    warnings: tuple[BudgetWarning, ...]


@dataclass(frozen=True, slots=True)
class LoopIterationOutcome:
    """Outcome after charging a loop iteration."""

    should_stop: bool
    stop_reason: str | None
    warnings: tuple[BudgetWarning, ...]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetError(RuntimeError):
    """Base error for budget enforcement failures."""


class BudgetBreachHard(BudgetError):
    """Raised when a hard cap is exceeded and execution must halt."""

    def __init__(self, meter: "BudgetMeter", evaluation: BudgetEvaluation) -> None:
        metrics = evaluation.metrics or meter.metrics_exceeded()
        metrics_str = ", ".join(metrics) or "budget"
        message = (
            f"{meter.scope_type}:{meter.scope_id} hard budget exceeded for {metrics_str}"
        )
        super().__init__(message)
        self.scope_type = meter.scope_type
        self.scope_id = meter.scope_id
        self.metrics = tuple(metrics)
        self.remaining = evaluation.remaining
        self.breach_action = evaluation.breach_action


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise BudgetError(f"invalid numeric value: {value!r}") from exc
    if math.isfinite(number) and number <= 0:
        return None
    return number


def _normalize_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise BudgetError(f"invalid integer value: {value!r}") from exc
    if number <= 0:
        return None
    return number


def _normalize_time_ms(config: Mapping[str, object]) -> float | None:
    if "time_limit_ms" in config:
        limit = _normalize_float(config.get("time_limit_ms"))
        if limit is None:
            return None
        return limit
    if "time_limit_sec" in config:
        limit = _normalize_float(config.get("time_limit_sec"))
        if limit is None:
            return None
        return limit * 1000.0
    if "max_time_ms" in config:
        limit = _normalize_float(config.get("max_time_ms"))
        if limit is None:
            return None
        return limit
    return None


def _normalize_cost(cost: Mapping[str, object]) -> CostSnapshot:
    usd = float(cost.get("usd") or cost.get("cost_usd") or 0.0)

    tokens_value = cost.get("tokens")
    if tokens_value is None:
        tokens_in = int(cost.get("tokens_in") or 0)
        tokens_out = int(cost.get("tokens_out") or 0)
        tokens_value = tokens_in + tokens_out
    else:
        tokens_value = int(tokens_value)

    calls_value = int(cost.get("calls") or cost.get("call_count") or 0)

    if "time_ms" in cost:
        time_ms = float(cost.get("time_ms") or 0.0)
    elif "time_sec" in cost:
        time_ms = float(cost.get("time_sec") or 0.0) * 1000.0
    else:
        time_ms = float(cost.get("time_seconds") or 0.0) * 1000.0

    return CostSnapshot(usd=usd, tokens=tokens_value, calls=calls_value, time_ms=time_ms)


# ---------------------------------------------------------------------------
# Budget meter implementation
# ---------------------------------------------------------------------------


class BudgetMeter:
    """Tracks spend and enforces a hard/soft budget for a scope."""

    def __init__(
        self,
        *,
        scope_type: str,
        scope_id: str,
        config: Mapping[str, object] | None,
        default_mode: str = "hard",
        breach_action: str | None = None,
    ) -> None:
        self.scope_type = scope_type
        self.scope_id = scope_id
        data = dict(config or {})

        mode = str(data.get("mode", default_mode or "hard")).lower()
        if mode not in {"hard", "soft"}:
            raise BudgetError(f"unsupported budget mode: {mode!r}")
        self.mode = mode

        action_default = "stop" if scope_type == "loop" else "error"
        action = str(data.get("breach_action", breach_action or action_default)).lower()
        if action not in {"error", "stop", "warn"}:
            raise BudgetError(f"unsupported breach_action: {action!r}")
        if mode == "soft" and action == "error":
            action = "warn"
        if mode == "soft" and action == "stop":
            action = "warn"
        if mode == "hard" and action == "warn":
            action = "error"
        self._breach_action = action

        self._limits = {
            "usd": _normalize_float(data.get("max_usd")),
            "tokens": _normalize_int(data.get("max_tokens")),
            "calls": _normalize_int(data.get("max_calls")),
            "time_ms": _normalize_time_ms(data),
        }

        self._spent_usd = 0.0
        self._spent_tokens = 0
        self._spent_calls = 0
        self._spent_time_ms = 0.0

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def breach_action(self) -> str:
        return self._breach_action

    @property
    def is_unbounded(self) -> bool:
        return all(limit is None for limit in self._limits.values())

    def remaining(self) -> BudgetRemaining:
        return BudgetRemaining(
            usd=self._remaining_value("usd"),
            tokens=self._remaining_value("tokens"),
            calls=self._remaining_value("calls"),
            time_ms=self._remaining_value("time_ms"),
        )

    def limits(self) -> BudgetRemaining:
        return BudgetRemaining(
            usd=self._limits["usd"],
            tokens=self._limits["tokens"],
            calls=self._limits["calls"],
            time_ms=self._limits["time_ms"],
        )

    def spent_snapshot(self) -> CostSnapshot:
        return CostSnapshot(
            usd=self._spent_usd,
            tokens=self._spent_tokens,
            calls=self._spent_calls,
            time_ms=self._spent_time_ms,
        )

    # ------------------------------------------------------------------
    # Evaluation and enforcement
    # ------------------------------------------------------------------
    def can_spend(self, cost: Mapping[str, object]) -> bool:
        evaluation = self.evaluate(cost)
        return evaluation.allowed

    def evaluate(self, cost: Mapping[str, object]) -> BudgetEvaluation:
        snapshot = _normalize_cost(cost)
        metrics = self._metrics_for(snapshot)
        breached = bool(metrics)
        breach_kind = None
        allowed = True
        if breached:
            breach_kind = "soft" if self.mode == "soft" else "hard"
            allowed = breach_kind == "soft"
        remaining = self._remaining_after(snapshot)
        return BudgetEvaluation(
            cost=snapshot,
            breached=breached,
            breach_kind=breach_kind,
            metrics=metrics,
            allowed=allowed,
            remaining=remaining,
            breach_action=self._breach_action,
        )

    def charge(
        self,
        cost: Mapping[str, object],
        *,
        evaluation: BudgetEvaluation | None = None,
    ) -> BudgetChargeResult:
        evaluation = evaluation or self.evaluate(cost)
        if evaluation.breach_kind == "hard" and self._breach_action != "stop":
            raise BudgetBreachHard(self, evaluation)

        snapshot = evaluation.cost
        self._apply(snapshot)

        metrics_after = self.metrics_exceeded()
        if metrics_after:
            breach_kind = "soft" if self.mode == "soft" else "hard"
        else:
            breach_kind = evaluation.breach_kind
        metrics = metrics_after or evaluation.metrics
        breached = bool(metrics)

        should_stop = breach_kind == "hard" and self._breach_action == "stop"
        remaining = self.remaining()
        result = BudgetChargeResult(
            meter=self,
            cost=snapshot,
            spent=self.spent_snapshot(),
            remaining=remaining,
            breached=breached,
            breach_kind=breach_kind,
            metrics=metrics,
            breach_action=self._breach_action,
            should_stop=should_stop,
        )
        if breach_kind == "hard" and self._breach_action != "stop":  # pragma: no cover
            # Defensive check: if we crossed due to inaccurate preflight, raise now.
            raise BudgetBreachHard(self, self.evaluate(cost))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply(self, cost: CostSnapshot) -> None:
        self._spent_usd = math.fsum((self._spent_usd, cost.usd))
        self._spent_tokens += cost.tokens
        self._spent_calls += cost.calls
        self._spent_time_ms = math.fsum((self._spent_time_ms, cost.time_ms))

    def _remaining_value(self, metric: str) -> float | int | None:
        limit = self._limits[metric]
        if limit is None:
            return None
        spent = {
            "usd": self._spent_usd,
            "tokens": self._spent_tokens,
            "calls": self._spent_calls,
            "time_ms": self._spent_time_ms,
        }[metric]
        return limit - spent

    def _remaining_after(self, cost: CostSnapshot) -> BudgetRemaining:
        def calc(metric: str, increment: float | int) -> float | int | None:
            limit = self._limits[metric]
            if limit is None:
                return None
            spent = {
                "usd": self._spent_usd,
                "tokens": self._spent_tokens,
                "calls": self._spent_calls,
                "time_ms": self._spent_time_ms,
            }[metric]
            return limit - (spent + increment)

        return BudgetRemaining(
            usd=calc("usd", cost.usd),
            tokens=calc("tokens", cost.tokens),
            calls=calc("calls", cost.calls),
            time_ms=calc("time_ms", cost.time_ms),
        )

    def _metrics_for(self, cost: CostSnapshot) -> tuple[str, ...]:
        metrics: list[str] = []
        proposed = {
            "usd": self._spent_usd + cost.usd,
            "tokens": self._spent_tokens + cost.tokens,
            "calls": self._spent_calls + cost.calls,
            "time_ms": self._spent_time_ms + cost.time_ms,
        }
        for metric, proposed_value in proposed.items():
            limit = self._limits[metric]
            if limit is None:
                continue
            if self._breach_action == "stop":
                threshold_exceeded = proposed_value >= limit
            else:
                threshold_exceeded = proposed_value > limit
            if threshold_exceeded:
                metrics.append(metric)
        return tuple(metrics)

    def metrics_exceeded(self) -> tuple[str, ...]:
        metrics: list[str] = []
        spent = {
            "usd": self._spent_usd,
            "tokens": self._spent_tokens,
            "calls": self._spent_calls,
            "time_ms": self._spent_time_ms,
        }
        for metric, value in spent.items():
            limit = self._limits[metric]
            if limit is None:
                continue
            if self._breach_action == "stop":
                threshold_exceeded = value >= limit
            else:
                threshold_exceeded = value > limit
            if threshold_exceeded:
                metrics.append(metric)
        return tuple(metrics)


# ---------------------------------------------------------------------------
# Budget manager orchestrating scopes
# ---------------------------------------------------------------------------


class BudgetManager:
    """Manage run, node, and loop budgets for :class:`FlowRunner`."""

    def __init__(
        self,
        *,
        trace: RunnerTraceRecorder | None = None,
        event_sink: Callable[[RunnerTraceEvent], None] | None = None,
    ) -> None:
        self._trace = trace
        self._event_sink = event_sink
        self._run_meter: BudgetMeter | None = None
        self._node_meters: dict[str, BudgetMeter] = {}
        self._node_soft_meters: dict[str, BudgetMeter] = {}
        self._loop_meters: dict[str, BudgetMeter] = {}

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear previously registered budget scopes."""

        self._run_meter = None
        self._node_meters.clear()
        self._node_soft_meters.clear()
        self._loop_meters.clear()

    def configure_run(self, budget: Mapping[str, object] | None) -> None:
        meter = BudgetMeter(scope_type="run", scope_id="run", config=budget)
        self._run_meter = None if meter.is_unbounded else meter

    def register_node(
        self,
        node_id: str,
        *,
        hard_budget: Mapping[str, object] | None,
        soft_budget: Mapping[str, object] | None,
    ) -> None:
        meter = BudgetMeter(scope_type="node", scope_id=node_id, config=hard_budget)
        if not meter.is_unbounded:
            self._node_meters[node_id] = meter

        soft_meter = BudgetMeter(
            scope_type="node_soft",
            scope_id=node_id,
            config=soft_budget,
            default_mode="soft",
            breach_action="warn",
        )
        if not soft_meter.is_unbounded:
            self._node_soft_meters[node_id] = soft_meter

    def register_loop(self, loop_id: str, budget: Mapping[str, object] | None) -> None:
        meter = BudgetMeter(
            scope_type="loop",
            scope_id=loop_id,
            config=budget,
            default_mode="hard",
            breach_action="stop",
        )
        if not meter.is_unbounded:
            self._loop_meters[loop_id] = meter

    # ------------------------------------------------------------------
    # Budget enforcement API
    # ------------------------------------------------------------------
    def preflight_node(
        self, node_id: str, cost: Mapping[str, object]
    ) -> BudgetPreflightResult:
        warnings: list[BudgetWarning] = []

        warnings.extend(self._evaluate_meter(self._run_meter, cost))
        warnings.extend(self._evaluate_meter(self._node_meters.get(node_id), cost))
        warnings.extend(self._evaluate_meter(self._node_soft_meters.get(node_id), cost))

        return BudgetPreflightResult(warnings=tuple(warnings))

    def commit_node(
        self, node_id: str, cost: Mapping[str, object]
    ) -> BudgetCommitResult:
        warnings: list[BudgetWarning] = []

        warnings.extend(self._charge_and_trace(self._run_meter, cost)[0])
        warnings.extend(self._charge_and_trace(self._node_meters.get(node_id), cost)[0])
        warnings.extend(
            self._charge_and_trace(self._node_soft_meters.get(node_id), cost)[0]
        )

        return BudgetCommitResult(warnings=tuple(warnings))

    def commit_loop_iteration(
        self, loop_id: str, cost: Mapping[str, object]
    ) -> LoopIterationOutcome:
        warnings: list[BudgetWarning] = []
        should_stop = False
        stop_reason: str | None = None

        warnings.extend(self._charge_and_trace(self._run_meter, cost)[0])

        loop_meter = self._loop_meters.get(loop_id)
        warnings_loop, result = self._charge_and_trace(loop_meter, cost)
        for warning in warnings_loop:
            if warning.severity == "hard":
                should_stop = True
                stop_reason = "budget_stop"
            warnings.append(warning)
        if result is not None and result.should_stop:
            should_stop = True
            stop_reason = "budget_stop"
        if (
            result is not None
            and result.meter is not None
            and result.meter.breach_action == "stop"
        ):
            remaining = result.remaining
            limits = result.meter.limits()
            for metric in ("usd", "tokens", "calls", "time_ms"):
                limit_value = getattr(limits, metric)
                remaining_value = getattr(remaining, metric)
                if limit_value is None or remaining_value is None:
                    continue
                if remaining_value <= 0:
                    should_stop = True
                    stop_reason = "budget_stop"
                    break

        return LoopIterationOutcome(
            should_stop=should_stop,
            stop_reason=stop_reason,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_meter(
        self, meter: BudgetMeter | None, cost: Mapping[str, object]
    ) -> Iterable[BudgetWarning]:
        if meter is None:
            return ()
        evaluation = meter.evaluate(cost)
        if evaluation.breach_kind == "hard":
            raise BudgetBreachHard(meter, evaluation)
        if evaluation.breach_kind == "soft":
            return (self._warning_from(meter, evaluation.metrics, "soft"),)
        return ()

    def _charge_and_trace(
        self, meter: BudgetMeter | None, cost: Mapping[str, object]
    ) -> tuple[list[BudgetWarning], BudgetChargeResult | None]:
        if meter is None:
            return [], None
        result = meter.charge(cost)
        self._emit_charge_events(result)
        warnings: list[BudgetWarning] = []
        if result.breached and result.breach_kind == "soft":
            warnings.append(self._warning_from(meter, result.metrics, "soft"))
        if result.breached and result.breach_kind == "hard" and result.should_stop:
            warnings.append(self._warning_from(meter, result.metrics, "hard"))
        return warnings, result

    def _warning_from(
        self, meter: BudgetMeter, metrics: tuple[str, ...], severity: str
    ) -> BudgetWarning:
        limits = meter.limits()
        spent = meter.spent_snapshot()
        parts = []
        for metric in metrics:
            limit_value = getattr(limits, metric)
            spent_value = getattr(spent, metric)
            parts.append(f"{metric} limit {limit_value} spent {spent_value}")
        message = (
            f"{meter.scope_type}:{meter.scope_id} {severity} budget breach for "
            + ", ".join(parts)
        )
        return BudgetWarning(
            scope_type=meter.scope_type,
            scope_id=meter.scope_id,
            metrics=metrics,
            severity=severity,
            message=message,
        )

    def _emit_charge_events(self, result: BudgetChargeResult) -> None:
        payload = {
            "cost": result.cost.as_dict(),
            "spent": result.spent.as_dict(),
            "remaining": result.remaining.as_dict(),
            "mode": result.meter.mode,
        }
        emit_trace_event(
            self._trace,
            self._event_sink,
            event="budget_charge",
            scope_type=result.meter.scope_type,
            scope_id=result.meter.scope_id,
            payload=payload,
        )
        if result.breached:
            breach_payload = {
                "metrics": result.metrics,
                "severity": result.breach_kind,
                "remaining": result.remaining.as_dict(),
                "action": result.breach_action,
            }
            if result.should_stop:
                breach_payload["stop_reason"] = "budget_stop"
            emit_trace_event(
                self._trace,
                self._event_sink,
                event="budget_breach",
                scope_type=result.meter.scope_type,
                scope_id=result.meter.scope_id,
                payload=breach_payload,
            )

