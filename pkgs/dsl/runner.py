from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from .budget import (
    BudgetCharge,
    BudgetCheck,
    BudgetExceededError,
    BudgetMeter,
    BudgetMode,
    Cost,
)

__all__ = [
    "FlowRunner",
    "RunResult",
    "LoopSummary",
    "LoopIterationContext",
    "LoopIterationResult",
]


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: str
    outputs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LoopSummary:
    loop_id: str
    iterations: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class LoopIterationContext:
    run_id: str
    loop_id: str
    iteration: int
    run_meter: BudgetMeter
    loop_meter: BudgetMeter | None


@dataclass(frozen=True, slots=True)
class LoopIterationResult:
    cost: Cost = field(default_factory=Cost)
    should_stop: bool = False
    outputs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunSummary:
    status: str
    loop_summaries: Sequence[LoopSummary]


class FlowRunner:
    def __init__(
        self,
        *,
        cache_mode: str = "readwrite",
        max_concurrency: int = 4,
        fail_fast: bool = False,
        trace_sink: Callable[[Mapping[str, object]], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.cache_mode = cache_mode
        self.max_concurrency = max_concurrency
        self.fail_fast = fail_fast
        self._trace_sink = trace_sink
        self._clock = clock or time.time
        self._trace_events: list[dict[str, object]] = []
        self.last_error: Exception | None = None
        self.last_run: RunSummary | None = None
        self.budget_meter: BudgetMeter | None = None
        self._current_run_id: str | None = None

    @property
    def trace_events(self) -> Sequence[Mapping[str, object]]:
        return tuple(self._trace_events)

    def run(self, spec: Mapping[str, object], vars: Mapping[str, object]) -> RunResult:
        run_id = uuid.uuid4().hex
        self._current_run_id = run_id
        self._trace_events = []
        self.last_error = None
        self.last_run = None

        self._emit_trace(
            "run_start", {"vars_present": cast(object, bool(vars))}
        )

        globals_section = cast(Mapping[str, object], spec.get("globals") or {})
        run_budget_spec = cast(Mapping[str, object] | None, globals_section.get("run_budget"))
        self.budget_meter = BudgetMeter.from_spec(
            run_budget_spec,
            scope="run",
            label="run",
            default_mode=BudgetMode.HARD,
        )

        loop_summaries: list[LoopSummary] = []
        outputs: dict[str, object] = {}

        try:
            graph_section = cast(Mapping[str, object], spec.get("graph") or {})
            control_nodes_obj = graph_section.get("control") or []
            control_nodes = cast(Sequence[Mapping[str, object]], control_nodes_obj)
            for loop_spec in control_nodes:
                if loop_spec.get("kind") != "loop":
                    continue
                summary = self._execute_loop(run_id, loop_spec, self.budget_meter)
                loop_summaries.append(summary)
        except BudgetExceededError as exc:
            self.last_error = exc
            self.last_run = RunSummary(status="error", loop_summaries=tuple(loop_summaries))
            self._emit_trace("run_end", {"status": "error"})
            return RunResult(run_id=run_id, status="error", outputs=outputs)

        self.last_run = RunSummary(status="ok", loop_summaries=tuple(loop_summaries))
        self._emit_trace("run_end", {"status": "ok"})
        return RunResult(run_id=run_id, status="ok", outputs=outputs)

    # ------------------------------------------------------------------
    # Loop execution helpers
    # ------------------------------------------------------------------
    def _execute_loop(
        self,
        run_id: str,
        loop_spec: Mapping[str, object],
        run_meter: BudgetMeter,
    ) -> LoopSummary:
        loop_id = str(loop_spec.get("id", "loop"))
        stop_spec = cast(Mapping[str, object], loop_spec.get("stop") or {})
        max_iterations = cast(int | None, stop_spec.get("max_iterations"))
        loop_budget_spec = cast(Mapping[str, object] | None, stop_spec.get("budget"))
        loop_breach_action = (
            cast(str | None, loop_budget_spec.get("breach_action"))
            if loop_budget_spec
            else None
        )
        loop_meter = (
            BudgetMeter.from_spec(
                loop_budget_spec,
                scope="loop",
                label=loop_id,
                default_mode=BudgetMode.HARD,
                breach_action=loop_breach_action,
            )
            if loop_budget_spec
            else None
        )

        iterations = 0
        stop_reason = "completed"
        while True:
            if max_iterations is not None and iterations >= int(max_iterations):
                stop_reason = "iteration_cap"
                break

            context = LoopIterationContext(
                run_id=run_id,
                loop_id=loop_id,
                iteration=iterations,
                run_meter=run_meter,
                loop_meter=loop_meter,
            )

            estimate = self._estimate_loop_iteration_cost(loop_spec, iterations, context)

            if loop_meter:
                preview = loop_meter.preview(estimate)
                if not preview.allowed:
                    action = "stop" if loop_meter.breach_action == "stop" else "error"
                    self._emit_budget_breach("loop", loop_id, preview, action=action)
                    if action == "stop":
                        stop_reason = "budget_stop"
                        break
                    raise self._error_from_preview(loop_meter, preview)
                if preview.breach_kind == "soft":
                    self._emit_budget_breach("loop", loop_id, preview, action="warn")

            run_preview = run_meter.preview(estimate)
            if not run_preview.allowed:
                self._emit_budget_breach("run", loop_id, run_preview, action="error")
                raise self._error_from_preview(run_meter, run_preview)
            if run_preview.breach_kind == "soft":
                self._emit_budget_breach("run", loop_id, run_preview, action="warn")

            self._emit_trace(
                "loop_iter",
                {
                    "loop_id": loop_id,
                    "iteration": iterations,
                    "estimate": cast(object, estimate.to_dict()),
                },
            )

            result = self._run_loop_iteration(loop_spec, iterations, context)
            actual_cost = result.cost

            try:
                run_charge = run_meter.charge(actual_cost)
            except BudgetExceededError:
                preview = run_meter.preview(actual_cost)
                self._emit_budget_breach("run", loop_id, preview, action="error")
                raise
            else:
                self._emit_budget_charge("run", loop_id, actual_cost, run_charge)
                if run_charge.breached:
                    self._emit_budget_breach(
                        "run",
                        loop_id,
                        BudgetCheck(
                            allowed=True,
                            breach_kind=run_charge.breach_kind,
                            metric=run_charge.metric,
                            limit=run_charge.limit,
                            attempted=run_charge.attempted,
                            remaining=run_charge.remaining,
                        ),
                        action="warn",
                    )

            if loop_meter is not None:
                try:
                    loop_charge = loop_meter.charge(actual_cost)
                except BudgetExceededError:
                    preview = loop_meter.preview(actual_cost)
                    action = "stop" if loop_meter.breach_action == "stop" else "error"
                    self._emit_budget_breach("loop", loop_id, preview, action=action)
                    if action == "stop":
                        stop_reason = "budget_stop"
                        break
                    raise
                else:
                    self._emit_budget_charge("loop", loop_id, actual_cost, loop_charge)
                    if loop_charge.breached:
                        self._emit_budget_breach(
                            "loop",
                            loop_id,
                            BudgetCheck(
                                allowed=True,
                                breach_kind=loop_charge.breach_kind,
                                metric=loop_charge.metric,
                                limit=loop_charge.limit,
                                attempted=loop_charge.attempted,
                                remaining=loop_charge.remaining,
                            ),
                            action="warn",
                        )

            iterations += 1
            if result.should_stop:
                stop_reason = "body_stop"
                break

        return LoopSummary(loop_id=loop_id, iterations=iterations, stop_reason=stop_reason)

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------
    def _estimate_loop_iteration_cost(
        self,
        loop_spec: Mapping[str, object],
        iteration: int,
        context: LoopIterationContext,
    ) -> Cost:
        return Cost()

    def _run_loop_iteration(
        self,
        loop_spec: Mapping[str, object],
        iteration: int,
        context: LoopIterationContext,
    ) -> LoopIterationResult:
        return LoopIterationResult(should_stop=True)

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------
    def _emit_trace(self, event: str, payload: Mapping[str, object]) -> None:
        record: dict[str, object] = {
            "event": event,
            "run_id": self._current_run_id or "",
            "ts": self._clock(),
        }
        record.update(payload)
        self._trace_events.append(record)
        if self._trace_sink is not None:
            self._trace_sink(record)

    def _emit_budget_charge(
        self,
        scope: str,
        loop_id: str,
        cost: Cost,
        charge: BudgetCharge,
    ) -> None:
        event: dict[str, object] = {
            "event": "budget_charge",
            "run_id": self._current_run_id or "",
            "scope": scope,
            "loop_id": loop_id,
            "ts": self._clock(),
            "cost": cast(object, cost.to_dict()),
            "remaining": cast(object, charge.remaining.to_dict()),
        }
        if charge.metric:
            event["metric"] = charge.metric
        self._trace_events.append(event)
        if self._trace_sink is not None:
            self._trace_sink(event)

    def _emit_budget_breach(
        self,
        scope: str,
        loop_id: str,
        check: BudgetCheck,
        *,
        action: str,
    ) -> None:
        event: dict[str, object] = {
            "event": "budget_breach",
            "run_id": self._current_run_id or "",
            "scope": scope,
            "loop_id": loop_id,
            "ts": self._clock(),
            "action": action,
            "breach_kind": check.breach_kind,
            "metric": check.metric,
            "remaining": cast(object, check.remaining.to_dict()),
        }
        self._trace_events.append(event)
        if self._trace_sink is not None:
            self._trace_sink(event)

    def _error_from_preview(self, meter: BudgetMeter, check: BudgetCheck) -> BudgetExceededError:
        metric = check.metric or "usd"
        limit = check.limit or 0.0
        attempted = check.attempted or limit
        spent = meter.spent
        if metric == "tokens":
            spent_value = float(spent.tokens)
        elif metric == "calls":
            spent_value = float(spent.calls)
        elif metric == "time_sec":
            spent_value = float(spent.time_sec)
        else:
            spent_value = float(spent.usd)
        return BudgetExceededError(
            scope=meter.label,
            metric=metric,
            limit=limit,
            spent=spent_value,
            attempted=attempted,
            mode=meter.mode,
        )
