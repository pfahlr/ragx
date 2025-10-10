"""Minimal FlowRunner implementation with budget integration."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from .budget import (
    BudgetBreachHard,
    BudgetChargeOutcome,
    BudgetMeter,
    BudgetRemaining,
    BudgetSpec,
    CostSnapshot,
)
from .trace import InMemoryTraceWriter, TraceWriter

__all__ = ["FlowRunner", "RunResult"]


@dataclass(slots=True, frozen=True)
class RunResult:
    """Structured result returned by :meth:`FlowRunner.run`."""

    run_id: str
    status: str
    outputs: Mapping[str, Any]
    trace: Sequence[Mapping[str, object]]
    loop_iterations: Mapping[str, int]
    loop_stop_reasons: Mapping[str, str]


class FlowRunner:
    """Execute DSL flows with budget enforcement for loops and nodes."""

    def __init__(
        self,
        *,
        trace_writer: TraceWriter | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._trace = trace_writer or InMemoryTraceWriter()
        self._clock = clock or time.perf_counter
        self._reported_breaches: set[str] = set()

    def run(self, spec: Mapping[str, Any], vars: Mapping[str, Any] | None = None) -> RunResult:
        run_id = str(uuid.uuid4())
        self._reported_breaches.clear()
        self._trace.emit("run_start", {"run_id": run_id})
        run_meter = self._build_run_meter(spec)
        node_map = {node["id"]: node for node in spec.get("graph", {}).get("nodes", [])}
        loop_iterations: MutableMapping[str, int] = {}
        loop_stop_reasons: MutableMapping[str, str] = {}
        outputs: MutableMapping[str, Any] = {}
        status = "ok"
        try:
            for loop in spec.get("graph", {}).get("control", []):
                loop_id = loop.get("id", "loop")
                loop_meter = self._build_loop_meter(loop)
                loop_stop_reasons[loop_id] = "completed"
                iterations = 0
                max_iterations = self._coerce_int(loop.get("stop", {}).get("max_iterations"))
                while True:
                    if max_iterations is not None and iterations >= max_iterations:
                        loop_stop_reasons[loop_id] = "max_iterations"
                        break
                    loop_break = False
                    for node_id in loop.get("target_subgraph", []):
                        node = node_map[node_id]
                        self._trace.emit("node_start", {"run_id": run_id, "node_id": node_id})
                        node_meter = self._build_node_meter(node)
                        cost = self._execute_node(node)
                        try:
                            self._apply_cost(
                                run_id,
                                node_id,
                                cost,
                                run_meter=run_meter,
                                node_meter=node_meter,
                                loop_meter=loop_meter,
                                loop_id=loop_id,
                            )
                        except BudgetBreachHard:
                            status = "error"
                            loop_stop_reasons[loop_id] = "budget_breach"
                            self._trace.emit(
                                "node_end",
                                {
                                    "run_id": run_id,
                                    "node_id": node_id,
                                    "status": "error",
                                    "cost": self._cost_dict(cost),
                                },
                            )
                            loop_break = True
                            break
                        else:
                            self._trace.emit(
                                "node_end",
                                {
                                    "run_id": run_id,
                                    "node_id": node_id,
                                    "status": "ok",
                                    "cost": self._cost_dict(cost),
                                },
                            )
                    iterations += 1
                    loop_iterations[loop_id] = iterations
                    self._trace.emit(
                        "loop_iter",
                        {"run_id": run_id, "loop_id": loop_id, "iter": iterations},
                    )
                    if loop_break:
                        break
                    if loop_meter and loop_meter.exceeded:
                        self._emit_budget_breach(run_id, "loop", loop_id, loop_meter, "hard")
                        loop_stop_reasons[loop_id] = "budget_breach"
                        if loop_meter.stop_behavior == "stop":
                            break
                        raise BudgetBreachHard(loop_id, limit="loop", amount=iterations)
                loop_iterations.setdefault(loop_id, iterations)
        finally:
            self._trace.emit("run_end", {"run_id": run_id, "status": status})
        return RunResult(
            run_id=run_id,
            status=status,
            outputs=MappingProxyType(dict(outputs)),
            trace=self._trace.snapshot(),
            loop_iterations=MappingProxyType(dict(loop_iterations)),
            loop_stop_reasons=MappingProxyType(dict(loop_stop_reasons)),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_run_meter(self, spec: Mapping[str, Any]) -> BudgetMeter | None:
        run_budget = spec.get("globals", {}).get("run_budget")
        if not run_budget:
            return None
        return BudgetMeter.from_spec(
            BudgetSpec.from_mapping(run_budget, scope="run"),
            scope="run",
            clock=self._clock,
        )

    def _build_node_meter(self, node: Mapping[str, Any]) -> BudgetMeter | None:
        budget = node.get("budget")
        if not budget:
            return None
        return BudgetMeter.from_spec(
            BudgetSpec.from_mapping(budget, scope="node"),
            scope="node",
            clock=self._clock,
        )

    def _build_loop_meter(self, loop: Mapping[str, Any]) -> BudgetMeter | None:
        stop = loop.get("stop", {})
        budget = stop.get("budget")
        if not budget:
            return None
        spec = BudgetSpec.from_mapping(budget, scope="loop")
        return BudgetMeter.from_spec(spec, scope="loop", clock=self._clock)

    def _apply_cost(
        self,
        run_id: str,
        node_id: str,
        cost: CostSnapshot,
        *,
        run_meter: BudgetMeter | None,
        node_meter: BudgetMeter | None,
        loop_meter: BudgetMeter | None,
        loop_id: str | None,
    ) -> None:
        if run_meter is not None:
            self._charge_meter(run_id, "run", "run", cost, run_meter)
        if node_meter is not None:
            self._charge_meter(run_id, "node", node_id, cost, node_meter)
        if loop_meter is not None and loop_id is not None:
            self._charge_meter(run_id, "loop", loop_id, cost, loop_meter)

    def _charge_meter(
        self,
        run_id: str,
        scope: str,
        identifier: str,
        cost: CostSnapshot,
        meter: BudgetMeter,
    ) -> BudgetChargeOutcome:
        outcome = meter.charge(cost)
        self._trace.emit(
            "budget_charge",
            {
                "run_id": run_id,
                "scope": scope,
                "meter_id": identifier,
                "cost": self._cost_dict(cost),
                "remaining": self._remaining_dict(outcome.remaining),
            },
        )
        key = self._meter_key(scope, identifier)
        if outcome.breach_kind and key not in self._reported_breaches:
            self._emit_budget_breach(run_id, scope, identifier, meter, outcome.breach_kind)
        elif meter.exceeded and key not in self._reported_breaches:
            kind = "hard" if meter.mode == "hard" else "soft"
            self._emit_budget_breach(run_id, scope, identifier, meter, kind)
        return outcome

    def _emit_budget_breach(
        self,
        run_id: str,
        scope: str,
        identifier: str,
        meter: BudgetMeter,
        kind: str,
    ) -> None:
        key = self._meter_key(scope, identifier)
        if key in self._reported_breaches:
            return
        self._reported_breaches.add(key)
        self._trace.emit(
            "budget_breach",
            {
                "run_id": run_id,
                "scope": scope,
                "meter_id": identifier,
                "breach_kind": kind,
                "mode": meter.mode,
                "stop_behavior": meter.stop_behavior,
            },
        )

    @staticmethod
    def _meter_key(scope: str, identifier: str) -> str:
        return f"{scope}:{identifier}"

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    @staticmethod
    def _cost_dict(cost: CostSnapshot) -> Mapping[str, object]:
        return MappingProxyType(asdict(cost))

    @staticmethod
    def _remaining_dict(remaining: BudgetRemaining) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "usd": remaining.usd,
                "calls": remaining.calls,
                "tokens": remaining.tokens,
                "seconds": remaining.seconds,
            }
        )

    @staticmethod
    def _execute_node(node: Mapping[str, Any]) -> CostSnapshot:
        spec = node.get("spec", {})
        mock_cost = spec.get("mock_cost", {})
        usd = FlowRunner._coerce_float(mock_cost.get("usd")) or 0.0
        calls = FlowRunner._coerce_int(mock_cost.get("calls")) or 0
        tokens_in = FlowRunner._coerce_int(mock_cost.get("tokens_in")) or 0
        tokens_out = FlowRunner._coerce_int(mock_cost.get("tokens_out")) or 0
        seconds = FlowRunner._coerce_float(mock_cost.get("seconds")) or 0.0
        return CostSnapshot(
            usd=usd,
            calls=calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            seconds=seconds,
        )

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, int | float):
            return float(value)
        return None
