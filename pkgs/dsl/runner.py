"""Minimal FlowRunner implementation focused on budget enforcement."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .budget import BudgetDecision, BudgetExceededError, BudgetMeter, Cost

__all__ = [
    "NodeExecution",
    "RunResult",
    "FlowRunner",
]


@dataclass(frozen=True, slots=True)
class NodeExecution:
    """Result from executing a single node."""

    node_id: str
    outputs: Mapping[str, object]
    cost: Cost


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured outcome from :meth:`FlowRunner.run`."""

    run_id: str
    status: str
    outputs: Mapping[str, Mapping[str, object]]
    trace: Sequence[Mapping[str, object]]


class FlowRunner:
    """Execute flow specs with budget enforcement hooks."""

    def __init__(
        self,
        *,
        id_factory: Callable[[], uuid.UUID] | None = None,
        now_factory: Callable[[], float] | None = None,
    ) -> None:
        self._id_factory = id_factory or uuid.uuid4
        self._now_factory = now_factory or time.time
        self.policy_stack = None
        self.budget_meter: BudgetMeter | None = None
        self.adapters: dict[str, object] = {}
        self._trace: list[dict[str, object]] = []
        self._node_meters: dict[str, BudgetMeter] = {}
        self._soft_node_meters: dict[str, BudgetMeter] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, spec: Mapping[str, object], vars: Mapping[str, object]) -> RunResult:
        run_id = str(self._id_factory())
        self._trace = []
        context_outputs: dict[str, Mapping[str, object]] = {}
        status = "ok"

        self._trace_event(
            "run_start",
            {
                "run_id": run_id,
                "ts": self._now_factory(),
            },
        )

        globals_cfg = self._as_mapping(spec.get("globals"))
        run_budget_cfg = self._as_mapping(globals_cfg.get("run_budget"))
        run_budget = run_budget_cfg if run_budget_cfg else None
        self.budget_meter = BudgetMeter.from_budget(run_budget, scope="run")

        try:
            graph = self._as_mapping(spec.get("graph"))
            nodes = self._index_nodes(graph.get("nodes"))
            control_nodes = tuple(self._iter_mappings(graph.get("control")))
            for loop in control_nodes:
                if loop.get("kind") != "loop":
                    continue
                self._run_loop(loop, nodes, context_outputs)
        except BudgetExceededError as exc:
            status = "error"
            self._trace_event(
                "run_error",
                {
                    "run_id": run_id,
                    "reason": "budget_exceeded",
                    "details": exc.decision.as_dict(),
                },
            )
        finally:
            self._trace_event(
                "run_end",
                {
                    "run_id": run_id,
                    "status": status,
                    "ts": self._now_factory(),
                },
            )

        return RunResult(
            run_id=run_id,
            status=status,
            outputs=context_outputs,
            trace=tuple(self._trace),
        )

    # ------------------------------------------------------------------
    # Loop execution
    # ------------------------------------------------------------------
    def _run_loop(
        self,
        loop_spec: Mapping[str, object],
        nodes: Mapping[str, Mapping[str, object]],
        context_outputs: dict[str, Mapping[str, object]],
    ) -> None:
        loop_mapping = self._as_mapping(loop_spec)
        loop_id = str(loop_mapping.get("id", "loop"))
        stop_config = self._as_mapping(loop_mapping.get("stop"))
        max_iterations = self._coerce_max_iterations(stop_config.get("max_iterations"))

        budget_cfg = self._as_mapping(stop_config.get("budget"))
        loop_meter = BudgetMeter.from_budget(
            budget_cfg if budget_cfg else None,
            scope=f"loop:{loop_id}",
        )
        breach_action = "error"
        action_value = budget_cfg.get("breach_action") if budget_cfg else None
        if isinstance(action_value, str):
            breach_action = action_value.lower()

        target_nodes = tuple(self._iter_strings(loop_mapping.get("target_subgraph")))

        self._trace_event(
            "loop_start",
            {
                "loop_id": loop_id,
                "target": target_nodes,
            },
        )

        iteration = 0
        while iteration < max_iterations:
            if loop_meter:
                hint = self._iteration_cost_hint(loop_spec, iteration)
                if not loop_meter.can_spend(hint):
                    decision = loop_meter.last_decision
                    if decision is None:
                        decision = BudgetDecision(
                            scope=loop_meter.scope,
                            allowed=False,
                            breached=(),
                            soft_breach=False,
                            remaining=loop_meter.remaining,
                            spent=loop_meter.spent,
                        )
                    if breach_action == "stop":
                        self._emit_loop_stop(loop_id, "budget_stop", decision, iteration)
                        return
                    raise BudgetExceededError(decision)

            self._trace_event(
                "loop_iter",
                {
                    "loop_id": loop_id,
                    "iteration": iteration,
                },
            )

            stop_reason: str | None = None
            for node_id_str in target_nodes:
                if node_id_str not in nodes:
                    continue
                node = nodes[node_id_str]
                execution = self._execute_node(
                    node,
                    context_outputs,
                    loop_id=loop_id,
                )
                stop_reason = self._apply_costs(
                    node=node,
                    cost=execution.cost,
                    loop_id=loop_id,
                    loop_meter=loop_meter,
                    loop_breach_action=breach_action,
                )
                context_outputs[node_id_str] = execution.outputs
                if stop_reason:
                    break

            if stop_reason is not None:
                self._emit_loop_stop(loop_id, stop_reason, loop_meter.last_decision, iteration)
                return

            iteration += 1

        self._emit_loop_stop(loop_id, "max_iterations", loop_meter.last_decision, iteration)

    def _emit_loop_stop(
        self,
        loop_id: str,
        reason: str,
        decision: BudgetDecision | None,
        iteration: int,
    ) -> None:
        payload: dict[str, Any] = {
            "loop_id": loop_id,
            "reason": reason,
            "iteration": iteration,
        }
        if decision is not None:
            payload["details"] = decision.as_dict()
        self._trace_event("loop_stop", payload)

    # ------------------------------------------------------------------
    # Cost enforcement
    # ------------------------------------------------------------------
    def _apply_costs(
        self,
        *,
        node: Mapping[str, object],
        cost: Cost,
        loop_id: str,
        loop_meter: BudgetMeter | None,
        loop_breach_action: str,
    ) -> str | None:
        node_id = str(node.get("id"))
        stop_reason: str | None = None

        if self.budget_meter is not None:
            self._charge_and_trace(
                meter=self.budget_meter,
                cost=cost,
                scope="run",
                context={"node_id": node_id, "loop_id": loop_id},
            )

        if loop_meter is not None:
            stop_reason = self._charge_and_trace(
                meter=loop_meter,
                cost=cost,
                scope=f"loop:{loop_id}",
                context={"node_id": node_id, "loop_id": loop_id},
                breach_action=loop_breach_action,
            )
            if stop_reason is not None:
                return stop_reason

        node_meter = self._node_meters.get(node_id)
        node_budget_cfg = self._as_mapping(node.get("budget"))
        if node_meter is None and node_budget_cfg:
            node_meter = BudgetMeter.from_budget(node_budget_cfg, scope=f"node:{node_id}")
            self._node_meters[node_id] = node_meter
        if node_meter is not None:
            self._charge_and_trace(
                meter=node_meter,
                cost=cost,
                scope=f"node:{node_id}",
                context={"loop_id": loop_id},
            )

        spec_cfg = self._as_mapping(node.get("spec"))
        soft_cfg = self._as_mapping(spec_cfg.get("budget"))
        if soft_cfg:
            soft_meter = self._soft_node_meters.get(node_id)
            if soft_meter is None:
                soft_budget_payload = dict(soft_cfg)
                soft_budget_payload.setdefault("mode", "soft")
                soft_meter = BudgetMeter.from_budget(
                    soft_budget_payload,
                    scope=f"node_soft:{node_id}",
                )
                self._soft_node_meters[node_id] = soft_meter
            self._charge_and_trace(
                meter=soft_meter,
                cost=cost,
                scope=f"node_soft:{node_id}",
                context={"loop_id": loop_id},
                allow_soft=True,
            )

        return None

    def _charge_and_trace(
        self,
        *,
        meter: BudgetMeter,
        cost: Cost,
        scope: str,
        context: Mapping[str, object],
        breach_action: str | None = None,
        allow_soft: bool = False,
    ) -> str | None:
        try:
            decision = meter.charge(cost)
        except BudgetExceededError as exc:
            decision = exc.decision
            self._trace_event(
                "budget_charge",
                {
                    "scope": scope,
                    "cost": cost.as_dict(),
                    "allowed": False,
                    "decision": decision.as_dict(),
                    "context": dict(context),
                },
            )
            if breach_action == "stop":
                return "budget_stop"
            if allow_soft:
                return None
            raise
        else:
            self._trace_event(
                "budget_charge",
                {
                    "scope": scope,
                    "cost": cost.as_dict(),
                    "allowed": decision.allowed,
                    "decision": decision.as_dict(),
                    "context": dict(context),
                },
            )
            return None

    # ------------------------------------------------------------------
    # Helpers / overridables
    # ------------------------------------------------------------------
    def _execute_node(
        self,
        node: Mapping[str, object],
        context: Mapping[str, Mapping[str, object]],
        *,
        loop_id: str | None = None,
    ) -> NodeExecution:
        node_id = str(node.get("id", "node"))
        return NodeExecution(node_id=node_id, outputs={}, cost=Cost())

    def _iteration_cost_hint(
        self, loop: Mapping[str, object], iteration_index: int
    ) -> Cost:
        return Cost()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _trace_event(self, event: str, payload: Mapping[str, object]) -> None:
        data = dict(payload)
        data["event"] = event
        self._trace.append(data)

    @staticmethod
    def _as_mapping(value: object) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            return value
        return {}

    def _index_nodes(self, nodes: object) -> Mapping[str, Mapping[str, object]]:
        index: dict[str, Mapping[str, object]] = {}
        for node in self._iter_mappings(nodes):
            identifier = str(node.get("id"))
            index[identifier] = node
        return index

    @staticmethod
    def _iter_mappings(value: object) -> Iterable[Mapping[str, object]]:
        if isinstance(value, Mapping):
            yield value
        elif isinstance(value, Iterable) and not isinstance(value, str | bytes):
            for item in value:
                if isinstance(item, Mapping):
                    yield item

    @staticmethod
    def _iter_strings(value: object) -> Iterable[str]:
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, Iterable) and not isinstance(value, str | bytes):
            for item in value:
                yield str(item)

    @staticmethod
    def _coerce_max_iterations(value: object) -> float:
        if isinstance(value, int | float):
            numeric = float(value)
            if numeric > 0:
                return numeric
        return math.inf
