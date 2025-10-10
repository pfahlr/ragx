"""Minimal FlowRunner implementation with budget enforcement and tracing."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .budget import BudgetBreachHard, BudgetError, BudgetMeter
from .models import mapping_proxy

__all__ = ["FlowRunner", "RunResult"]


TraceEvent = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured result returned by :class:`FlowRunner.run`."""

    run_id: str
    status: str
    outputs: Mapping[str, Sequence[Mapping[str, Any]]]
    trace: Sequence[TraceEvent]
    stop_reasons: Sequence[Mapping[str, Any]]


class FlowRunner:
    """Execute flow control loops with budget enforcement."""

    def __init__(
        self,
        *,
        tool_adapters: Mapping[str, Callable[..., Mapping[str, Any]]],
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._tool_adapters = dict(tool_adapters)
        self._clock = clock or time.time
        self._trace_buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, spec: Mapping[str, Any], vars: Mapping[str, Any]) -> RunResult:
        run_id = uuid.uuid4().hex
        self._trace_buffer = []

        globals_section = spec.get("globals", {})
        run_budget_config = globals_section.get("run_budget") or {}
        run_meter = BudgetMeter(
            kind="run",
            subject=run_id,
            config=run_budget_config,
            mode=run_budget_config.get("mode"),
        )

        nodes_section = spec.get("graph", {}).get("nodes", [])
        nodes: dict[str, Mapping[str, Any]] = {}
        node_meters: dict[str, BudgetMeter] = {}
        for entry in nodes_section:
            node_id = entry["id"]
            nodes[node_id] = entry
            budget_cfg = entry.get("budget")
            if budget_cfg:
                node_meters[node_id] = BudgetMeter(
                    kind="node",
                    subject=node_id,
                    config=budget_cfg,
                    mode=budget_cfg.get("mode"),
                )

        outputs: dict[str, list[Mapping[str, Any]]] = {}
        stop_reasons: list[Mapping[str, Any]] = []

        for loop in spec.get("control", []):
            self._execute_loop(
                run_id=run_id,
                loop=loop,
                nodes=nodes,
                run_meter=run_meter,
                node_meters=node_meters,
                outputs=outputs,
                stop_reasons=stop_reasons,
            )

        result_outputs = {
            node_id: tuple(mapping_proxy(entry) for entry in data)
            for node_id, data in outputs.items()
        }

        return RunResult(
            run_id=run_id,
            status="ok",
            outputs=mapping_proxy(result_outputs),
            trace=tuple(mapping_proxy(event) for event in self._trace_buffer),
            stop_reasons=tuple(stop_reasons),
        )

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_loop(
        self,
        *,
        run_id: str,
        loop: Mapping[str, Any],
        nodes: Mapping[str, Mapping[str, Any]],
        run_meter: BudgetMeter,
        node_meters: Mapping[str, BudgetMeter],
        outputs: dict[str, list[Mapping[str, Any]]],
        stop_reasons: list[Mapping[str, Any]],
    ) -> None:
        loop_id = loop["id"]
        stop_config = loop.get("stop", {})
        max_iterations = stop_config.get("max_iterations")
        budget_config = stop_config.get("budget") or {}
        loop_meter: BudgetMeter | None = None
        if budget_config:
            loop_meter = BudgetMeter(
                kind="loop",
                subject=loop_id,
                config=budget_config,
                mode=budget_config.get("mode"),
                breach_action=budget_config.get("breach_action"),
            )

        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                stop_reasons.append(
                    mapping_proxy({"scope": "loop", "id": loop_id, "reason": "max_iterations"})
                )
                break

            self._emit(
                "loop_iter",
                run_id,
                {"loop_id": loop_id, "iteration": iteration},
            )

            for node_id in loop.get("target_subgraph", []):
                node = nodes.get(node_id)
                if node is None:
                    raise BudgetError(
                        f"Loop '{loop_id}' references unknown node '{node_id}'"
                    )
                self._execute_node(
                    run_id=run_id,
                    node=node,
                    iteration=iteration,
                    loop_id=loop_id,
                    run_meter=run_meter,
                    loop_meter=loop_meter,
                    node_meter=node_meters.get(node_id),
                    outputs=outputs,
                )

            iteration += 1
            if loop_meter:
                remaining = loop_meter.remaining()
                exhausted = any(
                    value is not None and value <= 0 for value in remaining.values()
                )
                breached = loop_meter.exceeded
            else:
                remaining = mapping_proxy({})
                exhausted = False
                breached = False

            if loop_meter and (breached or exhausted):
                event_payload = {
                    "meter": "loop",
                    "scope": "loop",
                    "loop_id": loop_id,
                    "iteration": iteration,
                    "action": loop_meter.breach_action,
                    "overages": loop_meter.overages(),
                    "remaining": remaining,
                    "breached": breached,
                    "exhausted": exhausted,
                }
                self._emit("budget_breach", run_id, event_payload)
                stop_reasons.append(
                    mapping_proxy({"scope": "loop", "id": loop_id, "reason": "budget"})
                )
                if loop_meter.breach_action == "stop":
                    break
                raise BudgetBreachHard(loop_meter.scope, loop_meter.overages())

    def _execute_node(
        self,
        *,
        run_id: str,
        node: Mapping[str, Any],
        iteration: int,
        loop_id: str,
        run_meter: BudgetMeter,
        loop_meter: BudgetMeter | None,
        node_meter: BudgetMeter | None,
        outputs: dict[str, list[Mapping[str, Any]]],
    ) -> None:
        node_id = node["id"]
        spec = node.get("spec", {})
        tool_ref = spec.get("tool_ref")
        if tool_ref is None:
            raise BudgetError(f"Node '{node_id}' missing spec.tool_ref")

        adapter = self._tool_adapters.get(tool_ref)
        if adapter is None:
            raise BudgetError(f"No adapter configured for tool '{tool_ref}'")

        payload = adapter(
            node=node,
            iteration=iteration,
            loop_id=loop_id,
        )
        node_outputs = payload.get("outputs", {})
        outputs.setdefault(node_id, []).append(node_outputs)
        cost_payload = payload.get("cost", {})

        meters: list[tuple[str, BudgetMeter | None]] = [
            ("run", run_meter),
            ("loop", loop_meter),
            ("node", node_meter),
        ]
        for meter_label, meter in meters:
            if meter is None:
                continue
            charge = meter.charge(cost_payload)
            self._emit(
                "budget_charge",
                run_id,
                {
                    "meter": meter_label,
                    "scope": meter.kind,
                    "subject": meter.subject,
                    "node_id": node_id,
                    "loop_id": loop_id,
                    "cost": charge.cost,
                    "remaining": charge.remaining,
                    "overages": charge.overages,
                    "breached": charge.breached,
                    "mode": charge.mode,
                },
            )

    # ------------------------------------------------------------------
    # Tracing helpers
    # ------------------------------------------------------------------
    def _emit(self, event: str, run_id: str, data: Mapping[str, Any]) -> None:
        record = {
            "event": event,
            "ts": self._clock(),
            "run_id": run_id,
            "data": mapping_proxy(dict(data)),
        }
        self._trace_buffer.append(record)
