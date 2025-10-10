from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import BudgetBreachError, BudgetMeter

__all__ = ["RunResult", "FlowRunner"]


@dataclass(slots=True)
class RunResult:
    """Minimal run result returned by :class:`FlowRunner`."""

    run_id: str
    status: str
    outputs: dict[str, dict[str, Any]]
    stop_reasons: list[dict[str, Any]] = field(default_factory=list)


class FlowRunner:
    """Execute a simplified flow spec with budget enforcement."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, Callable[..., Mapping[str, Any]]] | None = None,
        trace_path: Path | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._adapters = dict(adapters or {})
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._trace_events: list[dict[str, Any]] = []
        self._outputs: dict[str, dict[str, Any]] = {}
        self._stop_reasons: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def trace_events(self) -> list[dict[str, Any]]:
        return list(self._trace_events)

    def run(self, *, spec: Mapping[str, Any], vars: Mapping[str, Any]) -> RunResult:
        self._trace_events.clear()
        self._outputs = {}
        self._stop_reasons = []
        run_id = self._run_id_factory()
        self._record_event("run_start", scope="run", run_id=run_id)

        globals_cfg = spec.get("globals", {}) if isinstance(spec, Mapping) else {}
        run_meter = self._build_meter(name="run", scope="run", config=globals_cfg.get("run_budget"))

        node_specs = {node["id"]: node for node in spec.get("nodes", [])}
        node_meters = {
            node_id: self._build_meter(name=node_id, scope="node", config=node.get("budget"))
            for node_id, node in node_specs.items()
        }
        spec_meters = {
            node_id: self._build_meter(
                name=node_id,
                scope="spec",
                config=node.get("spec", {}).get("budget"),
            )
            for node_id, node in node_specs.items()
        }

        status = "ok"
        for control in spec.get("control", []):
            if control.get("kind") != "loop":
                continue
            loop_status = self._execute_loop(
                run_id=run_id,
                loop=control,
                node_specs=node_specs,
                node_meters=node_meters,
                spec_meters=spec_meters,
                run_meter=run_meter,
                vars=vars,
            )
            if loop_status == "halted":
                status = "halted"
                break

        self._flush_trace()
        return RunResult(
            run_id=run_id,
            status=status,
            outputs=self._outputs,
            stop_reasons=list(self._stop_reasons),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_loop(
        self,
        *,
        run_id: str,
        loop: Mapping[str, Any],
        node_specs: Mapping[str, Mapping[str, Any]],
        node_meters: Mapping[str, BudgetMeter | None],
        spec_meters: Mapping[str, BudgetMeter | None],
        run_meter: BudgetMeter | None,
        vars: Mapping[str, Any],
    ) -> str | None:
        loop_id = loop.get("id", "loop")
        stop_cfg = loop.get("stop", {})
        max_iterations = stop_cfg.get("max_iterations")
        loop_meter = self._build_meter(name=loop_id, scope="loop", config=stop_cfg.get("budget"))
        breach_action = (stop_cfg.get("budget", {}) or {}).get("breach_action", "stop")
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            for node_id in loop.get("target_subgraph", []):
                node = node_specs.get(node_id)
                if node is None:
                    raise KeyError(f"Unknown node id '{node_id}' referenced in loop '{loop_id}'")
                self._execute_node(
                    run_id=run_id,
                    node=node,
                    loop_id=loop_id,
                    iteration=iteration,
                    run_meter=run_meter,
                    loop_meter=loop_meter,
                    node_meter=node_meters.get(node_id),
                    spec_meter=spec_meters.get(node_id),
                    vars=vars,
                )
            iteration += 1
            if loop_meter and loop_meter.is_exhausted:
                metric, limit = self._detect_breach(loop_meter)
                self._record_event(
                    "budget_breach",
                    scope=loop_meter.full_scope,
                    action=breach_action,
                    details={"metric": metric, "limit": limit},
                )
                if breach_action == "stop":
                    self._stop_reasons.append(
                        {
                            "scope": loop_meter.full_scope,
                            "reason": "budget_exhausted",
                            "details": {"metric": metric, "limit": limit},
                        }
                    )
                    return "halted"
                raise RuntimeError("loop budget hard cap exceeded")
            if max_iterations is not None and iteration >= max_iterations:
                break
        return None

    def _execute_node(
        self,
        *,
        run_id: str,
        node: Mapping[str, Any],
        loop_id: str,
        iteration: int,
        run_meter: BudgetMeter | None,
        loop_meter: BudgetMeter | None,
        node_meter: BudgetMeter | None,
        spec_meter: BudgetMeter | None,
        vars: Mapping[str, Any],
    ) -> None:
        node_id = node.get("id", "node")
        if node.get("kind") != "unit":
            raise NotImplementedError("Only unit nodes are supported in this simplified runner")
        spec = node.get("spec", {})
        tool_ref = spec.get("tool_ref")
        if tool_ref not in self._adapters:
            raise KeyError(f"No adapter registered for tool '{tool_ref}'")
        adapter = self._adapters[tool_ref]
        inputs = node.get("inputs", {})
        context = {
            "run_id": run_id,
            "loop_id": loop_id,
            "iteration": iteration,
            "vars": vars,
        }
        result = adapter(inputs=inputs, context=context)
        if not isinstance(result, Mapping):
            raise TypeError("Adapter must return a mapping with outputs and cost")
        outputs = dict(result.get("outputs", {}))
        cost = dict(result.get("cost", {}))
        self._outputs[node_id] = outputs

        self._charge_all(cost, [run_meter, loop_meter, node_meter, spec_meter])

        self._record_event(
            "node_end",
            scope=f"node:{node_id}",
            loop_id=loop_id,
            iteration=iteration,
            outputs=outputs,
        )

    def _charge_all(self, cost: Mapping[str, Any], meters: list[BudgetMeter | None]) -> None:
        for meter in meters:
            if meter is None:
                continue
            try:
                result = meter.charge(cost)
            except BudgetBreachError as err:
                self._record_event(
                    "budget_breach",
                    scope=err.scope,
                    action="error",
                    details={
                        "metric": err.metric,
                        "limit": err.limit,
                        "attempted": err.attempted,
                    },
                )
                raise
            else:
                self._record_event(
                    "budget_charge",
                    scope=result.scope,
                    cost=dict(result.cost),
                    spent=dict(result.spent),
                )
                if result.warning is not None:
                    self._record_event(
                        "budget_warn",
                        scope=result.warning.scope,
                        over=dict(result.warning.over),
                        mode=result.warning.mode,
                    )

    def _detect_breach(self, meter: BudgetMeter) -> tuple[str, float | None]:
        spent = meter.spent
        for metric, limit in meter.limits.items():
            if limit is None:
                continue
            if spent.get(metric, 0.0) >= limit - 1e-9:
                return metric, limit
        return "unknown", None

    def _build_meter(
        self, *, name: str, scope: str, config: Mapping[str, Any] | None
    ) -> BudgetMeter | None:
        if not config:
            return None
        return BudgetMeter(name=name, scope=scope, config=config)

    def _record_event(self, event: str, *, scope: str, **payload: Any) -> None:
        record = {"event": event, "scope": scope, **payload}
        self._trace_events.append(record)

    def _flush_trace(self) -> None:
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("w", encoding="utf-8") as fh:
            for event in self._trace_events:
                fh.write(json.dumps(event) + "\n")
