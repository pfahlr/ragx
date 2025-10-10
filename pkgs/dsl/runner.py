"""Minimal FlowRunner implementation with budget integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

from .budget import BudgetBreach, BudgetCharge, BudgetExceededError, BudgetMeter


class ToolAdapter(Protocol):
    """Adapter interface used by the runner tests."""

    def estimate_cost(
        self,
        node: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> Mapping[str, float | int]:
        ...

    def execute(self, node: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class RunResult:
    """Execution result returned by :class:`FlowRunner`."""

    run_id: str
    status: str
    outputs: Mapping[str, Mapping[str, Any]]


@dataclass(slots=True)
class MeterContext:
    meter: BudgetMeter
    scope: str
    owner_id: str | None
    breach_action: str


@dataclass(slots=True)
class LoopContext:
    loop_id: str
    meter: BudgetMeter | None
    breach_action: str


class _LoopStopSignal(RuntimeError):
    def __init__(self, loop_id: str, breach: BudgetBreach | None = None) -> None:
        super().__init__(f"Loop {loop_id} stopped")
        self.loop_id = loop_id
        self.breach = breach


class FlowRunner:
    """Execute a simplified DSL flow with budget enforcement."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        run_id_factory: Callable[[], str] | None = None,
        trace_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.adapters = dict(adapters)
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))
        self._trace_sink = trace_sink
        self._trace: list[dict[str, Any]] = []
        self._loop_stack: list[LoopContext] = []
        self._node_lookup: dict[str, Mapping[str, Any]] = {}
        self._node_meters: dict[str, BudgetMeter] = {}
        self._soft_node_meters: dict[str, BudgetMeter] = {}
        self._loop_meters: dict[str, BudgetMeter] = {}
        self._run_meter: BudgetMeter | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, spec: Mapping[str, Any], vars: Mapping[str, Any]) -> RunResult:  # noqa: ARG002
        self._trace.clear()
        self._node_lookup = {node["id"]: node for node in spec.get("nodes", [])}
        self._loop_stack.clear()
        self._node_meters.clear()
        self._soft_node_meters.clear()
        self._loop_meters.clear()

        run_budget = (spec.get("globals") or {}).get("run_budget")
        self._run_meter = BudgetMeter(scope="run", config=run_budget)

        loop_body_nodes = {
            target
            for node in spec.get("nodes", [])
            if node.get("kind") == "loop"
            for target in node.get("target_subgraph", [])
        }

        outputs: dict[str, Mapping[str, Any]] = {}
        status = "ok"
        try:
            for node in spec.get("nodes", []):
                node_id = node.get("id")
                if not node_id:
                    continue
                if node_id in loop_body_nodes and node.get("kind") != "loop":
                    # Executed as part of its parent loop.
                    continue
                if node.get("kind") == "loop":
                    self._execute_loop(node, outputs)
                elif node.get("kind") == "unit":
                    self._execute_unit(node, outputs)
                else:
                    raise NotImplementedError(f"Unsupported node kind: {node.get('kind')!r}")
        except BudgetExceededError as exc:
            status = "error"
            # Trace already recorded by the charging layer.
            self._record_trace(
                {
                    "event": "run_error",
                    "reason": "budget_exceeded",
                    "scope": exc.scope,
                    "metric": exc.metric,
                    "attempted": exc.attempted,
                }
            )
        return RunResult(
            run_id=self._run_id_factory(),
            status=status,
            outputs=MappingProxyType(outputs),
        )

    @property
    def trace_events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(event) for event in self._trace)

    # ------------------------------------------------------------------
    # Node execution helpers
    # ------------------------------------------------------------------
    def _execute_unit(self, node: Mapping[str, Any], outputs: dict[str, Mapping[str, Any]]) -> None:
        node_id = node["id"]
        tool_ref = node.get("spec", {}).get("tool_ref")
        if tool_ref not in self.adapters:
            raise KeyError(f"Unknown tool adapter: {tool_ref!r}")
        adapter = self.adapters[tool_ref]

        inputs: Mapping[str, Any] = MappingProxyType({})
        contexts = self._collect_meter_contexts(node)

        estimate = adapter.estimate_cost(node, inputs)
        try:
            self._enforce_preflight(node_id, estimate, contexts)
        except _LoopStopSignal:
            raise
        except BudgetExceededError as exc:
            self._record_budget_breach(
                node_id=node_id,
                scope=exc.scope,
                breach=BudgetBreach(
                    scope=exc.scope,
                    metric=exc.metric,
                    level="hard",
                    limit=exc.limit,
                    attempted=exc.attempted,
                    spent_before=self._current_spend(exc.scope, exc.metric),
                ),
            )
            raise

        self._record_trace({"event": "node_start", "node_id": node_id})
        result = adapter.execute(node, inputs)

        try:
            self._apply_cost(node_id, result.cost if hasattr(result, "cost") else {})
        except _LoopStopSignal as stop:
            self._record_trace(
                {
                    "event": "node_end",
                    "node_id": node_id,
                    "outputs": dict(getattr(result, "outputs", {})),
                    "cost": dict(getattr(result, "cost", {})),
                    "stop_reason": "loop_budget",
                    "loop_id": stop.loop_id,
                }
            )
            raise

        self._record_trace(
            {
                "event": "node_end",
                "node_id": node_id,
                "outputs": dict(getattr(result, "outputs", {})),
                "cost": dict(getattr(result, "cost", {})),
            }
        )
        outputs[node_id] = MappingProxyType(dict(getattr(result, "outputs", {})))

    def _execute_loop(self, node: Mapping[str, Any], outputs: dict[str, Mapping[str, Any]]) -> None:
        loop_id = node["id"]
        stop_conf = (node.get("stop") or {}).get("budget")
        loop_meter = None
        breach_action = "stop"
        if stop_conf:
            loop_meter = self._loop_meters.get(loop_id)
            if loop_meter is None:
                loop_meter = BudgetMeter(scope=f"loop:{loop_id}", config=stop_conf)
                self._loop_meters[loop_id] = loop_meter
            breach_action = str(stop_conf.get("breach_action", "stop")).lower()
        self._loop_stack.append(
            LoopContext(loop_id=loop_id, meter=loop_meter, breach_action=breach_action)
        )
        iterations = 0
        max_iterations = (node.get("stop") or {}).get("max_iterations")
        try:
            while True:
                if max_iterations is not None and iterations >= max_iterations:
                    self._record_trace(
                        {
                            "event": "loop_stop",
                            "loop_id": loop_id,
                            "reason": "max_iterations",
                            "iterations": iterations,
                        }
                    )
                    break
                try:
                    for target_id in node.get("target_subgraph", []):
                        target = self._node_lookup.get(target_id)
                        if target is None:
                            raise KeyError(f"Loop target node not found: {target_id!r}")
                        self._execute_unit(target, outputs)
                except _LoopStopSignal as stop:
                    if stop.loop_id != loop_id:
                        raise
                    self._record_trace(
                        {
                            "event": "loop_stop",
                            "loop_id": loop_id,
                            "reason": "budget_stop",
                            "iterations": iterations,
                            "breach": {
                                "metric": stop.breach.metric if stop.breach else None,
                                "limit": stop.breach.limit if stop.breach else None,
                            },
                        }
                    )
                    break
                iterations += 1
        finally:
            self._loop_stack.pop()

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------
    def _collect_meter_contexts(self, node: Mapping[str, Any]) -> list[MeterContext]:
        contexts: list[MeterContext] = []
        if self._run_meter is not None:
            contexts.append(
                MeterContext(
                    meter=self._run_meter,
                    scope="run",
                    owner_id=None,
                    breach_action="error" if self._run_meter.mode == "hard" else "warn",
                )
            )
        for loop_ctx in self._loop_stack:
            if loop_ctx.meter is None:
                continue
            contexts.append(
                MeterContext(
                    meter=loop_ctx.meter,
                    scope=f"loop:{loop_ctx.loop_id}",
                    owner_id=loop_ctx.loop_id,
                    breach_action=loop_ctx.breach_action,
                )
            )
        node_budget = node.get("budget")
        if node_budget:
            meter = self._node_meters.get(node["id"])
            if meter is None:
                meter = BudgetMeter(scope=f"node:{node['id']}", config=node_budget)
                self._node_meters[node["id"]] = meter
            contexts.append(
                MeterContext(
                    meter=meter,
                    scope=f"node:{node['id']}",
                    owner_id=node["id"],
                    breach_action="error" if meter.mode == "hard" else "warn",
                )
            )
        soft_budget = node.get("spec", {}).get("budget")
        if soft_budget:
            meter = self._soft_node_meters.get(node["id"])
            if meter is None:
                meter = BudgetMeter(
                    scope=f"node:{node['id']}:soft",
                    config=soft_budget,
                    default_mode="soft",
                )
                self._soft_node_meters[node["id"]] = meter
            contexts.append(
                MeterContext(
                    meter=meter,
                    scope=f"node:{node['id']}:soft",
                    owner_id=node["id"],
                    breach_action="warn",
                )
            )
        return contexts

    def _enforce_preflight(
        self,
        node_id: str,
        cost: Mapping[str, float | int],
        contexts: list[MeterContext],
    ) -> None:
        for context in contexts:
            check = context.meter.can_spend(cost)
            if not check.allowed:
                breach = check.breach
                if breach is None:
                    continue
                if context.breach_action == "stop" and context.owner_id:
                    raise _LoopStopSignal(context.owner_id, breach)
                raise BudgetExceededError(
                    scope=breach.scope,
                    metric=breach.metric,
                    limit=breach.limit,
                    attempted=breach.attempted,
                )
            if check.breach is not None:
                self._record_budget_warning(
                    node_id=node_id,
                    scope=context.scope,
                    breach=check.breach,
                )

    def _apply_cost(self, node_id: str, cost: Mapping[str, float | int]) -> None:
        contexts = self._collect_meter_contexts(self._node_lookup[node_id])
        for context in contexts:
            try:
                charge = context.meter.charge(cost)
            except BudgetExceededError as exc:
                spent_before = self._current_spend(context.scope, exc.metric)
                breach = BudgetBreach(
                    scope=context.scope,
                    metric=exc.metric,
                    level="hard",
                    limit=exc.limit,
                    attempted=exc.attempted,
                    spent_before=spent_before,
                )
                self._record_budget_breach(node_id=node_id, scope=context.scope, breach=breach)
                if context.breach_action == "stop" and context.owner_id:
                    raise _LoopStopSignal(context.owner_id, breach) from None
                raise
            else:
                self._handle_soft_breaches(node_id, context.scope, charge)

    def _handle_soft_breaches(self, node_id: str, scope: str, charge: BudgetCharge) -> None:
        for breach in charge.breaches:
            self._record_budget_warning(node_id=node_id, scope=scope, breach=breach)

    def _current_spend(self, scope: str, metric: str) -> float:
        meter = None
        if scope == "run":
            meter = self._run_meter
        elif scope.startswith("loop:"):
            loop_id = scope.split(":", 1)[1]
            meter = self._loop_meters.get(loop_id)
        elif scope.startswith("node:"):
            suffix = scope.split(":", 1)[1]
            if suffix.endswith(":soft"):
                node_id = suffix[:-5]
                meter = self._soft_node_meters.get(node_id)
            else:
                meter = self._node_meters.get(suffix)
        if meter is None:
            return 0.0
        return float(meter.spent_snapshot().get(metric, 0.0))

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------
    def _record_budget_warning(self, *, node_id: str, scope: str, breach: BudgetBreach) -> None:
        self._record_trace(
            {
                "event": "budget_warning",
                "node_id": node_id,
                "scope": scope,
                "level": "soft",
                "metric": breach.metric,
                "attempted": breach.attempted,
                "remaining_before": breach.remaining_before,
            }
        )

    def _record_budget_breach(self, *, node_id: str, scope: str, breach: BudgetBreach) -> None:
        self._record_trace(
            {
                "event": "budget_breach",
                "node_id": node_id,
                "scope": scope,
                "level": breach.level,
                "metric": breach.metric,
                "attempted": breach.attempted,
                "limit": breach.limit,
            }
        )

    def _record_trace(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        self._trace.append(payload)
        if self._trace_sink is not None:
            self._trace_sink(MappingProxyType(payload))

