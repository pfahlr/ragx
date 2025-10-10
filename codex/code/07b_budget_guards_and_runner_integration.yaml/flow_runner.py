from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, MutableMapping, Protocol, Sequence

try:  # pragma: no cover - allow standalone loading
    from .budget_manager import BudgetDecision, BudgetManager
    from .budget_models import BudgetBreachError, CostSnapshot
    from .trace import TraceEventEmitter
except ImportError:  # pragma: no cover
    from budget_manager import BudgetDecision, BudgetManager  # type: ignore[F401]
    from budget_models import BudgetBreachError, CostSnapshot  # type: ignore[F401]
    from trace import TraceEventEmitter  # type: ignore[F401]


class ToolAdapter(Protocol):
    def estimate_cost(self) -> CostSnapshot:  # pragma: no cover - interface
        ...

    def execute(self) -> "ExecutionReport":  # pragma: no cover - interface
        ...


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    output: Mapping[str, object]
    cost: CostSnapshot
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FlowNode:
    id: str
    kind: str
    adapter_key: str
    budget_scope: str | None = None

    def scope_id(self) -> str:
        return self.budget_scope or self.id


@dataclass(frozen=True, slots=True)
class LoopConfig:
    id: str
    nodes: Sequence[FlowNode]
    max_iterations: int
    budget_scope: str | None = None

    def scope_id(self) -> str:
        return self.budget_scope or self.id


PlanNode = FlowNode | LoopConfig


@dataclass(frozen=True, slots=True)
class FlowPlan:
    run_id: str
    nodes: Sequence[PlanNode]


@dataclass(frozen=True, slots=True)
class LoopSummary:
    iterations: int
    stop_reason: str


@dataclass(frozen=True, slots=True)
class FlowResult:
    loop_summaries: Mapping[str, LoopSummary]


@dataclass(slots=True)
class _NodeOutcome:
    executed: bool
    decision: BudgetDecision | None = None
    preflight: BudgetDecision | None = None


class FlowRunner:
    def __init__(
        self,
        *,
        budget_manager: BudgetManager,
        trace_emitter: TraceEventEmitter,
    ) -> None:
        self._budget_manager = budget_manager
        self._trace_emitter = trace_emitter

    def run(self, plan: FlowPlan, *, adapters: Mapping[str, ToolAdapter]) -> FlowResult:
        loop_summaries: MutableMapping[str, LoopSummary] = {}
        for node in plan.nodes:
            if isinstance(node, LoopConfig):
                summary = self._execute_loop(node, adapters)
                loop_summaries[node.scope_id()] = summary
            elif isinstance(node, FlowNode):
                self._execute_node(node, adapters, iteration_label=node.id)
            else:  # pragma: no cover - defensive guard
                raise TypeError(f"Unsupported plan node: {type(node)!r}")
        return FlowResult(loop_summaries=MappingProxyType(dict(loop_summaries)))

    # ------------------------------------------------------------------
    # Loop + node execution helpers
    # ------------------------------------------------------------------
    def _execute_loop(
        self, loop: LoopConfig, adapters: Mapping[str, ToolAdapter]
    ) -> LoopSummary:
        iterations = 0
        stop_reason: str | None = None
        for index in range(loop.max_iterations):
            executed_nodes = 0
            for flow_node in loop.nodes:
                label = f"{flow_node.id}#{index + 1}"
                outcome = self._execute_node(flow_node, adapters, iteration_label=label)
                if not outcome.executed:
                    stop_reason = "budget_stop"
                    break
                executed_nodes += 1
                if outcome.decision and outcome.decision.stop_requested:
                    stop_reason = "budget_stop"
                    break
            if stop_reason is not None:
                if executed_nodes:
                    iterations += 1
                break
            iterations += 1
        else:
            stop_reason = "max_iterations"

        if stop_reason is None:
            stop_reason = "completed"

        self._trace_emitter.emit(
            "loop_stop",
            loop.scope_id(),
            {"reason": stop_reason, "iterations": iterations},
        )
        return LoopSummary(iterations=iterations, stop_reason=stop_reason)

    def _execute_node(
        self,
        node: FlowNode,
        adapters: Mapping[str, ToolAdapter],
        *,
        iteration_label: str,
    ) -> _NodeOutcome:
        adapter = adapters[node.adapter_key]
        scope_id = node.scope_id()
        estimate = adapter.estimate_cost()
        preview = self._budget_manager.preflight(scope_id, estimate, label=f"{iteration_label}:preflight")
        if preview.stop_requested:
            critical = self._critical_breach(preview)
            if critical is not None:
                self._emit_preflight_traces(preview, iteration_label)
                raise BudgetBreachError(critical)
            return _NodeOutcome(executed=False, preflight=preview)
        report = adapter.execute()
        decision = self._budget_manager.commit(scope_id, report.cost, label=iteration_label)
        return _NodeOutcome(executed=True, decision=decision)

    def _critical_breach(self, decision: BudgetDecision) -> object | None:
        for breach in decision.breaches:
            action = (breach.breach_action or "").lower()
            if breach.breach_kind is not None and getattr(breach.breach_kind, "value", str(breach.breach_kind)) == "hard":
                return breach
            if action == "error":
                return breach
        return None

    def _emit_preflight_traces(self, decision: BudgetDecision, label: str) -> None:
        for breach in decision.breaches:
            payload = {
                "label": f"{label}:preflight",
                "remaining": breach.remaining,
                "overages": breach.overages,
                "breach_action": breach.breach_action,
                "phase": "preflight",
            }
            self._trace_emitter.emit("budget_breach", breach.scope, payload)
