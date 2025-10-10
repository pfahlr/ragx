"""FlowRunner integrating budgets, policies, and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from .budget import BreachAction, BudgetDecision, BudgetManager, CostSnapshot
from .policy import PolicyStack
from .trace import TraceEventEmitter


@dataclass
class FlowNode:
    node_id: str
    adapter: Any
    input_payload: Mapping[str, Any]


@dataclass
class RunContext:
    run_id: str


@dataclass
class RunResult:
    completed_nodes: List[str]
    warnings: List[str]
    breaches: List[Mapping[str, Any]]


class FlowRunner:
    """Execute nodes while enforcing budgets and emitting traces."""

    def __init__(self, budget_manager: BudgetManager, trace_emitter: TraceEventEmitter, policy_stack: PolicyStack) -> None:
        self._budget_manager = budget_manager
        self._trace = trace_emitter
        self._policy_stack = policy_stack

    def run(self, flow: Sequence[FlowNode], context: RunContext) -> RunResult:
        completed: List[str] = []
        warnings: List[str] = []
        breaches: List[Mapping[str, Any]] = []

        for node in flow:
            adapter_name = getattr(node.adapter, "name", node.adapter.__class__.__name__)
            policy_payload = {"adapter": adapter_name}
            self._policy_stack.push(node.node_id, policy_payload)
            self._trace.emit_policy_event("policy_push", node.node_id, policy_payload)

            estimate = node.adapter.estimate_cost(context, node)
            preflight = self._budget_manager.preflight("run", context.run_id, estimate)
            if preflight.should_stop:
                breaches.append(self._decision_payload(preflight, phase="preflight"))
                self._trace.emit_budget_breach(preflight)
                self._policy_stack.resolve(node.node_id)
                self._trace.emit_policy_event("policy_resolved", node.node_id, {"phase": "preflight"})
                self._policy_stack.pop(node.node_id)
                self._trace.emit_policy_event("policy_pop", node.node_id, {"phase": "preflight"})
                break

            node.adapter.execute(context, node)
            actual_snapshot = self._resolve_actual_cost(node.adapter, estimate)
            decision = self._budget_manager.commit("run", context.run_id, actual_snapshot)
            self._trace.emit_budget_charge(decision)

            if decision.is_breached and decision.action == BreachAction.WARN:
                warnings.append(self._format_warning(context.run_id, decision, phase="commit"))
            if decision.should_stop:
                breaches.append(self._decision_payload(decision, phase="commit"))
                self._trace.emit_budget_breach(decision)

            completed.append(node.node_id)
            self._policy_stack.resolve(node.node_id)
            self._trace.emit_policy_event("policy_resolved", node.node_id, {"phase": "commit"})
            self._policy_stack.pop(node.node_id)
            self._trace.emit_policy_event("policy_pop", node.node_id, {"phase": "commit"})

            if decision.should_stop:
                break

        return RunResult(completed_nodes=completed, warnings=warnings, breaches=breaches)

    def _resolve_actual_cost(self, adapter: Any, estimate: CostSnapshot) -> CostSnapshot:
        actual_fn = getattr(adapter, "actual_cost_snapshot", None)
        if callable(actual_fn):
            return actual_fn()
        actual_ms = getattr(adapter, "actual_ms", None)
        if actual_ms is not None:
            return CostSnapshot(milliseconds=float(actual_ms))
        return estimate

    def _format_warning(self, scope_id: str, decision: BudgetDecision, phase: str) -> str:
        return (
            f"Budget warn {scope_id}:{phase} overage={decision.overage_ms:.1f}ms "
            f"remaining={decision.remaining_ms:.1f}ms"
        )

    def _decision_payload(self, decision: BudgetDecision, phase: str) -> Mapping[str, Any]:
        return {
            "scope_id": decision.scope_id,
            "phase": phase,
            "action": decision.action.value,
            "overage_ms": decision.overage_ms,
            "remaining_ms": decision.remaining_ms,
        }
