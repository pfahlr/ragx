"""FlowRunner integrating budget manager and policy stack for sandbox tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence

from .budget import (
    BudgetDecisionStatus,
    BudgetManager,
    Cost,
    ScopeKey,
)
from .policy import PolicyDecision, PolicyStack
from .trace import TraceEventEmitter


class ToolAdapter(Protocol):
    def estimate(self, context: dict) -> Cost:
        ...

    def execute(self, context: dict) -> object:
        ...


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    adapter: ToolAdapter
    scope_key: ScopeKey


@dataclass(frozen=True)
class LoopSpec:
    loop_id: str
    iterations: int
    nodes: Sequence[str]
    scope_key: ScopeKey


@dataclass(frozen=True)
class FlowSpec:
    flow_id: str
    nodes: Sequence[NodeSpec]
    loops: Sequence[LoopSpec]
    run_scope: ScopeKey = field(default_factory=lambda: ScopeKey("run", "run"))

    def node_map(self) -> Dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}


@dataclass(frozen=True)
class LoopSummary:
    loop_id: str
    iterations_completed: int
    stop_reason: str | None


@dataclass(frozen=True)
class RunResult:
    completed: bool
    stop_reason: str | None
    loop_summaries: Sequence[LoopSummary]


class FlowRunner:
    """Executes flows with budget enforcement."""

    def __init__(
        self,
        budget_manager: BudgetManager,
        trace_emitter: TraceEventEmitter,
        policy_stack: PolicyStack,
    ) -> None:
        self._budget_manager = budget_manager
        self.trace_emitter = trace_emitter
        self._policy_stack = policy_stack

    def run(self, flow: FlowSpec) -> RunResult:
        node_lookup = flow.node_map()
        loop_summaries: List[LoopSummary] = []
        run_stop_reason: str | None = None

        for loop in flow.loops:
            iterations_completed = 0
            loop_stop_reason: str | None = None
            saw_warning = False

            for iteration in range(loop.iterations):
                iteration_stop = False
                for node_id in loop.nodes:
                    node = node_lookup[node_id]
                    context = {"flow_id": flow.flow_id, "loop_id": loop.loop_id, "iteration": iteration}
                    decision = self._policy_stack.evaluate(node.node_id, context)
                    if not decision.allowed:
                        run_stop_reason = f"policy_violation({node.node_id})"
                        loop_stop_reason = "policy_violation"
                        iteration_stop = True
                        self._policy_stack.complete(node.node_id)
                        break

                    cost = node.adapter.estimate(context)
                    for scope_key in (flow.run_scope, loop.scope_key, node.scope_key):
                        outcome = self._budget_manager.commit(scope_key, cost)
                        self.trace_emitter.emit_budget_charge(outcome)
                        if outcome.breach:
                            self.trace_emitter.emit_budget_breach(outcome.breach)
                            if outcome.decision.status is BudgetDecisionStatus.WARN and loop_stop_reason != "budget_stop":
                                saw_warning = True
                                loop_stop_reason = loop_stop_reason or "soft_budget_warn"
                        if outcome.requires_stop and loop_stop_reason != "budget_stop":
                            loop_stop_reason = "budget_stop"
                            if run_stop_reason is None:
                                run_stop_reason = f"budget_breach({outcome.charge.scope})"
                            iteration_stop = True
                            break
                    if not iteration_stop:
                        node.adapter.execute(context)
                    self._policy_stack.complete(node.node_id)
                    if iteration_stop:
                        break
                iterations_completed += 1
                if iteration_stop:
                    break

            if loop_stop_reason is None and saw_warning:
                loop_stop_reason = "soft_budget_warn"

            loop_summary = LoopSummary(
                loop_id=loop.loop_id,
                iterations_completed=iterations_completed,
                stop_reason=loop_stop_reason,
            )
            loop_summaries.append(loop_summary)
            self.trace_emitter.emit_loop_summary(loop.scope_key, iterations_completed, loop_stop_reason)

            if run_stop_reason:
                break

        completed = run_stop_reason is None
        return RunResult(completed=completed, stop_reason=run_stop_reason, loop_summaries=tuple(loop_summaries))
