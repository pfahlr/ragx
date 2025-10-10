"""Adapter-driven FlowRunner for the sandbox environment.

The loop structure reuses the adapter/loop orchestration ideas from
`codex/implement-budget-guards-with-test-first-approach-fa0vm9` while relying on
BudgetManager previews from the zwi2ny/qhq0jq branches. Policy enforcement is
kept synchronous and emits `policy_resolved` traces before any budget work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .budget import BudgetHardStop, BudgetManager, BudgetSpec, CostSnapshot
from .policy import PolicyStack
from .trace import TraceEventEmitter


class ToolAdapter(Protocol):
    """Protocol describing the adapter interface expected by the runner."""

    name: str

    def estimate(self, context: "RunContext") -> CostSnapshot:
        ...

    def execute(self, context: "RunContext") -> object:
        ...


@dataclass(frozen=True, slots=True)
class FlowNode:
    node_id: str
    adapter: ToolAdapter
    budget_spec: BudgetSpec
    scope_type: str


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str


@dataclass(frozen=True, slots=True)
class NodeExecution:
    node_id: str
    adapter: str
    result: object
    cost: CostSnapshot


@dataclass(frozen=True, slots=True)
class RunResult:
    completed_nodes: list[NodeExecution]
    stop_reason: str | None = None


class FlowRunner:
    """Coordinates policies, budgets, and adapters for sequential node execution."""

    def __init__(
        self,
        *,
        budget_manager: BudgetManager,
        policy_stack: PolicyStack,
        emitter: TraceEventEmitter,
    ) -> None:
        self._budget_manager = budget_manager
        self._policy_stack = policy_stack
        self._emitter = emitter

    def run(self, flow: object, context: RunContext) -> RunResult:
        completed: list[NodeExecution] = []
        stop_reason: str | None = None

        nodes = getattr(flow, "nodes", [])
        for node in nodes:
            adapter = node.adapter
            self._policy_stack.check(context.run_id, node.node_id, adapter.name)

            estimate = adapter.estimate(context)
            scope_key = f"{context.run_id}:{node.budget_spec.scope_id}"
            decision = self._budget_manager.preflight(
                context.run_id,
                scope_key,
                node.scope_type,
                node.node_id,
                node.budget_spec,
                estimate,
            )

            if decision.should_stop:
                try:
                    self._budget_manager.commit(decision)
                except BudgetHardStop:
                    stop_reason = "budget_hard_stop"
                    self._emitter.loop_summary(
                        run_id=context.run_id,
                        node_id=node.node_id,
                        stop_reason=stop_reason,
                        spent=decision.projected_spend,
                    )
                    break
            else:
                result = adapter.execute(context)
                try:
                    outcome = self._budget_manager.commit(decision)
                except BudgetHardStop:
                    stop_reason = "budget_hard_stop"
                    self._emitter.loop_summary(
                        run_id=context.run_id,
                        node_id=node.node_id,
                        stop_reason=stop_reason,
                        spent=decision.projected_spend,
                    )
                    break
                completed.append(
                    NodeExecution(
                        node_id=node.node_id,
                        adapter=adapter.name,
                        result=result,
                        cost=outcome.cost,
                    )
                )
                self._policy_stack.resolved(context.run_id, node.node_id, adapter.name)
                self._emitter.loop_summary(
                    run_id=context.run_id,
                    node_id=node.node_id,
                    stop_reason=None,
                    spent=outcome.projected_spend,
                )

        return RunResult(completed_nodes=completed, stop_reason=stop_reason)
