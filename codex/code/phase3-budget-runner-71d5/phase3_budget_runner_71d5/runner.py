from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Union

from pkgs.dsl.policy import PolicyStack

from .adapters import ToolAdapter, ToolExecutionResult
from .budgeting import (
    BreachAction,
    BudgetChargeResult,
    BudgetContext,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
    ScopeKey,
    ScopeType,
)
from .trace import TraceEventEmitter

__all__ = [
    "FlowPlan",
    "FlowRunner",
    "LoopPlan",
    "NodeExecution",
    "NodePlan",
    "RunResult",
]


@dataclass(frozen=True, slots=True)
class NodePlan:
    node_id: str
    tool: str
    budget: BudgetSpec | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoopPlan:
    loop_id: str
    iterations: int
    body: Sequence[NodePlan]
    budget: BudgetSpec | None = None


PlanStep = Union[NodePlan, LoopPlan]


@dataclass(frozen=True, slots=True)
class FlowPlan:
    flow_id: str
    run_budget: BudgetSpec | None
    steps: Sequence[PlanStep]


@dataclass(frozen=True, slots=True)
class NodeExecution:
    node_id: str
    tool: str
    cost: CostSnapshot
    iteration: int | None
    warnings: tuple[str, ...]
    stopped: bool


@dataclass(frozen=True, slots=True)
class RunResult:
    executions: tuple[NodeExecution, ...]
    warnings: tuple[str, ...]
    stop_reason: str | None


class FlowRunner:
    """Budget-aware FlowRunner that enforces policies and emits traces."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        policy_stack: PolicyStack,
        budget_manager: BudgetManager,
        trace_emitter: TraceEventEmitter,
    ) -> None:
        self._adapters = dict(adapters)
        self._policy = policy_stack
        self._budgets = budget_manager
        self._trace = trace_emitter

    def run(self, plan: FlowPlan, *, context: Mapping[str, object] | None = None) -> RunResult:
        run_scope = ScopeKey(ScopeType.RUN, plan.flow_id)
        self._budgets.enter_scope(run_scope, plan.run_budget, parent=None)
        executions: list[NodeExecution] = []
        warnings: list[str] = []
        stop_reason: str | None = None

        try:
            for step in plan.steps:
                if isinstance(step, LoopPlan):
                    loop_result = self._execute_loop(step, run_scope)
                    executions.extend(loop_result.executions)
                    warnings.extend(loop_result.warnings)
                    if loop_result.stop_reason is not None:
                        stop_reason = loop_result.stop_reason
                        break
                else:
                    node_exec, node_warnings, reason = self._execute_node(step, run_scope, loop_scope=None, iteration=None)
                    executions.append(node_exec)
                    warnings.extend(node_warnings)
                    if reason is not None:
                        stop_reason = reason
                        break
        finally:
            self._budgets.exit_scope(run_scope)

        return RunResult(executions=tuple(executions), warnings=tuple(warnings), stop_reason=stop_reason)

    def _execute_loop(self, loop: LoopPlan, run_scope: ScopeKey) -> RunResult:
        loop_scope = ScopeKey(ScopeType.LOOP, loop.loop_id)
        self._budgets.enter_scope(loop_scope, loop.budget, parent=run_scope)
        executions: list[NodeExecution] = []
        warnings: list[str] = []
        stop_reason: str | None = None

        try:
            for iteration in range(loop.iterations):
                for node in loop.body:
                    node_exec, node_warnings, reason = self._execute_node(
                        node,
                        run_scope,
                        loop_scope=loop_scope,
                        iteration=iteration,
                    )
                    executions.append(node_exec)
                    warnings.extend(node_warnings)
                    if reason is not None:
                        stop_reason = reason
                        break
                if stop_reason is not None:
                    break
        finally:
            snapshot = self._budgets.snapshot(loop_scope)
            payload = {
                "scope": loop_scope.identifier,
                "scope_type": loop_scope.scope_type.value,
                "iterations": len({execution.iteration for execution in executions if execution.iteration is not None}),
                "warnings": tuple(warnings),
            }
            if snapshot is not None:
                payload["spent"] = snapshot.spent.as_dict()
                payload["remaining"] = snapshot.remaining.as_dict()
                payload["overages"] = snapshot.overages.as_dict()
            self._trace.emit("loop_summary", scope=loop_scope.identifier, payload=payload)
            self._budgets.exit_scope(loop_scope)

        return RunResult(executions=tuple(executions), warnings=tuple(warnings), stop_reason=stop_reason)

    def _execute_node(
        self,
        node: NodePlan,
        run_scope: ScopeKey,
        *,
        loop_scope: ScopeKey | None,
        iteration: int | None,
    ) -> tuple[NodeExecution, tuple[str, ...], str | None]:
        adapter = self._adapters.get(node.tool)
        if adapter is None:
            raise KeyError(f"unknown tool adapter '{node.tool}'")

        context = dict(node.parameters)
        if iteration is not None:
            context["iteration"] = iteration

        node_scope: ScopeKey | None = None
        if node.budget is not None:
            node_scope = ScopeKey(ScopeType.NODE, node.node_id)
            parent = loop_scope or run_scope
            self._budgets.enter_scope(node_scope, node.budget, parent=parent)

        policy_snapshot = self._policy.enforce(node.tool, raise_on_violation=True)
        self._trace.emit(
            "policy_resolved",
            scope="stack",
            payload={"allowed": sorted(policy_snapshot.allowed), "denied": dict(policy_snapshot.denied)},
        )

        budget_context = BudgetContext(run=run_scope, loop=loop_scope, node=node_scope)
        estimate = adapter.estimate(node, context)
        preview = self._budgets.preview(budget_context, estimate, label=f"{node.node_id}:estimate")
        warnings = list(preview.warnings)
        stop_reason = self._breach_reason(preview)

        execution_result: ToolExecutionResult | None = None
        if stop_reason is None:
            execution_result = adapter.execute(node, context)
            commit = self._budgets.commit(
                budget_context,
                execution_result.cost,
                label=f"{node.node_id}:execute",
            )
            warnings.extend(commit.warnings)
            stop_reason = self._breach_reason(commit)
        else:
            commit = None

        if node_scope is not None:
            self._budgets.exit_scope(node_scope)

        final_cost = execution_result.cost if execution_result is not None else estimate
        execution = NodeExecution(
            node_id=node.node_id,
            tool=node.tool,
            cost=final_cost,
            iteration=iteration,
            warnings=tuple(warnings),
            stopped=stop_reason is not None,
        )
        return execution, tuple(warnings), stop_reason

    def _breach_reason(self, result: BudgetChargeResult) -> str | None:
        if not result.should_stop:
            return None
        for scope, outcome in result.outcomes.items():
            if not outcome.breached:
                continue
            if outcome.spec.mode is BudgetMode.HARD or outcome.spec.breach_action is BreachAction.STOP:
                return f"budget_breach:{scope.scope_type.value}"
        for scope, outcome in result.outcomes.items():
            if outcome.breached:
                return f"budget_breach:{scope.scope_type.value}"
        return "budget_breach"
