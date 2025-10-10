"""FlowRunner orchestration with budget and trace integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, MutableMapping, Optional, Sequence

from . import budget, costs, trace


class ToolAdapter:
    """Protocol-like base class for adapters used by ``FlowRunner``."""

    def estimate(self, context: Mapping[str, str]) -> Mapping[str, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def execute(self, context: Mapping[str, str]) -> Mapping[str, object]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int
    scope_keys: Sequence[str]


@dataclass(frozen=True)
class NodeDefinition:
    node_id: str
    adapter_id: str
    scope_keys: Sequence[str]
    loop: Optional[LoopConfig] = None


@dataclass(frozen=True)
class FlowDefinition:
    flow_id: str
    nodes: Sequence[NodeDefinition]


class FlowRunStatus(Enum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class FlowRunResult:
    status: FlowRunStatus
    stop_reason: Optional[str]
    node_results: MutableMapping[str, List[Mapping[str, object]]]


class NullPolicyStack:
    """Fallback policy stack used when none is provided."""

    def resolve(self, node_id: str) -> None:  # pragma: no cover - default noop
        return None

    def validate(self, node_id: str) -> None:  # pragma: no cover - default noop
        return None


class FlowRunner:
    """Execute nodes using adapters while enforcing budgets and emitting traces."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        budget_manager: budget.BudgetManager,
        trace_emitter: trace.TraceEventEmitter,
        flow_scope_keys: Sequence[str],
        policy_stack: Optional[object] = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._budget_manager = budget_manager
        self._trace_emitter = trace_emitter
        self._flow_scope_keys = list(flow_scope_keys)
        self._policy_stack = policy_stack or NullPolicyStack()

    def run(self, flow: FlowDefinition, run_id: str) -> FlowRunResult:
        node_results: MutableMapping[str, List[Mapping[str, object]]] = {}
        status = FlowRunStatus.COMPLETED
        stop_reason: Optional[str] = None

        for node in flow.nodes:
            adapter = self._adapters[node.adapter_id]
            scopes = list(dict.fromkeys([*self._flow_scope_keys, *node.scope_keys]))

            if node.loop:
                loop_status, loop_reason = self._execute_loop(
                    flow_id=flow.flow_id,
                    run_id=run_id,
                    node=node,
                    adapter=adapter,
                    scopes=scopes,
                    node_results=node_results,
                )
                if loop_status is not FlowRunStatus.COMPLETED:
                    status = loop_status
                    stop_reason = loop_reason
                    break
            else:
                result = self._execute_once(
                    flow_id=flow.flow_id,
                    run_id=run_id,
                    node=node,
                    adapter=adapter,
                    scopes=scopes,
                    iteration=None,
                )
                if isinstance(result, FlowRunResult):
                    status = result.status
                    stop_reason = result.stop_reason
                    node_results.update(result.node_results)
                    break
                node_results.setdefault(node.node_id, []).append(result)

        return FlowRunResult(status=status, stop_reason=stop_reason, node_results=node_results)

    def _execute_loop(
        self,
        *,
        flow_id: str,
        run_id: str,
        node: NodeDefinition,
        adapter: ToolAdapter,
        scopes: Sequence[str],
        node_results: MutableMapping[str, List[Mapping[str, object]]],
    ) -> tuple[FlowRunStatus, Optional[str]]:
        loop_scope_keys = list(dict.fromkeys([*scopes, *(node.loop.scope_keys if node.loop else [])]))
        results = node_results.setdefault(node.node_id, [])

        for iteration in range(node.loop.max_iterations):
            context = {
                "flow_id": flow_id,
                "run_id": run_id,
                "node_id": node.node_id,
                "iteration": str(iteration),
            }
            estimate_raw = adapter.estimate(context)
            normalized = costs.normalize_cost(estimate_raw)
            check = self._budget_manager.preview(loop_scope_keys, normalized)
            stop_reason = None
            if check.action is budget.BudgetAction.ERROR:
                self._trace_emitter.record_budget(
                    run_id=run_id,
                    node_id=node.node_id,
                    loop_iteration=iteration,
                    check=check,
                    stop_reason="budget_error",
                )
                raise budget.BudgetBreachError(check)
            if check.action is budget.BudgetAction.STOP:
                stop_reason = "budget_stop"
            self._trace_emitter.record_budget(
                run_id=run_id,
                node_id=node.node_id,
                loop_iteration=iteration,
                check=check,
                stop_reason=stop_reason,
            )
            if check.action is budget.BudgetAction.STOP:
                return FlowRunStatus.STOPPED, "budget_stop"

            self._policy_stack.resolve(node.node_id)
            outcome = self._budget_manager.commit(loop_scope_keys, normalized)
            self._trace_emitter.record_outcome(
                run_id=run_id,
                node_id=node.node_id,
                loop_iteration=iteration,
                outcome=outcome,
            )
            self._policy_stack.validate(node.node_id)
            result = adapter.execute(context)
            results.append(result)

        return FlowRunStatus.COMPLETED, None

    def _execute_once(
        self,
        *,
        flow_id: str,
        run_id: str,
        node: NodeDefinition,
        adapter: ToolAdapter,
        scopes: Sequence[str],
        iteration: Optional[int],
    ) -> Mapping[str, object] | FlowRunResult:
        context = {
            "flow_id": flow_id,
            "run_id": run_id,
            "node_id": node.node_id,
            "iteration": "0" if iteration is None else str(iteration),
        }
        estimate_raw = adapter.estimate(context)
        normalized = costs.normalize_cost(estimate_raw)
        check = self._budget_manager.preview(scopes, normalized)
        stop_reason = None
        if check.action is budget.BudgetAction.ERROR:
            self._trace_emitter.record_budget(
                run_id=run_id,
                node_id=node.node_id,
                loop_iteration=iteration,
                check=check,
                stop_reason="budget_error",
            )
            raise budget.BudgetBreachError(check)
        if check.action is budget.BudgetAction.STOP:
            stop_reason = "budget_stop"
        self._trace_emitter.record_budget(
            run_id=run_id,
            node_id=node.node_id,
            loop_iteration=iteration,
            check=check,
            stop_reason=stop_reason,
        )
        if check.action is budget.BudgetAction.STOP:
            return FlowRunResult(
                status=FlowRunStatus.STOPPED,
                stop_reason="budget_stop",
                node_results={node.node_id: []},
            )

        self._policy_stack.resolve(node.node_id)
        outcome = self._budget_manager.commit(scopes, normalized)
        self._trace_emitter.record_outcome(
            run_id=run_id,
            node_id=node.node_id,
            loop_iteration=iteration,
            outcome=outcome,
        )
        self._policy_stack.validate(node.node_id)
        return adapter.execute(context)
