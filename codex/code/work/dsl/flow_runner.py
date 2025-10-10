"""Adapter-backed FlowRunner integrating budgets and policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pkgs.dsl.policy import PolicyStack, PolicyTraceEvent, PolicyViolationError

from . import budget_models as bm
from .budget_manager import BudgetBreachError, BudgetError, BudgetManager
from .trace import TraceEventEmitter

__all__ = ["ToolAdapter", "NodeExecution", "FlowRunner"]


@runtime_checkable
class ToolAdapter(Protocol):
    """Protocol implemented by tool adapters."""

    def estimate_cost(self, node: Mapping[str, object]) -> Mapping[str, object]:
        ...

    def execute(self, node: Mapping[str, object]) -> object:
        ...


@dataclass(frozen=True, slots=True)
class NodeExecution:
    """Immutable record of node execution."""

    node_id: str
    tool: str
    result: object


class _PolicyEventBridge:
    """Bridge PolicyStack events into the shared TraceEventEmitter."""

    def __init__(self, emitter: TraceEventEmitter) -> None:
        self._emitter = emitter

    def __call__(self, event: PolicyTraceEvent) -> None:
        self._emitter.emit(
            event.event,
            scope_type="policy",
            scope_id=event.scope,
            payload=event.data,
        )


class FlowRunner:
    """Execute flow nodes with policy enforcement and budget guards."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        budget_manager: BudgetManager,
        policy_stack: PolicyStack,
        trace: TraceEventEmitter | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._budgets = budget_manager
        self._policies = policy_stack
        self._trace = trace or budget_manager.trace
        self._loop_stack: list[bm.ScopeKey] = []
        # Attach a policy trace bridge if none exists so policy events surface in traces.
        if getattr(self._policies, "_event_sink", None) is None:
            bridge = _PolicyEventBridge(self._trace)
            setattr(self._policies, "_event_sink", bridge)
            self._policy_bridge = bridge
        else:
            self._policy_bridge = None

    def run(
        self,
        *,
        flow_id: str,
        run_id: str,
        nodes: Iterable[Mapping[str, object]],
    ) -> list[NodeExecution]:
        run_scope = bm.ScopeKey(scope_type="run", scope_id=run_id)
        self._budgets.enter_scope(run_scope)
        self._trace.emit(
            "run_start",
            scope_type="run",
            scope_id=run_id,
            payload={"flow_id": flow_id},
        )
        executions: list[NodeExecution] = []
        try:
            for raw_node in nodes:
                kind = str(raw_node.get("kind", "unit"))
                if kind == "loop":
                    self._execute_loop(
                        raw_node,
                        flow_id=flow_id,
                        run_id=run_id,
                        run_scope=run_scope,
                        executions=executions,
                    )
                else:
                    execution = self._execute_unit(
                        raw_node,
                        run_scope=run_scope,
                        run_id=run_id,
                        flow_id=flow_id,
                    )
                    if execution is not None:
                        executions.append(execution)
            self._trace.emit(
                "run_complete",
                scope_type="run",
                scope_id=run_id,
                payload={"flow_id": flow_id},
            )
            return executions
        finally:
            self._budgets.exit_scope(run_scope)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_budget(self, scope: bm.ScopeKey, cost: bm.CostSnapshot) -> None:
        decision = self._budgets.preview_charge(scope, cost)
        if decision.breached:
            self._budgets.record_breach(decision)
        if decision.should_stop:
            blocking = decision.blocking
            if blocking is None:  # pragma: no cover - defensive guard
                raise BudgetError("blocking outcome missing for stop decision")
            raise BudgetBreachError(scope, blocking)
        self._budgets.commit_charge(decision)

    def _execute_unit(
        self,
        node: Mapping[str, object],
        *,
        run_scope: bm.ScopeKey,
        run_id: str,
        flow_id: str,
    ) -> NodeExecution | None:
        node_id = str(node["id"])
        tool = str(node["tool"])
        adapter = self._adapters.get(tool)
        if adapter is None:
            raise KeyError(f"unknown adapter for tool '{tool}'")

        node_scope = bm.ScopeKey(scope_type="node", scope_id=node_id)
        self._budgets.enter_scope(node_scope)
        policy_scope = f"node:{node_id}"
        policy_pushed = False
        node_policy = node.get("policy")
        if node_policy is not None:
            self._policies.push(node_policy, scope=policy_scope, source=node_id)
            policy_pushed = True

        self._trace.emit(
            "node_start",
            scope_type="node",
            scope_id=node_id,
            payload={"tool": tool, "run_id": run_id, "flow_id": flow_id},
        )
        try:
            self._policies.effective_allowlist([tool])
            self._policies.enforce(tool)

            node_payload = dict(node)
            cost_snapshot = bm.CostSnapshot.from_raw(adapter.estimate_cost(node_payload))
            self._charge_active_scopes(run_scope, node_scope, cost_snapshot)
            result = adapter.execute(node_payload)
            execution = NodeExecution(node_id=node_id, tool=tool, result=result)
            self._trace.emit(
                "node_complete",
                scope_type="node",
                scope_id=node_id,
                payload={"tool": tool, "run_id": run_id, "flow_id": flow_id},
            )
            return execution
        finally:
            if policy_pushed:
                self._policies.pop(expected_scope=policy_scope)
            self._budgets.exit_scope(node_scope)

    def _execute_loop(
        self,
        loop_node: Mapping[str, object],
        *,
        flow_id: str,
        run_id: str,
        run_scope: bm.ScopeKey,
        executions: list[NodeExecution],
    ) -> None:
        loop_id = str(loop_node["id"])
        loop_scope = bm.ScopeKey(scope_type="loop", scope_id=loop_id)
        stop_config = loop_node.get("stop", {})
        max_iterations = int(stop_config.get("max_iterations", 1))
        if max_iterations < 1:
            max_iterations = 1
        sub_nodes = list(loop_node.get("target_subgraph", []))

        self._budgets.enter_scope(loop_scope)
        self._loop_stack.append(loop_scope)
        loop_policy = loop_node.get("policy")
        policy_scope = f"loop:{loop_id}"
        policy_pushed = False
        if loop_policy is not None:
            self._policies.push(loop_policy, scope=policy_scope, source=loop_id)
            policy_pushed = True

        iterations = 0
        stop_reason = "max_iterations"
        breach_outcome = None
        try:
            for _ in range(max_iterations):
                try:
                    for sub_node in sub_nodes:
                        execution = self._execute_unit(
                            sub_node,
                            run_scope=run_scope,
                            run_id=run_id,
                            flow_id=flow_id,
                        )
                        if execution is not None:
                            executions.append(execution)
                    iterations += 1
                except BudgetBreachError as exc:
                    if exc.scope.scope_type == "loop" and exc.scope.scope_id == loop_id:
                        stop_reason = "budget_breach"
                        breach_outcome = exc.outcome
                        break
                    raise
        finally:
            if policy_pushed:
                self._policies.pop(expected_scope=policy_scope)
            self._loop_stack.pop()
            self._budgets.exit_scope(loop_scope)
            payload = {
                "loop_id": loop_id,
                "run_id": run_id,
                "flow_id": flow_id,
                "iterations": iterations,
                "max_iterations": max_iterations,
                "stop_reason": stop_reason,
                "breached": breach_outcome is not None,
            }
            if breach_outcome is not None:
                payload["breach_spec"] = breach_outcome.spec.name
                payload["remaining"] = breach_outcome.charge.remaining.to_payload()
                payload["overage"] = breach_outcome.charge.overage.to_payload()
            self._trace.emit(
                "loop_summary",
                scope_type="loop",
                scope_id=loop_id,
                payload=payload,
            )

    def _charge_active_scopes(
        self,
        run_scope: bm.ScopeKey,
        node_scope: bm.ScopeKey,
        cost: bm.CostSnapshot,
    ) -> None:
        scopes = [run_scope, *self._loop_stack, node_scope]
        for scope in scopes:
            self._apply_budget(scope, cost)
