from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pkgs.dsl.policy import PolicyStack, PolicyViolationError

from . import budget_models as bm
from .budget_manager import BudgetBreachError, BudgetManager
from .policy_bridge import PolicyTraceBridge
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


@dataclass(frozen=True, slots=True)
class _LoopHalt:
    decision: bm.BudgetDecision


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
        downstream = getattr(policy_stack, "_event_sink", None)
        self._policy_bridge = PolicyTraceBridge(self._trace, downstream)
        policy_stack._event_sink = self._policy_bridge

    def run(
        self,
        *,
        flow_id: str,
        run_id: str,
        nodes: Iterable[Mapping[str, object]],
        run_policy: Mapping[str, Iterable[str]] | None = None,
    ) -> list[NodeExecution]:
        run_scope = bm.ScopeKey(scope_type="run", scope_id=run_id)
        self._budgets.enter_scope(run_scope)
        run_policy_scope = f"run:{run_id}"
        run_policy_active = False
        executions: list[NodeExecution] = []
        try:
            if run_policy is not None:
                self._policies.push(run_policy, scope=run_policy_scope, source=flow_id)
                run_policy_active = True
            self._trace.emit(
                "run_start",
                scope_type="run",
                scope_id=run_id,
                payload={"flow_id": flow_id},
            )
            for raw_node in nodes:
                node_type = str(raw_node.get("type", "tool"))
                if node_type == "loop":
                    executions.extend(self._execute_loop(run_scope, raw_node))
                    continue
                execution, _ = self._execute_tool_node(run_scope, raw_node, loop_scope=None)
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
            if run_policy_active:
                self._policies.pop(run_policy_scope)
            self._budgets.exit_scope(run_scope)

    # ------------------------------------------------------------------
    # Loop orchestration
    # ------------------------------------------------------------------
    def _execute_loop(
        self,
        run_scope: bm.ScopeKey,
        node: Mapping[str, object],
    ) -> list[NodeExecution]:
        loop_id = str(node["id"])
        body_nodes = self._normalise_body(node.get("body"))
        max_iterations = self._loop_max_iterations(node)
        loop_scope = bm.ScopeKey(scope_type="loop", scope_id=loop_id)
        self._budgets.enter_scope(loop_scope)

        loop_policy = node.get("policy")
        policy_scope = f"loop:{loop_id}"
        policy_active = False
        iterations = 0
        stop_reason = "completed"
        blocking: bm.BudgetChargeOutcome | None = None
        executions: list[NodeExecution] = []

        try:
            if loop_policy is not None:
                self._policies.push(loop_policy, scope=policy_scope, source=loop_id)
                policy_active = True

            while True:
                if max_iterations is not None and iterations >= max_iterations:
                    stop_reason = "max_iterations"
                    break

                if not body_nodes:
                    break

                iteration_records: list[NodeExecution] = []
                for body in body_nodes:
                    execution, halt = self._execute_tool_node(
                        run_scope,
                        body,
                        loop_scope=loop_scope,
                    )
                    if execution is not None:
                        iteration_records.append(execution)
                    if halt is not None:
                        blocking = halt.decision.blocking
                        stop_reason = "budget_stop"
                        break

                if iteration_records:
                    executions.extend(iteration_records)
                    iterations += 1

                if blocking is not None:
                    break

        finally:
            payload = {
                "loop_id": loop_id,
                "iterations": iterations,
                "stop_reason": stop_reason,
            }
            if blocking is not None:
                payload["breach"] = blocking.to_trace_payload(
                    scope_type="loop",
                    scope_id=loop_id,
                )
            self._trace.emit(
                "loop_summary",
                scope_type="loop",
                scope_id=loop_id,
                payload=payload,
            )
            if policy_active:
                self._policies.pop(policy_scope)
            self._budgets.exit_scope(loop_scope)

        return executions

    def _normalise_body(
        self, body: object | None
    ) -> Sequence[Mapping[str, object]]:
        if body is None:
            return ()
        if isinstance(body, Mapping):
            return (body,)  # pragma: no cover - defensive convenience
        try:
            normalised: list[Mapping[str, object]] = []
            for entry in body:  # type: ignore[arg-type]
                if not isinstance(entry, Mapping):
                    raise TypeError("loop body entries must be mappings")
                normalised.append(entry)
            return tuple(normalised)
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise TypeError("loop body must be iterable of mappings") from exc

    def _loop_max_iterations(self, node: Mapping[str, object]) -> int | None:
        raw_stop = node.get("stop")
        max_iterations = node.get("max_iterations")
        if max_iterations is None and isinstance(raw_stop, Mapping):
            max_iterations = raw_stop.get("max_iterations")
        if max_iterations is None:
            return None
        value = int(max_iterations)
        if value < 0:
            raise ValueError("max_iterations must be >= 0")
        return value

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------
    def _execute_tool_node(
        self,
        run_scope: bm.ScopeKey,
        node: Mapping[str, object],
        *,
        loop_scope: bm.ScopeKey | None,
    ) -> tuple[NodeExecution | None, _LoopHalt | None]:
        node_id = str(node["id"])
        tool = str(node["tool"])
        adapter = self._adapters.get(tool)
        if adapter is None:
            raise KeyError(f"unknown adapter for tool '{tool}'")

        node_scope = bm.ScopeKey(scope_type="node", scope_id=node_id)
        self._budgets.enter_scope(node_scope)
        self._trace.emit(
            "node_start",
            scope_type="node",
            scope_id=node_id,
            payload={"tool": tool},
        )

        node_policy = node.get("policy")
        policy_scope = f"node:{node_id}"
        policy_active = False

        try:
            if node_policy is not None:
                self._policies.push(node_policy, scope=policy_scope, source=node_id)
                policy_active = True

            self._policies.effective_allowlist([tool])
            self._policies.enforce(tool)

            cost_snapshot = bm.CostSnapshot.from_raw(adapter.estimate_cost(node))

            run_decision = self._preview_scope(run_scope, cost_snapshot)
            if run_decision.should_stop:
                self._commit_scope(run_decision)
            node_decision = self._preview_scope(node_scope, cost_snapshot)
            if node_decision.should_stop:
                self._commit_scope(node_decision)

            loop_decision: bm.BudgetDecision | None = None
            if loop_scope is not None:
                loop_decision = self._preview_scope(loop_scope, cost_snapshot)
                if loop_decision.should_stop:
                    return None, _LoopHalt(decision=loop_decision)

            self._commit_scope(run_decision)
            self._commit_scope(node_decision)
            if loop_decision is not None:
                self._commit_scope(loop_decision)

            result = adapter.execute(node)
            execution = NodeExecution(node_id=node_id, tool=tool, result=result)
            self._trace.emit(
                "node_complete",
                scope_type="node",
                scope_id=node_id,
                payload={"tool": tool},
            )
            return execution, None
        except BudgetBreachError:
            raise
        except PolicyViolationError as exc:
            self._trace.emit(
                "policy_violation",
                scope_type="node",
                scope_id=node_id,
                payload={"tool": tool, "error": str(exc)},
            )
            raise
        finally:
            if policy_active:
                self._policies.pop(policy_scope)
            self._budgets.exit_scope(node_scope)

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------
    def _preview_scope(
        self, scope: bm.ScopeKey, cost: bm.CostSnapshot
    ) -> bm.BudgetDecision:
        decision = self._budgets.preview_charge(scope, cost)
        if decision.breached:
            self._budgets.record_breach(decision)
        return decision

    def _commit_scope(self, decision: bm.BudgetDecision) -> None:
        try:
            self._budgets.commit_charge(decision)
        except BudgetBreachError as exc:
            if not hasattr(exc, "decision"):
                setattr(exc, "decision", decision)
            raise
