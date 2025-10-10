"""Adapter-backed FlowRunner integrating budgets and policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pkgs.dsl.policy import PolicyStack, PolicyViolationError

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
    iteration: int | None = None
    loop_id: str | None = None


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
                    self._run_loop(run_scope, raw_node, executions)
                    continue
                execution = self._run_unit_node(
                    run_scope=run_scope,
                    raw_node=raw_node,
                    loop_scope=None,
                    loop_id=None,
                    iteration=None,
                )
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
    def _run_loop(
        self,
        run_scope: bm.ScopeKey,
        raw_loop: Mapping[str, object],
        executions: list[NodeExecution],
    ) -> None:
        loop_id = str(raw_loop["id"])
        body = list(raw_loop.get("body", []))
        stop_cfg = raw_loop.get("stop")
        max_iterations_value: int | None = None
        if isinstance(stop_cfg, Mapping):
            raw_limit = stop_cfg.get("max_iterations")
            if raw_limit is not None:
                max_iterations_value = int(raw_limit)
        if max_iterations_value is None and "max_iterations" in raw_loop:
            max_iterations_value = int(raw_loop.get("max_iterations", 0))
        loop_scope = bm.ScopeKey(scope_type="loop", scope_id=loop_id)
        self._budgets.enter_scope(loop_scope)
        unlimited = max_iterations_value is None
        self._trace.emit(
            "loop_start",
            scope_type="loop",
            scope_id=loop_id,
            payload={"max_iterations": max_iterations_value},
        )
        try:
            if not body:
                self._trace.emit(
                    "loop_complete",
                    scope_type="loop",
                    scope_id=loop_id,
                    payload={"iterations": 0},
                )
                return
            if max_iterations_value is not None and max_iterations_value <= 0:
                self._trace.emit(
                    "loop_complete",
                    scope_type="loop",
                    scope_id=loop_id,
                    payload={"iterations": 0},
                )
                return
            iteration = 1
            executed = 0
            while unlimited or iteration <= (max_iterations_value or 0):
                self._trace.emit(
                    "loop_iteration_start",
                    scope_type="loop",
                    scope_id=loop_id,
                    payload={"iteration": iteration},
                )
                for node in body:
                    try:
                        execution = self._run_unit_node(
                            run_scope=run_scope,
                            raw_node=node,
                            loop_scope=loop_scope,
                            loop_id=loop_id,
                            iteration=iteration,
                        )
                        executions.append(execution)
                    except BudgetBreachError as exc:
                        if exc.scope == loop_scope:
                            self._trace.emit(
                                "loop_stop",
                                scope_type="loop",
                                scope_id=loop_id,
                                payload={
                                    "reason": "budget_stop",
                                    "spec_name": exc.outcome.spec.name,
                                    "iteration": iteration,
                                },
                            )
                            return
                        raise
                executed = iteration
                self._trace.emit(
                    "loop_iteration_complete",
                    scope_type="loop",
                    scope_id=loop_id,
                    payload={"iteration": iteration},
                )
                iteration += 1
                if not unlimited and iteration > (max_iterations_value or 0):
                    break
            self._trace.emit(
                "loop_complete",
                scope_type="loop",
                scope_id=loop_id,
                payload={"iterations": executed},
            )
        finally:
            self._budgets.exit_scope(loop_scope)

    def _run_unit_node(
        self,
        *,
        run_scope: bm.ScopeKey,
        raw_node: Mapping[str, object],
        loop_scope: bm.ScopeKey | None,
        loop_id: str | None,
        iteration: int | None,
    ) -> NodeExecution:
        node_id = str(raw_node["id"])
        tool = str(raw_node["tool"])
        adapter = self._adapters.get(tool)
        if adapter is None:
            raise KeyError(f"unknown adapter for tool '{tool}'")
        scope_id = f"{node_id}@{iteration}" if iteration is not None else node_id
        node_scope = bm.ScopeKey(scope_type="node", scope_id=scope_id)
        self._budgets.enter_scope(node_scope)
        payload = {"tool": tool}
        if iteration is not None:
            payload["iteration"] = iteration
        if loop_id is not None:
            payload["loop_id"] = loop_id
        self._trace.emit(
            "node_start",
            scope_type="node",
            scope_id=scope_id,
            payload=payload,
        )
        node_payload = dict(raw_node)
        if iteration is not None:
            node_payload.setdefault("iteration", iteration)
        if loop_id is not None:
            node_payload.setdefault("loop_id", loop_id)
        try:
            self._policies.enforce(tool)
            cost_snapshot = bm.CostSnapshot.from_raw(
                adapter.estimate_cost(node_payload)
            )
            run_decision = self._apply_budget(
                run_scope, cost_snapshot, commit=False
            )
            loop_decision: bm.BudgetDecision | None = None
            if loop_scope is not None:
                loop_decision = self._apply_budget(
                    loop_scope, cost_snapshot, commit=False
                )
            node_decision = self._apply_budget(
                node_scope, cost_snapshot, commit=False
            )
            self._budgets.commit_charge(node_decision)
            if loop_decision is not None:
                self._budgets.commit_charge(loop_decision)
            self._budgets.commit_charge(run_decision)
            result = adapter.execute(node_payload)
            record = NodeExecution(
                node_id=node_id,
                tool=tool,
                result=result,
                iteration=iteration,
                loop_id=loop_id,
            )
            self._trace.emit(
                "node_complete",
                scope_type="node",
                scope_id=scope_id,
                payload=payload,
            )
            return record
        except PolicyViolationError as exc:
            self._trace.emit(
                "policy_violation",
                scope_type="node",
                scope_id=scope_id,
                payload={"tool": tool, "error": str(exc)},
            )
            raise
        finally:
            self._budgets.exit_scope(node_scope)

    def _apply_budget(
        self,
        scope: bm.ScopeKey,
        cost: bm.CostSnapshot,
        *,
        commit: bool = True,
    ) -> bm.BudgetDecision:
        decision = self._budgets.preview_charge(scope, cost)
        if decision.breached:
            self._budgets.record_breach(decision)
        if decision.should_stop:
            blocking = decision.blocking
            if blocking is None:  # pragma: no cover - defensive guard
                raise BudgetError("blocking outcome missing for stop decision")
            raise BudgetBreachError(scope, blocking)
        if commit:
            self._budgets.commit_charge(decision)
        return decision
