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
                node_id = str(raw_node["id"])
                tool = str(raw_node["tool"])
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
                try:
                    self._policies.enforce(tool)
                    cost_snapshot = bm.CostSnapshot.from_raw(adapter.estimate_cost(raw_node))
                    self._apply_budget(run_scope, cost_snapshot)
                    self._apply_budget(node_scope, cost_snapshot)
                    result = adapter.execute(raw_node)
                    executions.append(
                        NodeExecution(node_id=node_id, tool=tool, result=result)
                    )
                    self._trace.emit(
                        "node_complete",
                        scope_type="node",
                        scope_id=node_id,
                        payload={"tool": tool},
                    )
                except PolicyViolationError as exc:
                    self._trace.emit(
                        "policy_violation",
                        scope_type="node",
                        scope_id=node_id,
                        payload={"tool": tool, "error": str(exc)},
                    )
                    raise
                finally:
                    self._budgets.exit_scope(node_scope)
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
