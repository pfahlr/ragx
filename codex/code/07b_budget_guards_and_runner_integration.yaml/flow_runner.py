"""FlowRunner integrating budget guards, policies, and adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pkgs.dsl.policy import PolicyStack, PolicyViolationError

from .budget_manager import BudgetBreachError, BudgetManager
from .budget_models import BudgetSpec
from .trace_emitter import TraceEventEmitter

__all__ = ["FlowRunner"]


class FlowRunner:
    """Execute flow nodes while enforcing policies and budgets."""

    def __init__(
        self,
        *,
        adapter: Any,
        policy_stack: PolicyStack,
        budget_manager: BudgetManager,
        emitter: TraceEventEmitter,
    ) -> None:
        self._adapter = adapter
        self._policy_stack = policy_stack
        self._budget_manager = budget_manager
        self._emitter = emitter

    def run(self, flow: Mapping[str, Any]) -> dict[str, Any]:
        run_scope_spec = flow.get("run_scope")
        if run_scope_spec is not None and not isinstance(run_scope_spec, BudgetSpec):
            raise TypeError("run_scope must be a BudgetSpec or None")
        run_scope_id = "run"
        self._budget_manager.enter_scope(
            scope_type="run",
            scope_id=run_scope_id,
            spec=run_scope_spec,
        )

        outputs: dict[str, Any] = {}
        nodes: Iterable[Mapping[str, Any]] = flow.get("nodes", [])
        try:
            for iteration, node in enumerate(nodes):
                node_id = str(node["id"])
                tool = str(node["tool"])
                node_scope_spec = node.get("budget")
                if node_scope_spec is not None and not isinstance(node_scope_spec, BudgetSpec):
                    raise TypeError("node budget must be a BudgetSpec or None")
                self._budget_manager.enter_scope(
                    scope_type="node",
                    scope_id=node_id,
                    spec=node_scope_spec,
                    parent_scope=run_scope_id,
                )
                try:
                    resolution = self._policy_stack.effective_allowlist([tool])
                    resolution_payload = {
                        "allowed": resolution.allowed,
                        "denied": resolution.denied,
                        "stack_depth": resolution.stack_depth,
                    }
                    self._emitter.emit_policy_resolved(
                        node_id=node_id,
                        resolution=resolution_payload,
                    )
                    try:
                        self._policy_stack.enforce(tool)
                    except PolicyViolationError as error:
                        self._emitter.emit_policy_violation(
                            node_id=node_id,
                            denial=error.denial,
                        )
                        self._emitter.emit_loop_stop(
                            scope="node",
                            node_id=node_id,
                            loop_iteration=iteration,
                            stop_reason="policy_violation",
                        )
                        raise

                    # Preview and commit budgets around execution.
                    estimate_preview = self._budget_manager.preview(
                        node_id,
                        self._adapter.estimate(dict(node)),
                    )
                    if estimate_preview.hard_breach:
                        breach_charge = next(
                            (charge for charge in estimate_preview.charges if charge.breached),
                            None,
                        )
                        self._emitter.emit_budget_breach(
                            node_id=node_id,
                            loop_iteration=iteration,
                            preview=estimate_preview,
                            charge=breach_charge,
                        )
                        self._emitter.emit_loop_stop(
                            scope="node",
                            node_id=node_id,
                            loop_iteration=iteration,
                            stop_reason="budget_breach",
                        )
                        raise BudgetBreachError(estimate_preview)

                    output, actual_cost = self._adapter.execute(dict(node))
                    commit_preview = self._budget_manager.preview(node_id, actual_cost)
                    try:
                        self._budget_manager.commit(
                            commit_preview,
                            node_id=node_id,
                            loop_iteration=iteration,
                        )
                    except BudgetBreachError:
                        self._emitter.emit_loop_stop(
                            scope="node",
                            node_id=node_id,
                            loop_iteration=iteration,
                            stop_reason="budget_breach",
                        )
                        raise
                    outputs[node_id] = output
                finally:
                    self._budget_manager.exit_scope(node_id)
        finally:
            self._budget_manager.exit_scope(run_scope_id)
        return outputs

