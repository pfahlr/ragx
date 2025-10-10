"""FlowRunner prototype integrating budget guards and trace emission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional

from .budget import BudgetBreachError, BudgetStopSignal
from .budget_manager import BudgetManager
from .costs import normalize_cost
from .policy import PolicyStack, PolicyViolationError
from .trace import TraceEventEmitter


@dataclass
class RunResult:
    run_id: str
    status: str
    outputs: MutableMapping[str, object] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class FlowRunner:
    """Minimal FlowRunner implementation sufficient for budget integration tests."""

    def __init__(
        self,
        adapters: Mapping[str, object],
        trace_emitter: TraceEventEmitter,
        id_factory: Optional[Callable[[], str]] = None,
        budget_manager_factory: Callable[..., BudgetManager] | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._trace = trace_emitter
        self._id_factory = id_factory or (lambda: "run")
        self._budget_manager_factory = budget_manager_factory or BudgetManager

    def run(self, spec: Mapping[str, object], vars: Mapping[str, object]) -> RunResult:  # noqa: ARG002
        run_id = self._id_factory()
        budget_manager = self._budget_manager_factory(trace_writer=self._trace)
        result = RunResult(run_id=run_id, status="ok")

        run_budget = spec.get("run_budget")
        if run_budget:
            budget_manager.register_scope("run", run_budget)

        policy_spec = spec.get("policies", {})
        policy_stack = PolicyStack(
            global_allow=policy_spec.get("allow"),
            global_deny=policy_spec.get("deny"),
        )

        self._trace.emit("run_start", flow_id=spec.get("flow_id"), run_id=run_id)

        try:
            for node in spec.get("nodes", []):
                kind = node.get("kind", "unit")
                if kind == "loop":
                    result.outputs[node["id"]] = self._execute_loop(node, policy_stack, budget_manager, result)
                elif kind == "unit":
                    output, warnings = self._execute_unit(node, policy_stack, budget_manager, parent_scopes=None)
                    result.outputs[node["id"]] = output
                    result.warnings.extend(warnings)
                else:
                    raise NotImplementedError(f"Unsupported node kind: {kind}")
        except BudgetBreachError as breach:
            result.status = "halted"
            result.stop_reason = f"budget_breach:{breach.outcome.scope_id}"
        except PolicyViolationError as violation:
            result.status = "error"
            result.stop_reason = f"policy_violation:{violation.node_id}"
            raise
        finally:
            self._trace.emit("run_end", run_id=run_id, status=result.status)

        return result

    def _execute_unit(
        self,
        node: Mapping[str, object],
        policy_stack: PolicyStack,
        budget_manager: BudgetManager,
        parent_scopes: Optional[List[str]],
    ) -> tuple[Mapping[str, object], List[str]]:
        node_id = node["id"]
        node_spec = node.get("spec", {})
        tool_ref = node_spec.get("tool_ref")
        allow = policy_stack.resolve(node_id=node_id, tool_ref=tool_ref, node_policy=node.get("policy"))
        deny = node.get("policy", {}).get("deny", []) if node.get("policy") else []
        self._trace.emit_policy_resolution(node_id=node_id, allow=allow, deny=deny)

        if tool_ref not in self._adapters:
            raise KeyError(f"Adapter for tool '{tool_ref}' not registered")
        adapter = self._adapters[tool_ref]

        adapter.estimate(node_spec)
        outputs, raw_cost = adapter.execute(node_spec)
        normalized_cost = normalize_cost(raw_cost)

        scopes: List[str] = []
        if "run" in budget_manager.scopes():
            scopes.append("run")
        if parent_scopes:
            scopes.extend(parent_scopes)

        node_scope: Optional[str] = None
        if node.get("budget"):
            node_scope = f"node:{node_id}"
            if node_scope not in budget_manager.scopes():
                budget_manager.register_scope(node_scope, node["budget"])
            scopes.append(node_scope)

        warnings: List[str] = []
        for scope_id in scopes:
            budget_manager.charge(scope_id, normalized_cost)
            warnings.extend(budget_manager.consume_warnings(scope_id))
        if node_scope:
            warnings.extend(budget_manager.consume_warnings(node_scope))

        return outputs, warnings

    def _execute_loop(
        self,
        node: Mapping[str, object],
        policy_stack: PolicyStack,
        budget_manager: BudgetManager,
        result: RunResult,
    ) -> List[Mapping[str, object]]:
        node_id = node["id"]
        loop_scope = f"loop:{node_id}"
        stop_spec = node.get("stop", {})
        loop_budget = stop_spec.get("budget")
        if loop_budget and loop_scope not in budget_manager.scopes():
            budget_manager.register_scope(loop_scope, loop_budget)

        max_iterations = stop_spec.get("max_iterations", 100)
        body = node.get("body", [])
        iterations: List[Mapping[str, object]] = []

        for iteration_index in range(max_iterations):
            try:
                iteration_outputs: Dict[str, object] = {}
                for body_node in body:
                    output, warnings = self._execute_unit(
                        body_node,
                        policy_stack,
                        budget_manager,
                        parent_scopes=[loop_scope] if loop_budget else None,
                    )
                    iteration_outputs[body_node["id"]] = output
                    result.warnings.extend(warnings)
                iterations.append(iteration_outputs)
            except BudgetStopSignal:
                result.warnings.extend(budget_manager.consume_warnings(loop_scope))
                break
        else:
            # loop exhausted max iterations without stop signal
            pass

        return iterations
