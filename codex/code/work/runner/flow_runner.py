"""Adapter-backed FlowRunner with budget enforcement."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Protocol, Sequence
from uuid import uuid4

from pkgs.dsl.policy import PolicyStack, PolicyViolationError

from .budgeting import BudgetScope, BudgetSpec, CostSnapshot
from .manager import BudgetManager
from .trace import TraceEventEmitter

__all__ = ["FlowRunner", "RunResult"]


class ToolAdapter(Protocol):
    """Protocol implemented by tool adapters used in tests."""

    def estimate_cost(self, inputs: Mapping[str, object]) -> CostSnapshot: ...

    def execute(
        self, inputs: Mapping[str, object]
    ) -> tuple[Mapping[str, object], CostSnapshot]: ...


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a FlowRunner run."""

    run_id: str
    status: str
    outputs: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class _NodeExecutionResult:
    executed: bool
    halted: bool
    reason: str | None = None
    loop_scope: BudgetScope | None = None


class FlowRunner:
    """Minimal FlowRunner integrating policies, budgets, and adapters."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        policy_stack: PolicyStack,
        trace_emitter: TraceEventEmitter | None = None,
        budget_manager: BudgetManager | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._policy_stack = policy_stack
        self._trace = trace_emitter or TraceEventEmitter()
        self._budget_manager = budget_manager or BudgetManager(trace_emitter=self._trace)
        self._outputs: MutableMapping[str, Mapping[str, object]] = {}
        self._status = "ok"
        self._run_scope = BudgetScope(scope_type="run", identifier="run")
        self._run_id = uuid4().hex
        self._loop_stack: list[BudgetScope] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        spec: Mapping[str, object],
        *,
        run_vars: Mapping[str, object] | None = None,
    ) -> RunResult:
        self._outputs = {}
        self._status = "ok"
        self._loop_stack = []
        self._run_id = spec.get("run_id") or uuid4().hex
        flow_id = str(spec.get("id", self._run_id))
        self._run_scope = BudgetScope(scope_type="run", identifier=flow_id)

        run_budget = spec.get("run_budget")
        if isinstance(run_budget, Mapping):
            self._budget_manager.register(
                BudgetSpec.from_dict(self._run_scope, run_budget)
            )

        with self._policy_scope(spec.get("policy"), scope="run", source=flow_id):
            for node in spec.get("nodes", []):
                kind = node.get("kind")
                if kind == "unit":
                    result = self._execute_unit(node, run_vars or {})
                    if result.halted and result.loop_scope is None:
                        self._status = "halted"
                        break
                elif kind == "loop":
                    self._execute_loop(node, run_vars or {})
                    if self._status != "ok":
                        break
                else:
                    raise ValueError(f"unsupported node kind: {kind}")

        return RunResult(
            run_id=self._run_id,
            status=self._status,
            outputs=dict(self._outputs),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_unit(
        self,
        node: Mapping[str, object],
        run_vars: Mapping[str, object],
        *,
        additional_scopes: Sequence[BudgetScope] | None = None,
    ) -> _NodeExecutionResult:
        node_id = str(node.get("id"))
        tool_name = str(node.get("tool"))
        adapter = self._adapters.get(tool_name)
        if adapter is None:
            raise KeyError(f"unknown adapter for tool '{tool_name}'")

        node_scope = BudgetScope(scope_type="node", identifier=node_id)
        node_budget = node.get("budget") if isinstance(node.get("budget"), Mapping) else None
        if node_budget:
            self._budget_manager.register(BudgetSpec.from_dict(node_scope, node_budget))

        scopes: list[BudgetScope] = [self._run_scope]
        if additional_scopes:
            scopes.extend(additional_scopes)
        if node_budget:
            scopes.append(node_scope)

        with self._policy_scope(node.get("policy"), scope=f"node:{node_id}", source=node_id):
            resolution = self._policy_stack.effective_allowlist([tool_name])
            if tool_name not in resolution.allowed:
                try:
                    self._policy_stack.enforce(tool_name)
                except PolicyViolationError:
                    self._status = "error"
                    return _NodeExecutionResult(False, True, "policy_violation", None)

            estimate = adapter.estimate_cost(self._build_adapter_inputs(run_vars))
            decision = self._budget_manager.preflight(scopes, estimate)
            if decision.blocked:
                loop_scope = (additional_scopes or [])[-1] if additional_scopes else None
                if loop_scope is not None:
                    return _NodeExecutionResult(False, True, "budget_exhausted", loop_scope)
                self._status = "halted"
                return _NodeExecutionResult(False, True, "budget_exhausted", None)

            outputs, cost = adapter.execute(self._build_adapter_inputs(run_vars))
            self._budget_manager.commit(scopes, cost)
            self._outputs[node_id] = dict(outputs)
            return _NodeExecutionResult(True, False, None, None)

    def _execute_loop(
        self,
        loop_spec: Mapping[str, object],
        run_vars: Mapping[str, object],
    ) -> None:
        loop_id = str(loop_spec.get("id"))
        loop_scope = BudgetScope(scope_type="loop", identifier=loop_id)
        loop_budget = loop_spec.get("budget") if isinstance(loop_spec.get("budget"), Mapping) else None
        if loop_budget:
            self._budget_manager.register(BudgetSpec.from_dict(loop_scope, loop_budget))

        max_iterations = loop_spec.get("max_iterations")
        max_iterations_value = int(max_iterations) if max_iterations is not None else None
        iterations = 0
        body = loop_spec.get("body", [])
        with self._policy_scope(loop_spec.get("policy"), scope=f"loop:{loop_id}", source=loop_id):
            while self._status == "ok":
                if max_iterations_value is not None and iterations >= max_iterations_value:
                    break
                self._loop_stack.append(loop_scope)
                try:
                    for node in body:
                        result = self._execute_unit(
                            node,
                            run_vars,
                            additional_scopes=tuple(self._loop_stack),
                        )
                        if result.halted:
                            if (
                                result.reason == "budget_exhausted"
                                and result.loop_scope == loop_scope
                            ):
                                self._trace.emit_loop_stop(
                                    loop_scope,
                                    reason="budget_exhausted",
                                    iterations=iterations,
                                )
                                return
                            self._status = "halted"
                            return
                finally:
                    self._loop_stack.pop()
                iterations += 1

    def _build_adapter_inputs(
        self, run_vars: Mapping[str, object]
    ) -> Mapping[str, object]:  # pragma: no cover - trivial wrapper
        return {
            "vars": dict(run_vars),
            "outputs": dict(self._outputs),
        }

    @contextmanager
    def _policy_scope(
        self,
        policy: Mapping[str, object] | None,
        *,
        scope: str,
        source: str,
    ):
        if policy is not None:
            self._policy_stack.push(policy, scope=scope, source=source)
            try:
                yield
            finally:
                self._policy_stack.pop(scope)
        else:
            yield

    @property
    def trace(self) -> TraceEventEmitter:
        return self._trace
