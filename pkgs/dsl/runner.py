"""FlowRunner with budget management and policy integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from .budget import BudgetManager, BudgetOutcome, BudgetScope, BudgetSpec
from .models import mapping_proxy
from .policy import PolicyStack, PolicyViolationError
from .trace import TraceEvent, TraceEventEmitter

__all__ = ["ToolAdapter", "RunResult", "FlowRunner"]


class ToolAdapter(ABC):
    """Abstract adapter executed by FlowRunner unit nodes."""

    @abstractmethod
    def estimate_cost(
        self, node_spec: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Mapping[str, float]:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self, node_spec: Mapping[str, Any], context: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, float]]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured result returned by :class:`FlowRunner.run`."""

    run_id: str
    status: str
    outputs: Mapping[str, tuple[Mapping[str, Any], ...]]
    warnings: tuple[Mapping[str, object], ...]
    stop_reason: Mapping[str, object] | None
    trace_events: tuple[TraceEvent, ...]


@dataclass(slots=True)
class _LoopContext:
    scope: BudgetScope | None
    iteration: int


@dataclass(slots=True)
class _RunHalted(Exception):
    reason: Mapping[str, object]


class FlowRunner:
    """Execute flow specifications with policy and budget enforcement."""

    def __init__(
        self,
        *,
        tool_adapters: Mapping[str, ToolAdapter],
        tool_registry: Mapping[str, Mapping[str, object]] | None = None,
        trace_emitter: TraceEventEmitter | None = None,
        run_id_factory: Callable[[], str] | None = None,
        enforce_budgets: bool = True,
    ) -> None:
        self._tool_adapters = dict(tool_adapters)
        self._tool_registry = {name: dict(definition) for name, definition in (tool_registry or {}).items()}
        self._trace = trace_emitter or TraceEventEmitter()
        self._run_id_factory = run_id_factory or (lambda: uuid4().hex)
        self._enforce_budgets = enforce_budgets
        self._run_scope: BudgetScope | None = None
        self._node_scopes: dict[str, BudgetScope] = {}
        self._spec_scopes: dict[str, BudgetScope] = {}
        self._loop_scopes: dict[str, BudgetScope] = {}

    def run(self, spec: Mapping[str, Any], vars: Mapping[str, Any]) -> RunResult:
        run_id = self._run_id_factory()
        self._trace.reset()
        self._run_scope = None
        self._node_scopes.clear()
        self._spec_scopes.clear()
        self._loop_scopes.clear()
        manager = BudgetManager()
        self._register_budgets(manager, spec, run_id)
        policy_stack = PolicyStack(tools=self._tool_registry)
        outputs: dict[str, list[Mapping[str, Any]]] = {}
        warnings: list[Mapping[str, object]] = []

        try:
            self._execute_graph(
                spec.get("graph", []),
                manager,
                policy_stack,
                outputs,
                warnings,
                vars,
                loop_context=None,
                run_id=run_id,
            )
            status = "ok"
            stop_reason = None
        except _RunHalted as halted:
            status = "halted"
            stop_reason = halted.reason
        except PolicyViolationError as exc:
            status = "error"
            stop_reason = mapping_proxy({"reason": "policy_violation", "detail": str(exc)})
        immutable_outputs = {key: tuple(values) for key, values in outputs.items()}
        immutable_warnings = tuple(warnings)
        immutable_stop = (
            mapping_proxy(dict(stop_reason)) if isinstance(stop_reason, Mapping) else stop_reason
        )
        return RunResult(
            run_id=run_id,
            status=status,
            outputs=mapping_proxy(immutable_outputs),
            warnings=immutable_warnings,
            stop_reason=immutable_stop,
            trace_events=self._trace.events,
        )

    # ------------------------------------------------------------------
    # Budget registration
    # ------------------------------------------------------------------
    def _register_budgets(
        self, manager: BudgetManager, spec: Mapping[str, Any], run_id: str
    ) -> None:
        run_budget = (
            spec.get("run", {}).get("budget")
            if isinstance(spec.get("run"), Mapping)
            else None
        )
        if run_budget:
            run_scope = BudgetScope.run(run_id)
            manager.register(run_scope, BudgetSpec.from_mapping(run_budget, name="run"))
            self._run_scope = run_scope

        for node in spec.get("graph", []):
            self._register_node_budgets(manager, node)

    def _register_node_budgets(self, manager: BudgetManager, node: Mapping[str, Any]) -> None:
        node_id = str(node.get("id"))
        node_budget = node.get("budget")
        if isinstance(node_budget, Mapping):
            scope = BudgetScope.node(node_id)
            manager.register(scope, BudgetSpec.from_mapping(node_budget, name=node_id))
            self._node_scopes[node_id] = scope

        spec_budget = node.get("spec", {}).get("budget") if isinstance(node.get("spec"), Mapping) else None
        if isinstance(spec_budget, Mapping):
            scope = BudgetScope.spec(f"{node_id}:spec")
            manager.register(scope, BudgetSpec.from_mapping(spec_budget, name=f"{node_id}:spec"))
            self._spec_scopes[node_id] = scope

        if node.get("kind") == "loop":
            stop_conf = node.get("stop", {}) if isinstance(node.get("stop"), Mapping) else {}
            loop_budget = stop_conf.get("budget") if isinstance(stop_conf, Mapping) else None
            if isinstance(loop_budget, Mapping):
                scope = BudgetScope.loop(node_id)
                manager.register(scope, BudgetSpec.from_mapping(loop_budget, name=node_id))
                self._loop_scopes[node_id] = scope
            for child in node.get("target", []):
                self._register_node_budgets(manager, child)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_graph(
        self,
        nodes: Iterable[Mapping[str, Any]],
        manager: BudgetManager,
        policy_stack: PolicyStack,
        outputs: MutableMapping[str, list[Mapping[str, Any]]],
        warnings: list[Mapping[str, object]],
        vars: Mapping[str, Any],
        *,
        loop_context: _LoopContext | None,
        run_id: str,
    ) -> None:
        for node in nodes:
            kind = node.get("kind")
            if kind == "unit":
                self._execute_unit(
                    node,
                    manager,
                    policy_stack,
                    outputs,
                    warnings,
                    vars,
                    loop_context=loop_context,
                    run_id=run_id,
                )
            elif kind == "loop":
                self._execute_loop(
                    node,
                    manager,
                    policy_stack,
                    outputs,
                    warnings,
                    vars,
                    run_id=run_id,
                )
            else:
                raise ValueError(f"unsupported node kind: {kind}")

    def _execute_loop(
        self,
        node: Mapping[str, Any],
        manager: BudgetManager,
        policy_stack: PolicyStack,
        outputs: MutableMapping[str, list[Mapping[str, Any]]],
        warnings: list[Mapping[str, object]],
        vars: Mapping[str, Any],
        *,
        run_id: str,
    ) -> None:
        loop_id = str(node.get("id"))
        scope = self._loop_scopes.get(loop_id)
        max_iterations = None
        stop_conf = node.get("stop", {}) if isinstance(node.get("stop"), Mapping) else {}
        if isinstance(stop_conf, Mapping):
            max_iterations = stop_conf.get("max_iterations")
        body = node.get("target", [])
        if not isinstance(body, Sequence):
            raise ValueError("loop target must be a sequence of nodes")

        iteration = 0
        while True:
            iteration += 1
            context = _LoopContext(scope=scope, iteration=iteration)
            try:
                self._execute_graph(
                    body,
                    manager,
                    policy_stack,
                    outputs,
                    warnings,
                    vars,
                    loop_context=context,
                    run_id=run_id,
                )
            except _RunHalted as halted:
                raise halted
            if max_iterations is not None and iteration >= int(max_iterations):
                break
            if scope is None and max_iterations is None:
                break  # avoid infinite loops when no stop conditions are provided

    def _execute_unit(
        self,
        node: Mapping[str, Any],
        manager: BudgetManager,
        policy_stack: PolicyStack,
        outputs: MutableMapping[str, list[Mapping[str, Any]]],
        warnings: list[Mapping[str, object]],
        vars: Mapping[str, Any],
        *,
        loop_context: _LoopContext | None,
        run_id: str,
    ) -> None:
        node_id = str(node.get("id"))
        spec = node.get("spec", {})
        if not isinstance(spec, Mapping):
            raise ValueError("unit node spec must be a mapping")
        tool_ref = spec.get("tool_ref")
        if not isinstance(tool_ref, str):
            raise ValueError("unit node requires a tool_ref")
        adapter = self._tool_adapters.get(tool_ref)
        if adapter is None:
            raise ValueError(f"no adapter registered for tool '{tool_ref}'")

        resolution = policy_stack.enforce(tool_ref)
        self._trace.policy_resolved(
            node_id,
            resolution,
            run_id=run_id,
            iteration=loop_context.iteration if loop_context else None,
        )

        scope_chain = self._gather_scopes(node_id, loop_context)
        context = {
            "vars": vars,
            "node_id": node_id,
            "loop_iteration": loop_context.iteration if loop_context else None,
        }

        estimate = adapter.estimate_cost(spec, context)
        preview = manager.preview(estimate, scope_chain)
        self._emit_breaches(preview, loop_context, run_id)
        self._record_warnings(preview, warnings)
        if preview.should_stop and self._enforce_budgets:
            reason = self._build_stop_reason(preview)
            raise _RunHalted(reason)

        outputs_map, actual_cost = adapter.execute(spec, context)
        commit = manager.commit(actual_cost, scope_chain)
        self._emit_charges(commit, loop_context, run_id)
        self._record_warnings(commit, warnings)

        outputs.setdefault(node_id, []).append(dict(outputs_map))

    def _gather_scopes(
        self, node_id: str, loop_context: _LoopContext | None
    ) -> list[BudgetScope]:
        scopes: list[BudgetScope] = []
        if self._run_scope is not None:
            scopes.append(self._run_scope)
        if loop_context and loop_context.scope is not None:
            scopes.append(loop_context.scope)
        if node_id in self._node_scopes:
            scopes.append(self._node_scopes[node_id])
        if node_id in self._spec_scopes:
            scopes.append(self._spec_scopes[node_id])
        return scopes

    def _emit_breaches(
        self, outcome: BudgetOutcome, loop_context: _LoopContext | None, run_id: str
    ) -> None:
        if not outcome.charges:
            return
        iteration = loop_context.iteration if loop_context else None
        for charge in outcome.charges:
            if charge.breached:
                stop_reason = "budget_stop" if charge.action == "stop" else "budget_warn"
                self._trace.budget_breach(
                    charge,
                    loop_iteration=iteration,
                    run_id=run_id,
                    stop_reason=stop_reason,
                )

    def _emit_charges(
        self, outcome: BudgetOutcome, loop_context: _LoopContext | None, run_id: str
    ) -> None:
        iteration = loop_context.iteration if loop_context else None
        for charge in outcome.charges:
            self._trace.budget_charge(
                charge,
                loop_iteration=iteration,
                run_id=run_id,
            )
            if charge.breached:
                stop_reason = "budget_stop" if charge.action == "stop" else "budget_warn"
                self._trace.budget_breach(
                    charge,
                    loop_iteration=iteration,
                    run_id=run_id,
                    stop_reason=stop_reason,
                )

    def _record_warnings(
        self, outcome: BudgetOutcome, warnings: list[Mapping[str, object]]
    ) -> None:
        for charge in outcome.warnings:
            warnings.append(
                mapping_proxy(
                    {
                        "scope": charge.scope.kind,
                        "id": charge.scope.identifier,
                        "breach_action": charge.action,
                    }
                )
            )

    def _build_stop_reason(self, outcome: BudgetOutcome) -> Mapping[str, object]:
        for charge in outcome.charges:
            if charge.breached and charge.action == "stop":
                return mapping_proxy(
                    {
                        "scope": charge.scope.kind,
                        "id": charge.scope.identifier,
                        "reason": "budget_stop",
                    }
                )
        return mapping_proxy({"reason": "budget_stop"})
