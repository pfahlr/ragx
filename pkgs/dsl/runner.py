"""FlowRunner implementation with budget guards and policy enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Protocol, Sequence
from uuid import uuid4

from .budget import BudgetBreachError, CostSnapshot
from .budget_manager import BudgetManager, BudgetScope
from .policy import PolicyStack, PolicyViolationError
from .trace import TraceEventEmitter

__all__ = [
    "FlowRunner",
    "RunResult",
    "ToolAdapter",
    "ToolExecutionResult",
]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Result returned by a tool adapter execution."""

    outputs: Mapping[str, object]
    cost: CostSnapshot


class ToolAdapter(Protocol):
    """Adapter protocol used by the runner to execute nodes."""

    def estimate_cost(self, node_spec: Mapping[str, object], context: Mapping[str, object]) -> CostSnapshot:  # noqa: D401
        """Return a conservative cost estimate for preflight checks."""

    def execute(self, node_spec: Mapping[str, object], context: Mapping[str, object]) -> ToolExecutionResult:  # noqa: D401
        """Execute the tool and return structured outputs plus actual cost."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured result returned by :class:`FlowRunner.run`."""

    run_id: str
    status: str
    outputs: Mapping[str, Mapping[str, object]]
    stop_reasons: tuple[str, ...]


class FlowRunner:
    """Execute a simple flow spec with budget and policy guards."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        trace: TraceEventEmitter | None = None,
    ) -> None:
        self._adapters: Dict[str, ToolAdapter] = dict(adapters)
        self._trace = trace or TraceEventEmitter()
        self._budget_manager: BudgetManager | None = None
        self._policy_stack: PolicyStack | None = None
        self._node_map: Mapping[str, Mapping[str, object]] = {}
        self._sequence: Sequence[str] = ()
        self._vars: Mapping[str, object] = {}
        self._outputs: MutableMapping[str, Mapping[str, object]] = {}
        self._stop_reasons: list[str] = []
        self._status: str = "ok"
        self._current_node: str | None = None
        self._run_id: str = uuid4().hex

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, *, spec: Mapping[str, object], vars: Mapping[str, object]) -> RunResult:
        self._reset_state(vars)
        globals_cfg = spec.get("globals", {})
        tools = globals_cfg.get("tools", {})
        tool_sets = globals_cfg.get("tool_sets", {})
        self._policy_stack = PolicyStack(tools=tools, tool_sets=tool_sets)
        run_budget = globals_cfg.get("run_budget")
        self._budget_manager = BudgetManager(run_spec=run_budget, trace=self._trace)

        graph = spec.get("graph", {})
        self._sequence = tuple(graph.get("sequence", []))
        nodes_mapping = graph.get("nodes", {})
        self._node_map = {
            node_id: node_spec for node_id, node_spec in nodes_mapping.items()
        }

        try:
            self._execute_sequence()
        except BudgetBreachError as exc:
            self._handle_budget_breach(exc)
        except PolicyViolationError as exc:
            self._handle_policy_violation(exc)

        return RunResult(
            run_id=self._run_id,
            status=self._status,
            outputs={node_id: dict(payload) for node_id, payload in self._outputs.items()},
            stop_reasons=tuple(self._stop_reasons),
        )

    @property
    def trace(self) -> TraceEventEmitter:
        return self._trace

    # ------------------------------------------------------------------
    # Internal execution helpers
    # ------------------------------------------------------------------
    def _reset_state(self, vars: Mapping[str, object]) -> None:
        self._vars = dict(vars)
        self._outputs = {}
        self._stop_reasons = []
        self._status = "ok"
        self._current_node = None
        self._run_id = uuid4().hex

    def _execute_sequence(self) -> None:
        for node_id in self._sequence:
            if self._status != "ok":
                break
            node = self._node_map.get(node_id)
            if node is None:
                raise KeyError(f"unknown node id {node_id!r}")
            self._current_node = node_id
            kind = node.get("kind")
            if kind == "unit":
                self._execute_unit(node_id, node)
            elif kind == "loop":
                self._execute_loop(node_id, node)
            else:
                # Transform/decision nodes are out of scope for this phase.
                continue

    def _execute_unit(self, node_id: str, node: Mapping[str, object]) -> None:
        assert self._budget_manager is not None
        assert self._policy_stack is not None

        spec = node.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError(f"node {node_id!r} missing spec")
        tool_ref = spec.get("tool_ref")
        if not isinstance(tool_ref, str):
            raise ValueError(f"node {node_id!r} missing tool_ref")

        policy_cfg = node.get("policy")
        if policy_cfg:
            self._policy_stack.push(policy_cfg, scope=f"node:{node_id}", source=node_id)
        try:
            self._policy_stack.enforce(tool_ref)
            adapter = self._resolve_adapter(tool_ref)
            context = {
                "vars": self._vars,
                "outputs": self._outputs,
                "node_id": node_id,
            }
            estimate = adapter.estimate_cost(spec, context)
            scopes = self._collect_scopes(node_id, node)
            self._preflight_cost(scopes, estimate)
            result = adapter.execute(spec, context)
            self._outputs[node_id] = dict(result.outputs)
            decisions = self._commit_cost(scopes, result.cost)
            if any(decision.should_stop for decision in decisions):
                self._status = "halted"
        finally:
            if policy_cfg:
                self._policy_stack.pop(expected_scope=f"node:{node_id}")

    def _execute_loop(self, node_id: str, loop_node: Mapping[str, object]) -> None:
        assert self._budget_manager is not None
        policy_cfg = loop_node.get("policy")
        if self._policy_stack is None:
            raise RuntimeError("policy stack not initialised")
        if policy_cfg:
            self._policy_stack.push(policy_cfg, scope=f"loop:{node_id}", source=node_id)

        stop_cfg = loop_node.get("stop", {})
        loop_budget = stop_cfg.get("budget")
        loop_scope = self._budget_manager.push_loop(node_id, loop_budget)
        max_iterations = stop_cfg.get("max_iterations")
        target_ids = list(loop_node.get("target_subgraph", []))

        iteration = 0
        try:
            while self._status == "ok":
                if max_iterations is not None and iteration >= int(max_iterations):
                    self._status = "halted"
                    self._stop_reasons.append(f"loop:{node_id}:max_iterations")
                    break
                for target_id in target_ids:
                    self._current_node = target_id
                    inner = self._node_map.get(target_id)
                    if inner is None:
                        raise KeyError(f"unknown loop target {target_id!r}")
                    kind = inner.get("kind")
                    if kind == "unit":
                        self._execute_unit(target_id, inner)
                    elif kind == "loop":
                        self._execute_loop(target_id, inner)
                    if self._status != "ok":
                        break
                iteration += 1
                if self._status != "ok":
                    break
            if any(reason.startswith(f"loop:{node_id}:budget_stop") for reason in self._stop_reasons):
                self._status = "halted"
        finally:
            self._budget_manager.pop_loop(node_id)
            if policy_cfg:
                self._policy_stack.pop(expected_scope=f"loop:{node_id}")

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------
    def _collect_scopes(self, node_id: str, node: Mapping[str, object]) -> list[BudgetScope]:
        assert self._budget_manager is not None
        scopes: list[BudgetScope] = []
        run_scope = BudgetScope("run", "run")
        if self._budget_manager.has_scope(run_scope):
            scopes.append(run_scope)
        scopes.extend(self._budget_manager.loop_stack)

        hard_budget = node.get("budget")
        if hard_budget:
            scope = BudgetScope("node", node_id)
            if not self._budget_manager.has_scope(scope):
                self._budget_manager.configure_scope(scope, hard_budget)
            scopes.append(scope)

        spec = node.get("spec")
        soft_budget = spec.get("budget") if isinstance(spec, Mapping) else None
        if soft_budget:
            scope = BudgetScope("node_soft", node_id)
            if not self._budget_manager.has_scope(scope):
                self._budget_manager.configure_scope(scope, soft_budget)
            scopes.append(scope)
        return scopes

    def _preflight_cost(self, scopes: Sequence[BudgetScope], cost: CostSnapshot) -> None:
        for scope in scopes:
            self._budget_manager.preflight(scope, cost)

    def _commit_cost(self, scopes: Sequence[BudgetScope], cost: CostSnapshot) -> list:
        decisions = []
        for scope in scopes:
            decision = self._budget_manager.commit(scope, cost)
            decisions.append(decision)
            if decision.should_stop:
                reason = f"{scope.scope_type}:{scope.scope_id}:budget_stop"
                if reason not in self._stop_reasons:
                    self._stop_reasons.append(reason)
        return decisions

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def _handle_budget_breach(self, exc: BudgetBreachError) -> None:
        decision = exc.decision
        if decision.should_stop:
            self._status = "halted"
            reason = f"{decision.scope_type}:{decision.scope_id}:budget_stop"
        else:
            self._status = "error"
            reason = f"{decision.scope_type}:{decision.scope_id}:budget_error"
        if reason not in self._stop_reasons:
            self._stop_reasons.append(reason)

    def _handle_policy_violation(self, exc: PolicyViolationError) -> None:
        self._status = "error"
        tool = getattr(exc.denial, "tool", "unknown")
        scope = self._current_node or "graph"
        reason = f"policy:{scope}:{tool}"
        if reason not in self._stop_reasons:
            self._stop_reasons.append(reason)

    def _resolve_adapter(self, tool_ref: str) -> ToolAdapter:
        adapter = self._adapters.get(tool_ref)
        if adapter is None:
            self._status = "error"
            reason = f"adapter:{self._current_node or 'node'}:missing"
            if reason not in self._stop_reasons:
                self._stop_reasons.append(reason)
            raise RuntimeError(f"no adapter registered for tool {tool_ref!r}")
        return adapter
