"""Deterministic FlowRunner prototype with budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from ..adapters.base import ToolAdapter
from ..budget.manager import BudgetManager
from ..budget.models import BudgetSpec, CostSnapshot
from ..trace.emitter import TraceEventEmitter

__all__ = ["RunResult", "FlowRunner"]


@dataclass(slots=True)
class RunResult:
    """Result object returned by :meth:`FlowRunner.run`."""

    run_id: str
    status: str
    outputs: dict[str, object]
    stop_reason: dict[str, object] | None = None


@dataclass(slots=True)
class _ExecutionOutcome:
    stop: bool
    stop_reason: dict[str, object] | None


class FlowRunner:
    """Simplified flow runner that emphasises budget + policy integration."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        budget_manager: BudgetManager | None = None,
        policy_adapter: "PolicyAdapter | None" = None,
        trace: TraceEventEmitter | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._adapters = dict(adapters)
        self._trace = trace if trace is not None else TraceEventEmitter()
        self._budget_manager = (
            budget_manager if budget_manager is not None else BudgetManager(emitter=self._trace)
        )
        self._policy_adapter = policy_adapter
        self._run_id_factory = run_id_factory or (lambda: "run")
        self._loop_stack: list[str] = []
        self._run_scope_id: str | None = None

    # ------------------------------------------------------------------
    def run(self, flow_spec: Mapping[str, object], variables: Mapping[str, object]) -> RunResult:
        run_id = str(flow_spec.get("id") or self._run_id_factory())
        self._run_scope_id = run_id
        outputs: dict[str, object] = {}
        status = "ok"
        stop_reason: dict[str, object] | None = None

        run_budget = flow_spec.get("budget")
        if run_budget is not None:
            self._budget_manager.register_scope(
                scope_type="run",
                scope_id=run_id,
                spec=run_budget
                if isinstance(run_budget, (Mapping, BudgetSpec))
                else None,
            )

        for node in flow_spec.get("nodes", []):
            node_type = node.get("type")
            if node_type == "unit":
                outcome = self._execute_unit(node, outputs, variables)
            elif node_type == "loop":
                outcome = self._execute_loop(node, outputs, variables)
            else:
                raise ValueError(f"unsupported node type: {node_type!r}")
            if outcome.stop:
                status = "halted"
                stop_reason = outcome.stop_reason
                break

        return RunResult(run_id=run_id, status=status, outputs=outputs, stop_reason=stop_reason)

    # ------------------------------------------------------------------
    def _execute_unit(
        self,
        node: Mapping[str, object],
        outputs: MutableMapping[str, object],
        variables: Mapping[str, object],
    ) -> _ExecutionOutcome:
        node_id = str(node.get("id"))
        tool = str(node.get("tool"))
        adapter = self._adapters[tool]
        inputs = self._materialize_inputs(node.get("inputs", {}), variables)

        node_budget = node.get("budget")
        if node_budget is not None:
            self._budget_manager.register_scope(
                scope_type="node",
                scope_id=node_id,
                spec=node_budget
                if isinstance(node_budget, (Mapping, BudgetSpec))
                else None,
            )

        tool_chain = [tool]
        allowed = self._resolve_policy(node_id, tool_chain)
        if tool not in allowed:
            raise RuntimeError(f"tool {tool!r} blocked by policy")

        estimate = adapter.estimate_cost(inputs)
        preflight = self._apply_budget_checks(
            scopes=self._iter_scopes(node_id),
            cost=estimate,
            stage="estimate",
            node_id=node_id,
        )
        if preflight.stop:
            return preflight

        outputs_map, cost = adapter.execute(inputs)
        self._store_outputs(outputs, node_id, outputs_map)

        commit = self._apply_budget_checks(
            scopes=self._iter_scopes(node_id),
            cost=cost,
            stage="execute",
            node_id=node_id,
        )
        return commit

    def _execute_loop(
        self,
        loop_node: Mapping[str, object],
        outputs: MutableMapping[str, object],
        variables: Mapping[str, object],
    ) -> _ExecutionOutcome:
        loop_id = str(loop_node.get("id"))
        stop_config = loop_node.get("stop", {})
        body = loop_node.get("body", [])
        if not isinstance(body, Sequence):
            raise ValueError("loop body must be a sequence of nodes")
        if not body:
            return _ExecutionOutcome(stop=False, stop_reason=None)

        loop_budget = stop_config.get("budget") if isinstance(stop_config, Mapping) else None
        if loop_budget is not None:
            self._budget_manager.register_scope(
                scope_type="loop",
                scope_id=loop_id,
                spec=loop_budget
                if isinstance(loop_budget, (Mapping, BudgetSpec))
                else None,
            )

        max_iterations = stop_config.get("max_iterations") if isinstance(stop_config, Mapping) else None
        iterations = 0
        while True:
            if max_iterations is not None and iterations >= int(max_iterations):
                break
            self._loop_stack.append(loop_id)
            for inner in body:
                outcome = self._execute_unit(inner, outputs, variables)
                if outcome.stop:
                    self._loop_stack.pop()
                    return outcome
            self._loop_stack.pop()
            iterations += 1
            if max_iterations is None and loop_budget is None:
                break

        return _ExecutionOutcome(stop=False, stop_reason=None)

    # ------------------------------------------------------------------
    def _iter_scopes(self, node_id: str) -> Iterable[tuple[str, str]]:
        scopes: list[tuple[str, str]] = []
        if self._run_scope_id is not None:
            scopes.append(("run", self._run_scope_id))
        for loop_id in self._loop_stack:
            scopes.append(("loop", loop_id))
        if self._budget_manager.ensure_scope("node", node_id) is not None:
            scopes.append(("node", node_id))
        return scopes

    def _apply_budget_checks(
        self,
        *,
        scopes: Iterable[tuple[str, str]],
        cost: CostSnapshot,
        stage: str,
        node_id: str,
    ) -> _ExecutionOutcome:
        for scope_type, scope_id in scopes:
            outcome = (
                self._budget_manager.preflight(scope_type, scope_id, cost, event_context={"stage": stage, "node": node_id})
                if stage == "estimate"
                else self._budget_manager.commit(scope_type, scope_id, cost, event_context={"stage": stage, "node": node_id})
            )
            if outcome is not None and outcome.stop:
                if stage == "estimate" and scope_type == "loop":
                    # Allow loop bodies to run once more; enforcement happens on commit.
                    continue
                reason = {"scope": scope_type, "scope_id": scope_id}
                if stage != "estimate":
                    if scope_type == "loop":
                        self._trace.emit("loop_halt", scope_type, scope_id, {"reason": "budget_stop"})
                    elif scope_type == "run":
                        self._trace.emit("run_halt", scope_type, scope_id, {"reason": "budget_stop"})
                return _ExecutionOutcome(stop=True, stop_reason=reason)
        return _ExecutionOutcome(stop=False, stop_reason=None)

    def _resolve_policy(self, node_id: str, tool_chain: Sequence[str]) -> Sequence[str]:
        if self._policy_adapter is None:
            allowed = tool_chain
        else:
            allowed = self._policy_adapter.resolve(node_id, tool_chain)
        self._trace.emit(
            "policy_resolved",
            "node",
            node_id,
            {"allowed": list(allowed), "candidates": list(tool_chain)},
        )
        return allowed

    @staticmethod
    def _materialize_inputs(inputs: Mapping[str, object], variables: Mapping[str, object]) -> Mapping[str, object]:
        # For now simply merge variables then override with node inputs.
        merged = dict(variables)
        merged.update(inputs)
        return merged

    def _store_outputs(self, outputs: MutableMapping[str, object], node_id: str, value: Mapping[str, object]) -> None:
        if self._loop_stack:
            bucket = outputs.setdefault(node_id, [])
            if isinstance(bucket, list):
                bucket.append(dict(value))
            else:  # pragma: no cover - defensive programming
                outputs[node_id] = [dict(value)]
        else:
            outputs[node_id] = dict(value)


class PolicyAdapter:
    """Protocol for policy integration (duck-typed in tests)."""

    def resolve(self, node_id: str, tool_chain: Sequence[str]) -> Sequence[str]:  # pragma: no cover - protocol placeholder
        raise NotImplementedError

    def push(self, policy: Mapping[str, object] | None, scope: str) -> None:  # pragma: no cover - optional
        raise NotImplementedError

    def pop(self, expected_scope: str | None = None) -> None:  # pragma: no cover - optional
        raise NotImplementedError
