"""Simplified FlowRunner integrating budget management and policy traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .budget_manager import BudgetBreachError, BudgetManager
from .budget_models import BreachAction, BudgetDecision, BudgetSpec, CostSnapshot, ScopeKey
from .trace_emitter import TraceEventEmitter


@dataclass(frozen=True)
class NodeExecution:
    node_id: str
    adapter: str
    output: Any
    cost: CostSnapshot


@dataclass
class RunResult:
    executions: List[NodeExecution] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None


class PolicyStack:
    """Tracks active policies and emits trace events."""

    def __init__(self, emitter: TraceEventEmitter) -> None:
        self._emitter = emitter
        self._stack: List[tuple[str, str]] = []

    def push(self, policy_id: Optional[str], node_id: str) -> None:
        if policy_id is None:
            return
        self._stack.append((policy_id, node_id))
        self._emitter.emit("policy_push", {"policy_id": policy_id, "node_id": node_id})

    def resolve(self, policy_id: Optional[str], node_id: str, status: str = "completed") -> None:
        if policy_id is None:
            return
        self._emitter.emit(
            "policy_resolved",
            {"policy_id": policy_id, "node_id": node_id, "status": status},
        )
        if self._stack and self._stack[-1] == (policy_id, node_id):
            self._stack.pop()

    def violation(self, policy_id: Optional[str], node_id: str, reason: str) -> None:
        if policy_id is None:
            return
        self._emitter.emit(
            "policy_violation",
            {"policy_id": policy_id, "node_id": node_id, "reason": reason},
        )
        if self._stack and self._stack[-1] == (policy_id, node_id):
            self._stack.pop()


class FlowRunner:
    """Executes flow nodes while enforcing budgets and emitting traces."""

    def __init__(
        self,
        *,
        budget_manager: BudgetManager,
        trace_emitter: TraceEventEmitter,
        policy_stack: Optional[PolicyStack] = None,
    ) -> None:
        self._manager = budget_manager
        self._trace = trace_emitter
        self._policy = policy_stack or PolicyStack(trace_emitter)

    def run(self, flow_spec: Mapping[str, Any], adapters: Mapping[str, Any]) -> RunResult:
        run_id = flow_spec.get("run_id", "run")
        result = RunResult()

        run_scope = None
        if "run_budget" in flow_spec:
            run_scope = self._ensure_scope("run", run_id, flow_spec["run_budget"])

        for spec_id, config in flow_spec.get("spec_budgets", {}).items():
            self._ensure_scope("spec", spec_id, config)
        for loop_id, config in flow_spec.get("loop_budgets", {}).items():
            self._ensure_scope("loop", loop_id, config)

        for node in flow_spec.get("nodes", []):
            node_id = node["id"]
            adapter_name = node["adapter"]
            policy_id = node.get("policy")
            loop_id = node.get("loop_id")
            spec_id = node.get("spec_id")
            node_scope = None
            if "budget" in node and node["budget"] is not None:
                node_scope = self._ensure_scope("node", node_id, node["budget"])

            adapter = adapters[adapter_name]
            context = {"run_id": run_id, "node": node}
            self._policy.push(policy_id, node_id)

            estimate_kwargs: Dict[str, Any] = {}
            if node.get("estimate_ms") is not None:
                estimate_kwargs["duration_ms"] = node.get("estimate_ms")
            elif node.get("estimate_seconds") is not None:
                estimate_kwargs["duration_seconds"] = node.get("estimate_seconds")
            elif node.get("execute_ms") is not None:
                estimate_kwargs["duration_ms"] = node.get("execute_ms")
            else:
                estimate_kwargs["duration_ms"] = 0
            estimated = CostSnapshot.from_inputs(**estimate_kwargs)
            scopes = self._scopes_for_node(run_scope, spec_id, loop_id, node_scope)

            stop_reason = self._run_preflight(scopes, estimated, policy_id, node_id)
            if stop_reason:
                result.stop_reason = stop_reason
                result.warnings.extend(self._manager.drain_warnings())
                break

            execution = adapter.execute(context)
            cost_payload = execution.get("cost")
            if cost_payload is None:
                fallback = node.get("execute_ms") or estimated.milliseconds
                cost_payload = {"duration_ms": fallback}
            actual_cost = CostSnapshot.from_inputs(payload=cost_payload)

            stop_reason = self._commit_costs(scopes, actual_cost, policy_id, node_id)
            result.warnings.extend(self._manager.drain_warnings())
            if stop_reason:
                result.stop_reason = stop_reason
                break

            result.executions.append(
                NodeExecution(
                    node_id=node_id,
                    adapter=adapter_name,
                    output=execution.get("output"),
                    cost=actual_cost,
                )
            )
            self._policy.resolve(policy_id, node_id)

        result.warnings.extend(self._manager.drain_warnings())
        return result

    def _scopes_for_node(
        self,
        run_scope: Optional[ScopeKey],
        spec_id: Optional[str],
        loop_id: Optional[str],
        node_scope: Optional[ScopeKey],
    ) -> List[ScopeKey]:
        scopes: List[ScopeKey] = []
        if run_scope is not None:
            scopes.append(run_scope)
        if spec_id is not None:
            scopes.append(ScopeKey("spec", spec_id))
        if loop_id is not None:
            scopes.append(ScopeKey("loop", loop_id))
        if node_scope is not None:
            scopes.append(node_scope)
        return scopes

    def _run_preflight(
        self,
        scopes: Iterable[ScopeKey],
        estimated: CostSnapshot,
        policy_id: Optional[str],
        node_id: str,
    ) -> Optional[str]:
        for scope in scopes:
            decision = self._manager.preflight(scope, estimated)
            spec = self._manager.spec_for(scope)
            self._emit_budget_event("budget_preflight", decision, spec.limit_ms)
            if decision.breach is not None:
                self._emit_budget_breach(decision, spec.limit_ms)
            if not decision.allowed and decision.action == BreachAction.STOP:
                reason = f"budget_breach:{scope.category}:{scope.identifier}"
                self._policy.violation(policy_id, node_id, reason)
                return reason
        return None

    def _commit_costs(
        self,
        scopes: Iterable[ScopeKey],
        actual_cost: CostSnapshot,
        policy_id: Optional[str],
        node_id: str,
    ) -> Optional[str]:
        for scope in scopes:
            spec = self._manager.spec_for(scope)
            try:
                decision = self._manager.commit(scope, actual_cost)
            except BudgetBreachError as exc:
                decision = exc.decision
                self._emit_budget_event("budget_charge", decision, spec.limit_ms)
                self._emit_budget_breach(decision, spec.limit_ms)
                reason = f"budget_breach:{scope.category}:{scope.identifier}"
                self._policy.violation(policy_id, node_id, reason)
                return reason
            self._emit_budget_event("budget_charge", decision, spec.limit_ms)
            if decision.breach is not None:
                self._emit_budget_breach(decision, spec.limit_ms)
        return None

    def _ensure_scope(self, category: str, identifier: str, config: Mapping[str, Any]) -> ScopeKey:
        scope = ScopeKey(category, identifier)
        if not self._manager.has_scope(scope):
            spec = BudgetSpec.from_config(scope=scope, config=config)
            self._manager.register_scope(spec)
        return scope

    def _emit_budget_event(
        self,
        event_type: str,
        decision: BudgetDecision,
        limit_ms: float,
    ) -> None:
        payload = dict(decision.to_payload())
        payload["limit_ms"] = limit_ms
        self._trace.emit(event_type, payload)

    def _emit_budget_breach(self, decision: BudgetDecision, limit_ms: float) -> None:
        payload = dict(decision.to_payload())
        payload["limit_ms"] = limit_ms
        self._trace.emit("budget_breach", payload)
