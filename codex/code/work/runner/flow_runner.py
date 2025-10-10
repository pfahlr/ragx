"""FlowRunner integrating budget guards and trace emission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Sequence

from .budget_manager import BudgetBreachError, BudgetManager
from .budget_models import BudgetChargeOutcome, BudgetScope
from .trace import TraceEventEmitter

__all__ = [
    "FlowNode",
    "LoopDefinition",
    "FlowDefinition",
    "RunContext",
    "FlowResult",
    "FlowRunner",
]


@dataclass(frozen=True)
class FlowNode:
    node_id: str
    adapter_id: str
    scopes: Sequence[BudgetScope]


@dataclass(frozen=True)
class LoopDefinition:
    loop_id: str
    scope: BudgetScope | None
    iterations: int
    nodes: Sequence[FlowNode]


@dataclass(frozen=True)
class FlowDefinition:
    flow_id: str
    run_scope: BudgetScope
    loop: LoopDefinition


@dataclass(frozen=True)
class RunContext:
    flow_id: str
    run_id: str


@dataclass
class FlowResult:
    status: str
    iterations_executed: int
    stop_reason: Mapping[str, object] | None
    warnings: List[Mapping[str, object]]


class FlowRunner:
    """Executes a FlowDefinition while enforcing budgets."""

    def __init__(
        self,
        *,
        budget_manager: BudgetManager,
        trace_emitter: TraceEventEmitter,
        adapters: Mapping[str, object],
    ) -> None:
        self._budget_manager = budget_manager
        self._trace = trace_emitter
        self._adapters: Dict[str, object] = dict(adapters)

    def run(self, flow: FlowDefinition, context: RunContext) -> FlowResult:
        self._trace.emit_run_event("run_start", flow_id=context.flow_id, run_id=context.run_id)
        warnings: List[Mapping[str, object]] = []
        stop_reason: Mapping[str, object] | None = None
        iterations_executed = 0
        loop = flow.loop
        for iteration in range(loop.iterations):
            for node in loop.nodes:
                adapter = self._adapter_for(node.adapter_id)
                estimate = adapter.estimate(node_id=node.node_id, iteration=iteration)
                scopes: List[BudgetScope] = [flow.run_scope]
                if loop.scope is not None:
                    scopes.append(loop.scope)
                scopes.extend(node.scopes)
                context_payload = self._context_payload(context, loop, node, iteration)
                try:
                    outcomes = self._budget_manager.charge(scopes, estimate)
                except BudgetBreachError as error:
                    outcomes = list(error.outcomes or [])
                    for outcome in outcomes:
                        self._trace.emit_budget_charge(outcome=outcome, context=context_payload)
                    breach_outcome = error.outcome
                    self._trace.emit_budget_breach(outcome=breach_outcome, context=context_payload)
                    stop_reason = self._stop_payload("budget_hard_breach", breach_outcome, context_payload)
                    self._trace.emit_run_event(
                        "run_end",
                        flow_id=context.flow_id,
                        run_id=context.run_id,
                        status="failed",
                        stop_reason=stop_reason,
                    )
                    return FlowResult(
                        status="failed",
                        iterations_executed=iterations_executed,
                        stop_reason=stop_reason,
                        warnings=warnings,
                    )
                else:
                    for outcome in outcomes:
                        self._trace.emit_budget_charge(outcome=outcome, context=context_payload)
                        if outcome.breached:
                            self._trace.emit_budget_breach(outcome=outcome, context=context_payload)
                            if outcome.action == "warn":
                                payload = self._stop_payload("budget_soft_breach", outcome, context_payload)
                                warnings.append(payload)
                            elif outcome.action == "stop" and stop_reason is None:
                                payload = self._stop_payload("budget_loop_stop", outcome, context_payload)
                                stop_reason = payload
                    adapter.execute(node_id=node.node_id, iteration=iteration)
            iterations_executed += 1
            if stop_reason is not None:
                break
        status = "stopped" if stop_reason else "completed"
        self._trace.emit_run_event(
            "run_end",
            flow_id=context.flow_id,
            run_id=context.run_id,
            status=status,
            stop_reason=stop_reason,
        )
        return FlowResult(
            status=status,
            iterations_executed=iterations_executed,
            stop_reason=stop_reason,
            warnings=warnings,
        )

    def _adapter_for(self, adapter_id: str) -> object:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Missing adapter: {adapter_id}") from exc

    def _context_payload(
        self,
        context: RunContext,
        loop: LoopDefinition,
        node: FlowNode,
        iteration: int,
    ) -> MutableMapping[str, object]:
        return {
            "flow_id": context.flow_id,
            "run_id": context.run_id,
            "loop_id": loop.loop_id,
            "loop_iteration": iteration,
            "node_id": node.node_id,
            "adapter_id": node.adapter_id,
        }

    def _stop_payload(
        self,
        reason: str,
        outcome: BudgetChargeOutcome,
        context_payload: Mapping[str, object],
    ) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "reason": reason,
            "scope": outcome.scope.as_dict(),
            "action": outcome.action,
            "overages": {metric: str(amount) for metric, amount in outcome.overages.items()},
            "remaining": {metric: str(amount) for metric, amount in outcome.remaining.items()},
            "context": {
                "node_id": context_payload["node_id"],
                "loop_iteration": context_payload["loop_iteration"],
            },
        }
        return payload
