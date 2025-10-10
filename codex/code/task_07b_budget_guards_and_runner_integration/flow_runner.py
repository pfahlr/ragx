"""FlowRunner integration harness for budget and policy testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .budget_manager import BudgetManager
from .budget_models import BudgetDecision, BudgetSpec, CostAmount
from .policy_stack import PolicyDecision, PolicyStack
from .trace_emitter import TraceEventEmitter


class ToolAdapter(Protocol):
    def estimate(self, node: "FlowNode") -> CostAmount:  # pragma: no cover - protocol
        ...

    def execute(self, node: "FlowNode") -> Tuple[Any, CostAmount]:  # pragma: no cover - protocol
        ...

    def identify(self, node: "FlowNode") -> str:  # pragma: no cover - protocol
        ...

    def describe(self, node: "FlowNode") -> Mapping[str, Any]:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class FlowNode:
    node_id: str
    adapter: ToolAdapter
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RunResult:
    executed_nodes: Sequence[str]
    warnings: Sequence[Any]
    stop_reason: Optional[str]


class FlowRunner:
    """Executes nodes while coordinating budget and policy enforcement."""

    def __init__(
        self,
        *,
        budget_manager: BudgetManager,
        policy_stack: PolicyStack,
        emitter: Optional[TraceEventEmitter] = None,
    ) -> None:
        self.budget_manager = budget_manager
        self.policy_stack = policy_stack
        self.emitter = emitter or TraceEventEmitter()

    def run(
        self,
        nodes: Iterable[FlowNode],
        *,
        run_scope: str,
        budget_specs: Mapping[str, BudgetSpec],
    ) -> RunResult:
        executed: List[str] = []
        warnings: List[Any] = []
        stop_reason: Optional[str] = None

        for scope_id, spec in budget_specs.items():
            try:
                self.budget_manager.register_scope(scope_id, spec)
            except ValueError:
                continue

        self.emitter.emit(
            "run_start",
            scope_type="run",
            scope_id=run_scope,
            payload={"scopes": list(budget_specs.keys())},
        )

        for node in nodes:
            self.emitter.emit(
                "node_start",
                scope_type="node",
                scope_id=node.node_id,
                payload={"adapter": node.adapter.identify(node)},
            )

            estimated = node.adapter.estimate(node)
            preview = self.budget_manager.preview(run_scope, estimated, node_id=node.node_id)
            if preview.decision is BudgetDecision.STOP:
                stop_reason = "budget_stop"
                break

            policy_decision: PolicyDecision = self.policy_stack.evaluate(node)
            if not policy_decision.allowed:
                stop_reason = "policy_violation"
                break

            output, actual_cost = node.adapter.execute(node)
            charge = self.budget_manager.commit(run_scope, actual_cost, node_id=node.node_id)
            executed.append(node.node_id)

            if charge.decision is BudgetDecision.WARN and charge.breach is not None:
                warnings.append(charge.breach)

            self.emitter.emit(
                "node_end",
                scope_type="node",
                scope_id=node.node_id,
                payload={"output": output},
            )

        if stop_reason is None:
            self.emitter.emit(
                "run_end",
                scope_type="run",
                scope_id=run_scope,
                payload={"executed": executed},
            )
        else:
            self.emitter.emit(
                "run_stop",
                scope_type="run",
                scope_id=run_scope,
                payload={"reason": stop_reason, "executed": executed},
            )

        return RunResult(executed_nodes=tuple(executed), warnings=tuple(warnings), stop_reason=stop_reason)


__all__ = ["FlowNode", "FlowRunner", "RunResult", "ToolAdapter"]
