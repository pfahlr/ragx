from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .budget import BudgetChargeOutcome, BudgetManager, BudgetMode, BudgetSpec
from .trace import TraceEventEmitter


class PolicyViolationError(RuntimeError):
    """Raised when a tool fails policy enforcement."""

    def __init__(self, tool: str) -> None:
        super().__init__(f"Tool '{tool}' blocked by policy")
        self.tool = tool


@dataclass(frozen=True, slots=True)
class NodeDefinition:
    """Declarative description for a flow node."""

    id: str
    tool: str
    scope: str | None = None


@dataclass(frozen=True, slots=True)
class FlowDefinition:
    """Minimal flow abstraction for the runner."""

    run_scope: str
    nodes: Sequence[NodeDefinition]


@dataclass(frozen=True, slots=True)
class NodeExecution:
    """Captured execution metadata for a node."""

    node_id: str
    result: Mapping[str, object]
    cost: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Aggregate execution outcome for a flow run."""

    executions: tuple[NodeExecution, ...]
    warnings: tuple[str, ...]
    stop_reason: str | None


class ToolAdapter(Protocol):
    """Protocol describing adapters consumed by :class:`FlowRunner`."""

    def estimate(self, node: NodeDefinition) -> Mapping[str, float]:
        ...

    def execute(self, node: NodeDefinition) -> tuple[Mapping[str, object], Mapping[str, float]]:
        ...


class PolicyStack:
    """Simplified policy stack honouring allow/deny semantics."""

    def __init__(
        self,
        *,
        allow_tools: Sequence[str] | None = None,
        deny_tools: Sequence[str] | None = None,
        trace_emitter: TraceEventEmitter | None = None,
    ) -> None:
        self._allow = frozenset(allow_tools or [])
        self._deny = frozenset(deny_tools or [])
        self._emitter = trace_emitter

    def enforce(self, node: NodeDefinition) -> None:
        if node.tool in self._deny or (self._allow and node.tool not in self._allow):
            self._emit(
                event="policy_violation",
                node=node,
                payload={"tool": node.tool},
            )
            raise PolicyViolationError(node.tool)
        self._emit(
            event="policy_resolved",
            node=node,
            payload={"tool": node.tool, "status": "allowed"},
        )

    def _emit(self, *, event: str, node: NodeDefinition, payload: Mapping[str, object]) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            event=event,
            scope=node.id,
            scope_type="policy",
            payload=payload,
        )


class FlowRunner:
    """Coordinates adapter execution, policy enforcement, and budget metering."""

    def __init__(
        self,
        *,
        adapter: ToolAdapter,
        budget_manager: BudgetManager,
        policy_stack: PolicyStack,
        trace_emitter: TraceEventEmitter | None = None,
    ) -> None:
        self._adapter = adapter
        self._budget_manager = budget_manager
        self._policy_stack = policy_stack
        self._emitter = trace_emitter

    def run(self, flow: FlowDefinition) -> RunResult:
        executions: list[NodeExecution] = []
        warnings: list[str] = []
        stop_reason: str | None = None

        for node in flow.nodes:
            self._policy_stack.enforce(node)
            scope = node.scope or flow.run_scope

            estimate = self._adapter.estimate(node)
            preflight = self._budget_manager.preflight(scope, estimate)
            warnings.extend(preflight.warnings)
            if preflight.stop_requested:
                stop_reason = "budget_preflight"
                break

            self._emit(event="node_start", node=node, payload={"estimate": estimate})
            result, cost = self._adapter.execute(node)
            outcome = self._budget_manager.commit(scope, cost)
            executions.append(
                NodeExecution(
                    node_id=node.id,
                    result=result,
                    cost=outcome.charged.metrics,
                )
            )
            warnings.extend(outcome.warnings)
            self._emit(event="node_end", node=node, payload={"result": result})

            if outcome.stop:
                stop_reason = "budget_commit"
                break

        return RunResult(
            executions=tuple(executions),
            warnings=tuple(warnings),
            stop_reason=stop_reason,
        )

    def _emit(self, *, event: str, node: NodeDefinition, payload: Mapping[str, object]) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            event=event,
            scope=node.id,
            scope_type="runner",
            payload=payload,
        )


__all__ = [
    "BudgetChargeOutcome",
    "BudgetManager",
    "BudgetMode",
    "BudgetSpec",
    "FlowDefinition",
    "FlowRunner",
    "NodeDefinition",
    "NodeExecution",
    "PolicyStack",
    "PolicyViolationError",
    "RunResult",
    "ToolAdapter",
]
