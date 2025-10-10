from __future__ import annotations

import importlib.util
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Mapping, TYPE_CHECKING

from pkgs.dsl.policy import PolicyStack

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .adapters import AdapterContext, ToolAdapter
    from .budget import BudgetBreach, BudgetManager, BudgetSpec
    from .trace import TraceEventEmitter
else:
    _MODULE_DIR = pathlib.Path(__file__).resolve().parent

    def _load_module(name: str, symbol: str | None = None):
        path = _MODULE_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"task07b_{name}", path)
        assert spec.loader is not None  # pragma: no cover - defensive
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _adapters_mod = _load_module("adapters")
    _budget_mod = _load_module("budget")
    _trace_mod = _load_module("trace")

    AdapterContext = _adapters_mod.AdapterContext
    ToolAdapter = _adapters_mod.ToolAdapter
    BudgetBreach = _budget_mod.BudgetBreach
    BudgetManager = _budget_mod.BudgetManager
    BudgetSpec = _budget_mod.BudgetSpec
    TraceEventEmitter = _trace_mod.TraceEventEmitter

__all__ = [
    "NodePlan",
    "LoopPlan",
    "RunPlan",
    "RunResult",
    "FlowRunner",
]


@dataclass(frozen=True, slots=True)
class NodePlan:
    node_id: str
    tool: str
    budget: BudgetSpec | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoopPlan:
    loop_id: str
    iterations: int
    nodes: tuple[NodePlan, ...]
    budget: BudgetSpec | None = None


@dataclass(frozen=True, slots=True)
class RunPlan:
    run_id: str
    loops: tuple[LoopPlan, ...]
    run_budget: BudgetSpec | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    outputs: tuple[object, ...]
    breaches: tuple[BudgetBreach, ...]
    stop_reason: str | None


class FlowRunner:
    """Execute looped node plans with policy enforcement and budget control."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, ToolAdapter],
        policy_stack: PolicyStack,
        budget_manager: BudgetManager,
        trace_emitter: TraceEventEmitter,
    ) -> None:
        self._adapters = dict(adapters)
        self._policy_stack = policy_stack
        self._budget_manager = budget_manager
        self._trace_emitter = trace_emitter

    def run(self, plan: RunPlan) -> RunResult:
        outputs: list[object] = []
        breaches: list[BudgetBreach] = []
        stop_reason: str | None = None

        with self._budget_manager.open_scope(plan.run_budget) as _run_scope:
            for loop_plan in plan.loops:
                loop_stop_reason: str | None = None
                executed_iterations = 0
                with self._budget_manager.open_scope(loop_plan.budget) as _loop_scope:
                    for iteration in range(loop_plan.iterations):
                        executed_iterations = iteration + 1
                        for node_plan in loop_plan.nodes:
                            adapter = self._resolve_adapter(node_plan.tool)
                            context = AdapterContext(
                                run_id=plan.run_id,
                                node_id=node_plan.node_id,
                                loop_id=loop_plan.loop_id,
                                iteration=iteration,
                                metadata=node_plan.metadata,
                            )
                            self._policy_stack.effective_allowlist([node_plan.tool])
                            snapshot = self._policy_stack.enforce(node_plan.tool)
                            with self._budget_manager.open_scope(node_plan.budget) as node_scope:
                                adapter.estimate(context)
                                result = adapter.execute(context)
                                outputs.append(result.output)
                                charge = node_scope.charge(result.cost)
                                for outcome in charge.outcomes:
                                    self._trace_emitter.emit_budget_charge(outcome)
                                if charge.breached is not None:
                                    breach_outcome = charge.outcome_for(
                                        charge.breached.scope_type, charge.breached.scope_id
                                    )
                                    self._trace_emitter.emit_budget_breach(breach_outcome)
                                    breaches.append(charge.breached)
                                    loop_stop_reason = charge.breached.stop_reason
                                    stop_reason = charge.breached.stop_reason
                                    break
                            _ = snapshot
                            if loop_stop_reason is not None:
                                break
                        if loop_stop_reason is not None:
                            self._trace_emitter.emit_loop_summary(
                                loop_id=loop_plan.loop_id,
                                iteration=executed_iterations,
                                stop_reason=loop_stop_reason,
                                remaining_iterations=max(loop_plan.iterations - executed_iterations, 0),
                            )
                            break
                    else:
                        self._trace_emitter.emit_loop_summary(
                            loop_id=loop_plan.loop_id,
                            iteration=executed_iterations,
                            stop_reason="completed",
                            remaining_iterations=0,
                        )
                if loop_stop_reason is not None:
                    break
        return RunResult(
            run_id=plan.run_id,
            outputs=tuple(outputs),
            breaches=tuple(breaches),
            stop_reason=stop_reason,
        )

    def _resolve_adapter(self, tool: str) -> ToolAdapter:
        try:
            return self._adapters[tool]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"no adapter registered for {tool}") from exc
