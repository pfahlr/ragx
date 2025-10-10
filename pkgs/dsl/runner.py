"""FlowRunner scaffolding with budget integration hooks."""

from __future__ import annotations

from collections.abc import Mapping

from .budget import BudgetManager
from .trace import RunnerTraceRecorder

__all__ = [
    "FlowRunner",
]


class FlowRunner:
    """Execute DSL flows with policy and budget enforcement (stubbed)."""

    def __init__(
        self,
        *,
        trace: RunnerTraceRecorder | None = None,
        budget_manager: BudgetManager | None = None,
    ) -> None:
        self._trace = trace or RunnerTraceRecorder()
        self._budget_manager = budget_manager or BudgetManager(trace=self._trace)

    @property
    def trace(self) -> RunnerTraceRecorder:
        return self._trace

    @property
    def budget_manager(self) -> BudgetManager:
        return self._budget_manager

    def plan(self, spec: Mapping[str, object], vars: Mapping[str, object]) -> Mapping[str, object]:
        raise NotImplementedError("FlowRunner.plan is not implemented yet")

    def run(self, spec: Mapping[str, object], vars: Mapping[str, object]) -> Mapping[str, object]:
        raise NotImplementedError("FlowRunner.run is not implemented yet")

    # ------------------------------------------------------------------
    # Budget plumbing used by tests (and future execution engine)
    # ------------------------------------------------------------------
    def prepare_budgets(self, spec: Mapping[str, object]) -> None:
        """Instantiate run/node/loop budget meters from the DSL spec."""

        self._budget_manager.reset()
        globals_cfg = spec.get("globals") or {}
        run_budget = globals_cfg.get("run_budget")
        self._budget_manager.configure_run(run_budget)

        graph = spec.get("graph") or {}
        nodes = graph.get("nodes") or []
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue
            kind = node.get("kind")
            if kind == "loop":
                stop_cfg = node.get("stop") or {}
                loop_budget = stop_cfg.get("budget")
                self._budget_manager.register_loop(node_id, loop_budget)
                continue

            hard_budget = node.get("budget")
            spec_cfg = node.get("spec")
            soft_budget = None
            if isinstance(spec_cfg, Mapping):
                soft_budget = spec_cfg.get("budget")
            self._budget_manager.register_node(
                node_id,
                hard_budget=hard_budget,
                soft_budget=soft_budget,
            )

