from __future__ import annotations

from copy import deepcopy

import pytest

from pkgs.dsl import (
    BudgetExceededError,
    Cost,
    FlowRunner,
    LoopIterationContext,
    LoopIterationResult,
)


class StubRunner(FlowRunner):
    def __init__(self, iteration_costs: list[float]) -> None:
        super().__init__()
        self._iteration_costs = iteration_costs

    def _estimate_loop_iteration_cost(
        self,
        loop_spec: dict,
        iteration: int,
        context: LoopIterationContext,
    ) -> Cost:
        if iteration >= len(self._iteration_costs):
            return Cost()
        return Cost(usd=self._iteration_costs[iteration])

    def _run_loop_iteration(
        self,
        loop_spec: dict,
        iteration: int,
        context: LoopIterationContext,
    ) -> LoopIterationResult:
        if iteration >= len(self._iteration_costs):
            return LoopIterationResult(should_stop=True)
        return LoopIterationResult(cost=Cost(usd=self._iteration_costs[iteration]))


BASE_SPEC: dict = {
    "version": "0.1",
    "globals": {
        "tools": {},
        "run_budget": {"max_usd": 10.0, "mode": "hard"},
    },
    "graph": {
        "nodes": [],
        "control": [
            {
                "id": "loop",
                "kind": "loop",
                "target_subgraph": [],
                "stop": {
                    "max_iterations": 5,
                    "budget": {"max_usd": 1.5, "breach_action": "stop"},
                },
            }
        ],
    },
}


@pytest.fixture
def budget_flow_spec() -> dict:
    return deepcopy(BASE_SPEC)


def test_loop_budget_stop_halts_iteration_without_error(budget_flow_spec: dict) -> None:
    runner = StubRunner([0.4, 0.7, 0.9, 0.2])

    result = runner.run(budget_flow_spec, vars={})

    assert result.status == "ok"
    assert result.outputs == {}
    assert runner.last_error is None
    assert runner.last_run is not None
    assert runner.last_run.loop_summaries

    loop_summary = runner.last_run.loop_summaries[0]
    assert loop_summary.loop_id == "loop"
    assert loop_summary.iterations == 2
    assert loop_summary.stop_reason == "budget_stop"

    breach_events = [
        event
        for event in runner.trace_events
        if event["event"] == "budget_breach" and event.get("scope") == "loop"
    ]
    assert breach_events
    assert breach_events[-1]["action"] == "stop"
    assert breach_events[-1]["remaining"]["max_usd"] == pytest.approx(0.4)


def test_run_budget_hard_breach_sets_error_status(budget_flow_spec: dict) -> None:
    spec = deepcopy(budget_flow_spec)
    spec["globals"]["run_budget"] = {"max_usd": 1.0, "mode": "hard"}

    runner = StubRunner([0.6, 0.6])
    result = runner.run(spec, vars={})

    assert result.status == "error"
    assert isinstance(runner.last_error, BudgetExceededError)
    assert runner.last_error.metric == "usd"

    breach_events = [
        event for event in runner.trace_events if event["event"] == "budget_breach"
    ]
    assert breach_events[-1]["scope"] == "run"
    assert breach_events[-1]["breach_kind"] == "hard"
