"""FlowRunner budget guard integration tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from pkgs.dsl.budget import Cost
from pkgs.dsl.runner import FlowRunner, NodeExecution, RunResult


class DummyFlowRunner(FlowRunner):
    """FlowRunner subclass that injects deterministic costs for testing."""

    def __init__(self, iteration_cost: Cost) -> None:
        super().__init__(id_factory=lambda: "run-test", now_factory=lambda: 0.0)
        self._iteration_cost = iteration_cost
        self.invocations: list[str] = []

    def _iteration_cost_hint(
        self, loop: Mapping[str, object], iteration_index: int
    ) -> Cost:
        return self._iteration_cost

    def _execute_node(
        self,
        node: Mapping[str, object],
        context: Mapping[str, object],
        *,
        loop_id: str | None = None,
    ) -> NodeExecution:
        node_id = str(node["id"])
        self.invocations.append(node_id)
        primary_output = node.get("outputs", [f"out_{node_id}"])[0]
        return NodeExecution(
            node_id=node_id,
            outputs={primary_output: f"payload-{len(self.invocations)}"},
            cost=self._iteration_cost,
        )


@pytest.fixture
def loop_spec() -> dict:
    return {
        "version": "0.1",
        "globals": {
            "tools": {"mock": {"type": "mock", "tags": []}},
            "run_budget": {"max_calls": 9, "mode": "hard"},
        },
        "graph": {
            "nodes": [
                {
                    "id": "worker",
                    "kind": "unit",
                    "spec": {"type": "tool", "tool_ref": "mock"},
                    "outputs": ["result"],
                    "budget": {"max_calls": 5, "mode": "hard"},
                }
            ],
            "control": [
                {
                    "id": "loop",
                    "kind": "loop",
                    "target_subgraph": ["worker"],
                    "stop": {
                        "max_iterations": 10,
                        "budget": {"max_calls": 3, "breach_action": "stop"},
                    },
                }
            ],
        },
    }


def test_loop_budget_stop_halts_iterations(loop_spec: dict) -> None:
    runner = DummyFlowRunner(iteration_cost=Cost(calls=1))
    result = runner.run(loop_spec, vars={"root": {}})

    assert isinstance(result, RunResult)
    assert result.status == "ok"
    assert runner.invocations == ["worker", "worker", "worker"]

    loop_events = [event for event in result.trace if event["event"] == "loop_stop"]
    assert loop_events, "expected loop_stop event"
    stop_event = loop_events[-1]
    assert stop_event["loop_id"] == "loop"
    assert stop_event["reason"] == "budget_stop"
    assert stop_event["details"]["breached"] == ("calls",)

    # The run budget should accumulate the same number of calls as executed iterations.
    run_spend = [
        event
        for event in result.trace
        if event["event"] == "budget_charge" and event["scope"] == "run"
    ]
    assert len(run_spend) == 3
    assert run_spend[-1]["decision"]["remaining"]["calls"] == 6
