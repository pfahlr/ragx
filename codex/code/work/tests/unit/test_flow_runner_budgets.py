from __future__ import annotations

from dataclasses import dataclass

from pkgs.dsl.budget import CostSnapshot
from pkgs.dsl.runner import FlowRunner, RunResult, ToolAdapter, ToolExecutionResult


@dataclass
class CountingAdapter(ToolAdapter):
    cost_per_call: CostSnapshot
    output_value: str = "ok"
    calls: int = 0

    def estimate_cost(self, node_spec: dict, context: dict) -> CostSnapshot:
        return self.cost_per_call

    def execute(self, node_spec: dict, context: dict) -> ToolExecutionResult:
        self.calls += 1
        return ToolExecutionResult(
            outputs={"value": f"{self.output_value}-{self.calls}"},
            cost=self.cost_per_call,
        )


def make_runner_spec() -> dict:
    return {
        "globals": {
            "tools": {
                "alpha": {"tags": []},
            },
            "run_budget": {"max_usd": 10.0, "mode": "hard"},
        },
        "graph": {
            "sequence": ["loop"],
            "nodes": {
                "loop": {
                    "id": "loop",
                    "kind": "loop",
                    "target_subgraph": ["loop_step"],
                    "stop": {
                        "budget": {"max_calls": 2, "breach_action": "stop"}
                    },
                },
                "loop_step": {
                    "id": "loop_step",
                    "kind": "unit",
                    "outputs": ["value"],
                    "spec": {"tool_ref": "alpha"},
                },
            },
        },
    }


def test_flow_runner_loop_budget_stop() -> None:
    spec = make_runner_spec()
    adapters = {"alpha": CountingAdapter(cost_per_call=CostSnapshot(calls=1))}
    runner = FlowRunner(adapters=adapters)
    result = runner.run(spec=spec, vars={})
    assert isinstance(result, RunResult)
    assert result.status == "halted"
    assert result.stop_reasons == ("loop:loop:budget_stop",)
    assert adapters["alpha"].calls == 2


def test_flow_runner_node_budget_error_sets_failure() -> None:
    spec = {
        "globals": {
            "tools": {"alpha": {"tags": []}},
            "run_budget": {"max_usd": 5.0, "mode": "hard"},
        },
        "graph": {
            "sequence": ["expensive"],
            "nodes": {
                "expensive": {
                    "id": "expensive",
                    "kind": "unit",
                    "outputs": ["value"],
                    "budget": {"max_usd": 1.0, "mode": "hard"},
                    "spec": {"tool_ref": "alpha"},
                }
            },
        },
    }
    adapters = {
        "alpha": CountingAdapter(cost_per_call=CostSnapshot(usd=2.0), output_value="exp"),
    }
    runner = FlowRunner(adapters=adapters)
    result = runner.run(spec=spec, vars={})
    assert result.status == "error"
    assert result.stop_reasons[0].startswith("node:expensive")


def test_flow_runner_respects_policy_allowlist() -> None:
    spec = {
        "globals": {
            "tools": {
                "alpha": {"tags": []},
                "beta": {"tags": []},
            },
            "tool_sets": {},
            "run_budget": {"max_usd": 5.0, "mode": "hard"},
        },
        "graph": {
            "sequence": ["restricted"],
            "nodes": {
                "restricted": {
                    "id": "restricted",
                    "kind": "unit",
                    "outputs": ["value"],
                    "policy": {"allow_tools": ["alpha"]},
                    "spec": {"tool_ref": "beta"},
                }
            },
        },
    }
    adapters = {
        "alpha": CountingAdapter(cost_per_call=CostSnapshot(usd=1.0), output_value="alpha"),
        "beta": CountingAdapter(cost_per_call=CostSnapshot(usd=1.0), output_value="beta"),
    }
    runner = FlowRunner(adapters=adapters)
    result = runner.run(spec=spec, vars={})
    assert result.status == "error"
    assert result.stop_reasons == ("policy:restricted:beta",)
