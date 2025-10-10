"""Integration-style tests for FlowRunner budget enforcement and tracing."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from pkgs.dsl.runner import FlowRunner, RunResult, ToolAdapter
from pkgs.dsl.trace import TraceEventEmitter


@dataclass
class RecordingAdapter(ToolAdapter):
    """Simple adapter that increments a counter and records executions."""

    cost_per_call: Mapping[str, float]
    payload_template: Mapping[str, Any]
    calls: int = 0

    def estimate_cost(self, node_spec: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, float]:
        return dict(self.cost_per_call)

    def execute(self, node_spec: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, float]]:
        self.calls += 1
        outputs = {"value": self.calls, **self.payload_template}
        return outputs, dict(self.cost_per_call)


@pytest.fixture()
def flow_spec() -> Mapping[str, Any]:
    return {
        "id": "demo",
        "run": {
            "budget": {
                "name": "run",
                "limit": {"calls": 10},
                "mode": "hard",
                "breach_action": "stop",
            }
        },
        "graph": [
            {
                "id": "loop",
                "kind": "loop",
                "stop": {
                    "budget": {
                        "name": "loop",
                        "limit": {"calls": 2},
                        "mode": "hard",
                        "breach_action": "stop",
                    }
                },
                "target": [
                    {
                        "id": "task",
                        "kind": "unit",
                        "budget": {
                            "name": "task",
                            "limit": {"calls": 1},
                            "mode": "soft",
                            "breach_action": "warn",
                        },
                        "spec": {
                            "tool_ref": "adder",
                        },
                    }
                ],
            }
        ],
    }


def test_flow_runner_emits_traces_and_stops_on_loop_budget(flow_spec: Mapping[str, Any]) -> None:
    adapter = RecordingAdapter(cost_per_call={"calls": 1}, payload_template={})
    emitter = TraceEventEmitter()
    runner = FlowRunner(
        tool_adapters={"adder": adapter},
        tool_registry={"adder": {"tags": []}},
        trace_emitter=emitter,
        run_id_factory=lambda: "run-1",
    )

    result = runner.run(flow_spec, vars={})
    assert isinstance(result, RunResult)
    assert result.status == "halted"
    assert result.stop_reason == {"scope": "loop", "id": "loop", "reason": "budget_stop"}
    assert result.outputs["task"] == ({"value": 1}, {"value": 2})
    assert any(warning["scope"] == "node" and warning["id"] == "task" for warning in result.warnings)

    breach_events = [event for event in emitter.events if event.event == "budget_breach"]
    assert breach_events, "loop breach should be traced"
    loop_breach = next(
        event for event in breach_events if event.data["scope_kind"] == "loop"
    )
    assert isinstance(loop_breach.data, MappingProxyType)
    assert loop_breach.data["scope_kind"] == "loop"
    assert loop_breach.data["loop_iteration"] == 3
    assert loop_breach.data["run_id"] == "run-1"

    policy_events = [event for event in emitter.events if event.event == "policy_resolved"]
    assert policy_events, "Policy resolution must be traced"
    policy_payload = policy_events[0].data
    assert isinstance(policy_payload, MappingProxyType)
    assert policy_payload["allowed"] == ("adder",)
    assert policy_payload["node_id"] == "task"

    charge_events = [event for event in emitter.events if event.event == "budget_charge"]
    assert len(charge_events) >= 4  # run + loop + node budgets charged multiple times
    assert all("remaining" in event.data for event in charge_events)
