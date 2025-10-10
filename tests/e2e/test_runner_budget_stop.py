"""Flow runner integration tests for loop-level budget stops."""

from __future__ import annotations

from typing import Any

from pkgs.dsl.runner import FlowRunner, RunResult


def test_runner_loop_stops_when_loop_budget_reaches_cap() -> None:
    calls: list[int] = []

    def mock_tool(**_: Any) -> dict[str, Any]:
        iteration = len(calls)
        calls.append(iteration)
        return {
            "outputs": {"result": f"iteration-{iteration}"},
            "cost": {"usd": 0.25, "calls": 1},
        }

    spec: dict[str, Any] = {
        "globals": {
            "run_budget": {"mode": "hard", "max_calls": 5, "max_usd": 5.0},
            "tools": {
                "mock.tool": {"type": "tool", "pricing": {"per_call_usd": 0.25}},
            },
        },
        "graph": {
            "nodes": [
                {
                    "id": "worker",
                    "kind": "unit",
                    "inputs": {},
                    "outputs": ["result"],
                    "spec": {"type": "tool", "tool_ref": "mock.tool"},
                    "budget": {"mode": "hard", "max_calls": 5},
                }
            ]
        },
        "control": [
            {
                "id": "self_loop",
                "kind": "loop",
                "target_subgraph": ["worker"],
                "stop": {"budget": {"max_calls": 3, "breach_action": "stop"}},
            }
        ],
    }

    runner = FlowRunner(tool_adapters={"mock.tool": mock_tool})
    result = runner.run(spec, vars={})

    assert isinstance(result, RunResult)
    assert result.status == "ok"
    assert len(calls) == 3

    loop_breaches = [
        event
        for event in result.trace
        if event["event"] == "budget_breach" and event["data"].get("scope") == "loop"
    ]
    assert loop_breaches, "loop breach should be recorded in the trace"
    assert loop_breaches[0]["data"]["loop_id"] == "self_loop"
    assert loop_breaches[0]["data"]["action"] == "stop"

    assert result.stop_reasons == (
        {"scope": "loop", "id": "self_loop", "reason": "budget"},
    )

    run_charge_events = [
        event
        for event in result.trace
        if event["event"] == "budget_charge" and event["data"]["meter"] == "run"
    ]
    assert run_charge_events, "run-level budget charges should be tracked"
