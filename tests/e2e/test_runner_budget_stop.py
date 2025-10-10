from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pkgs.dsl.runner import FlowRunner, RunResult


@pytest.fixture()
def budgeted_runner(tmp_path: Path) -> FlowRunner:
    calls: dict[str, int] = {"count": 0}

    def counter_adapter(*, inputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        return {
            "outputs": {"value": calls["count"]},
            "cost": {"calls": 1},
            "metadata": {"iteration": calls["count"]},
        }

    runner = FlowRunner(
        adapters={"counter": counter_adapter},
        trace_path=tmp_path / "trace.jsonl",
    )
    runner._test_call_counter = calls  # type: ignore[attr-defined]
    return runner


def build_loop_spec(max_iterations: int = 10) -> dict[str, Any]:
    return {
        "globals": {
            "run_budget": {"mode": "hard", "max_calls": 10},
        },
        "nodes": [
            {
                "id": "accumulate",
                "kind": "unit",
                "inputs": {},
                "outputs": ["value"],
                "spec": {
                    "type": "tool",
                    "tool_ref": "counter",
                },
                "budget": {"mode": "hard", "max_calls": 5},
            }
        ],
        "control": [
            {
                "id": "loop_one",
                "kind": "loop",
                "target_subgraph": ["accumulate"],
                "stop": {
                    "max_iterations": max_iterations,
                    "budget": {"max_calls": 3, "breach_action": "stop"},
                },
            }
        ],
    }


def test_runner_loop_stops_when_loop_budget_hits(budgeted_runner: FlowRunner) -> None:
    spec = build_loop_spec()
    result = budgeted_runner.run(spec=spec, vars={})

    assert isinstance(result, RunResult)
    assert result.status == "halted"
    assert result.outputs["accumulate"]["value"] == 3
    assert result.stop_reasons == [
        {
            "scope": "loop:loop_one",
            "reason": "budget_exhausted",
            "details": {"metric": "calls", "limit": 3},
        }
    ]

    call_count = budgeted_runner._test_call_counter["count"]  # type: ignore[attr-defined]
    assert call_count == 3

    trace_events = budgeted_runner.trace_events
    breaches = [e for e in trace_events if e["event"] == "budget_breach"]
    assert breaches
    assert breaches[-1]["scope"] == "loop:loop_one"
    assert breaches[-1]["action"] == "stop"

    charges = [
        e
        for e in trace_events
        if e["event"] == "budget_charge" and e["scope"] == "loop:loop_one"
    ]
    assert len(charges) == 3


def test_runner_loop_raises_on_hard_breach_action_error(budgeted_runner: FlowRunner) -> None:
    spec = build_loop_spec()
    spec["control"][0]["stop"]["budget"]["breach_action"] = "error"

    with pytest.raises(RuntimeError, match="loop budget hard cap exceeded"):
        budgeted_runner.run(spec=spec, vars={})
