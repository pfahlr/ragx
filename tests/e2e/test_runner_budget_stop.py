"""End-to-end budget integration tests for FlowRunner loops."""

from __future__ import annotations

from typing import Any

import pytest

from pkgs.dsl.runner import FlowRunner, RunResult
from pkgs.dsl.trace import InMemoryTraceWriter


@pytest.fixture()
def runner() -> FlowRunner:
    return FlowRunner(trace_writer=InMemoryTraceWriter())


def _loop_budget_spec(max_iterations: int, max_calls: int) -> dict[str, Any]:
    return {
        "version": "0.1",
        "globals": {
            "tools": {
                "mock_tool": {
                    "type": "tool",
                    "pricing": {"per_call_usd": 0.2},
                }
            },
            "run_budget": {"mode": "hard", "max_usd": 10.0},
        },
        "graph": {
            "nodes": [
                {
                    "id": "step",
                    "kind": "unit",
                    "spec": {
                        "type": "tool",
                        "tool_ref": "mock_tool",
                        "mock_cost": {"usd": 0.5, "calls": 1},
                    },
                    "inputs": {},
                    "outputs": ["result"],
                    "budget": {"mode": "hard", "max_usd": 2.0},
                }
            ],
            "control": [
                {
                    "id": "loop",
                    "kind": "loop",
                    "target_subgraph": ["step"],
                    "stop": {
                        "max_iterations": max_iterations,
                        "budget": {"max_calls": max_calls, "breach_action": "stop"},
                    },
                }
            ],
        },
    }


def test_loop_stops_on_budget_breach(runner: FlowRunner) -> None:
    spec = _loop_budget_spec(max_iterations=6, max_calls=3)
    result = runner.run(spec, vars={})
    assert isinstance(result, RunResult)
    assert result.status == "ok"
    assert result.loop_iterations.get("loop") == 3
    assert result.loop_stop_reasons.get("loop") == "budget_breach"
    breach_events = [e for e in result.trace if e["event"] == "budget_breach"]
    assert breach_events and breach_events[-1]["scope"] == "loop"


def test_loop_respects_max_iterations_when_budget_unlimited(runner: FlowRunner) -> None:
    spec = _loop_budget_spec(max_iterations=2, max_calls=10)
    result = runner.run(spec, vars={})
    assert result.loop_iterations["loop"] == 2
    assert result.loop_stop_reasons["loop"] == "max_iterations"
    assert not [e for e in result.trace if e["event"] == "budget_breach"]
