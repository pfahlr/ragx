from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from pkgs.dsl.runner import FlowRunner, RunResult


@dataclass
class FakeToolResult:
    outputs: Mapping[str, Any]
    cost: Mapping[str, float | int]


class FakeToolAdapter:
    def __init__(
        self,
        *,
        cost: Mapping[str, float | int],
        output: Mapping[str, Any] | None = None,
    ) -> None:
        default_output = {"result": "ok"}
        self._cost = dict(cost)
        self._output = dict(output or default_output)
        self.calls = 0

    def estimate_cost(
        self,
        node: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> Mapping[str, float | int]:
        return dict(self._cost)

    def execute(self, node: Mapping[str, Any], inputs: Mapping[str, Any]) -> FakeToolResult:
        self.calls += 1
        return FakeToolResult(outputs=self._output, cost=self._cost)


@pytest.fixture()
def runner() -> FlowRunner:
    adapters = {
        "expensive": FakeToolAdapter(cost={"usd": 3.0}),
        "cheap": FakeToolAdapter(cost={"usd": 1.0}),
        "loop-tool": FakeToolAdapter(cost={"calls": 1}),
    }
    return FlowRunner(adapters=adapters, run_id_factory=lambda: "run-budget")


def test_run_budget_hard_limit_triggers_error(runner: FlowRunner) -> None:
    spec = {
        "globals": {"run_budget": {"max_usd": 5.0, "mode": "hard"}},
        "nodes": [
            {
                "id": "first",
                "kind": "unit",
                "spec": {"type": "tool", "tool_ref": "expensive", "args": {}},
                "outputs": ["result"],
            },
            {
                "id": "second",
                "kind": "unit",
                "spec": {"type": "tool", "tool_ref": "expensive", "args": {}},
                "outputs": ["result"],
            },
        ],
    }

    result = runner.run(spec, vars={})

    assert isinstance(result, RunResult)
    assert result.status == "error"
    assert result.outputs.get("first", {}).get("result") == "ok"
    assert "second" not in result.outputs  # halted before second node completed

    breach_events = [evt for evt in runner.trace_events if evt["event"] == "budget_breach"]
    assert breach_events, "Expected a budget breach to be recorded"
    breach = breach_events[-1]
    assert breach["scope"] == "run"
    assert breach["level"] == "hard"
    assert breach["metric"] == "usd"


def test_soft_node_budget_emits_warning_and_continues(runner: FlowRunner) -> None:
    spec = {
        "globals": {},
        "nodes": [
            {
                "id": "soft-node",
                "kind": "unit",
                "budget": {"mode": "soft", "max_usd": 0.5},
                "spec": {"type": "tool", "tool_ref": "cheap", "args": {}},
                "outputs": ["value"],
            }
        ],
    }

    result = runner.run(spec, vars={})

    assert result.status == "ok"
    assert result.outputs["soft-node"]["result"] == "ok"
    warnings = [evt for evt in runner.trace_events if evt["event"] == "budget_warning"]
    assert warnings, "Soft budget breaches should emit warnings"
    assert warnings[0]["scope"] == "node:soft-node"


def test_loop_budget_with_stop_action_halts_loop_without_error() -> None:
    loop_runner = FlowRunner(
        adapters={"loop-tool": FakeToolAdapter(cost={"calls": 1})},
        run_id_factory=lambda: "loop-run",
    )
    spec = {
        "globals": {},
        "nodes": [
            {
                "id": "loop",
                "kind": "loop",
                "target_subgraph": ["body"],
                "stop": {"budget": {"max_calls": 3, "breach_action": "stop"}},
            },
            {
                "id": "body",
                "kind": "unit",
                "spec": {"type": "tool", "tool_ref": "loop-tool", "args": {}},
                "outputs": ["value"],
            },
        ],
    }

    result = loop_runner.run(spec, vars={})

    assert result.status == "ok"
    assert loop_runner.adapters["loop-tool"].calls == 3
    stop_events = [evt for evt in loop_runner.trace_events if evt["event"] == "loop_stop"]
    assert stop_events, "Loop should emit a stop event"
    stop = stop_events[-1]
    assert stop["loop_id"] == "loop"
    assert stop["reason"] == "budget_stop"


def test_loop_budget_error_action_propagates() -> None:
    loop_runner = FlowRunner(
        adapters={"loop-tool": FakeToolAdapter(cost={"calls": 2})},
        run_id_factory=lambda: "loop-run-error",
    )
    spec = {
        "globals": {"run_budget": {"max_calls": 10}},
        "nodes": [
            {
                "id": "loop",
                "kind": "loop",
                "target_subgraph": ["body"],
                "stop": {"budget": {"max_calls": 3, "breach_action": "error"}},
            },
            {
                "id": "body",
                "kind": "unit",
                "spec": {"type": "tool", "tool_ref": "loop-tool", "args": {}},
                "outputs": ["value"],
            },
        ],
    }

    result = loop_runner.run(spec, vars={})

    assert result.status == "error"
    breach_events = [evt for evt in loop_runner.trace_events if evt["event"] == "budget_breach"]
    assert any(evt["scope"] == "loop:loop" for evt in breach_events)

