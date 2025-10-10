from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "budget_integration.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("budget_integration", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    sys.modules[spec.name] = module
    loader.exec_module(module)  # type: ignore[assignment]
    return module


@pytest.fixture(scope="module")
def budget_module():
    return load_module()


class DummyAdapter:
    def __init__(self, module, estimates, actuals):
        self.CostSnapshot = module.CostSnapshot
        self.estimates = estimates
        self.actuals = actuals

    def estimate(self, node):
        return self.CostSnapshot.from_mapping(self.estimates[node["id"]])

    def execute(self, node):
        result = node.get("value", node["id"])
        cost = self.CostSnapshot.from_mapping(self.actuals[node["id"]])
        return result, cost


def build_runner(module, adapter, trace_writer=None):
    FlowRunner = module.FlowRunner
    return FlowRunner(adapter=adapter, trace_writer=trace_writer)


def test_flow_runner_emits_warnings_and_completes(budget_module):
    BudgetSpec = budget_module.BudgetSpec
    CostSnapshot = budget_module.CostSnapshot
    ListTraceWriter = budget_module.ListTraceWriter

    adapter = DummyAdapter(
        budget_module,
        estimates={"a": {"time_ms": 400}, "b": {"time_ms": 300}},
        actuals={"a": {"time_ms": 450}, "b": {"time_ms": 350}},
    )
    writer = ListTraceWriter()
    runner = build_runner(budget_module, adapter, writer)

    budgets = {
        "run": BudgetSpec(scope="run", limits={"time_ms": 700}, breach_action="warn"),
        "node:a": BudgetSpec(scope="node", limits={"time_ms": 500}, breach_action="warn"),
        "node:b": BudgetSpec(scope="node", limits={"time_ms": 400}, breach_action="warn"),
    }

    nodes = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
    results = runner.run(nodes, budgets)

    assert results == [1, 2]
    trace_events = [event["event"] for event in runner.get_trace()]
    assert trace_events.count("budget_breach") >= 1
    run_warnings = [e for e in runner.get_trace() if e["scope"] == "run" and e["event"] == "budget_breach"]
    assert run_warnings
    assert run_warnings[0]["data"]["breach_action"] == "warn"


def test_flow_runner_stops_on_hard_breach(budget_module):
    BudgetSpec = budget_module.BudgetSpec
    CostSnapshot = budget_module.CostSnapshot
    ListTraceWriter = budget_module.ListTraceWriter

    adapter = DummyAdapter(
        budget_module,
        estimates={"x": {"time_ms": 200}, "y": {"time_ms": 250}, "z": {"time_ms": 300}},
        actuals={"x": {"time_ms": 190}, "y": {"time_ms": 260}, "z": {"time_ms": 400}},
    )
    writer = ListTraceWriter()
    runner = build_runner(budget_module, adapter, writer)

    budgets = {
        "run": BudgetSpec(scope="run", limits={"time_ms": 700}, breach_action="stop"),
        "node:x": BudgetSpec(scope="node", limits={"time_ms": 250}, breach_action="warn"),
        "node:y": BudgetSpec(scope="node", limits={"time_ms": 250}, breach_action="stop"),
        "node:z": BudgetSpec(scope="node", limits={"time_ms": 350}, breach_action="stop"),
    }

    nodes = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    results = runner.run(nodes, budgets)

    assert results == ["x", "y"]
    # Ensure run scope recorded hard breach and stop event
    breach_events = [e for e in runner.get_trace() if e["event"] == "budget_breach"]
    assert breach_events
    assert any(e["data"]["breach_action"] == "stop" for e in breach_events)
    assert runner.manager.should_stop("run") is True
