"""Integration tests for FlowRunner budget and policy orchestration."""

from __future__ import annotations

import pytest

from pkgs.dsl.policy import PolicyStack, PolicyViolationError

from . import load_module

budget_models = load_module("budget_models")
budget_manager_mod = load_module("budget_manager")
trace_mod = load_module("trace_emitter")
flow_runner_mod = load_module("flow_runner")

BudgetSpec = budget_models.BudgetSpec
BudgetMode = budget_models.BudgetMode
BudgetManager = budget_manager_mod.BudgetManager
BudgetBreachError = budget_manager_mod.BudgetBreachError
TraceEventEmitter = trace_mod.TraceEventEmitter
FlowRunner = flow_runner_mod.FlowRunner


class RecordingAdapter:
    def __init__(self, *, estimates: dict[str, dict[str, float]], actuals: dict[str, dict[str, float]]) -> None:
        self._estimates = estimates
        self._actuals = actuals
        self.executed: list[str] = []

    def estimate(self, node: dict[str, object]) -> dict[str, float]:
        return dict(self._estimates[node["id"]])

    def execute(self, node: dict[str, object]) -> tuple[dict[str, object], dict[str, float]]:
        self.executed.append(node["id"])
        output = {"result": f"executed-{node['id']}"}
        return output, dict(self._actuals[node["id"]])


class EventCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def __call__(self, event: dict[str, object]) -> None:
        self.events.append(event)


def _setup_policy_stack(emitter: TraceEventEmitter) -> PolicyStack:
    tools = {"echo": {"tags": []}, "other": {"tags": []}}
    stack = PolicyStack(tools=tools, event_sink=emitter.policy_event_sink)
    stack.push({"allow_tools": ["echo"]}, scope="flow", source="test")
    return stack


def _build_runner(*, adapter: RecordingAdapter, emitter: TraceEventEmitter) -> FlowRunner:
    manager = BudgetManager(emitter=emitter)
    policy_stack = _setup_policy_stack(emitter)
    return FlowRunner(
        adapter=adapter,
        policy_stack=policy_stack,
        budget_manager=manager,
        emitter=emitter,
    )


def test_flow_runner_emits_budget_and_policy_traces() -> None:
    collector = EventCollector()
    emitter = TraceEventEmitter(
        writer=collector,
        flow_id="flow-1",
        run_id="run-1",
        metadata={"tenant": "demo"},
    )
    adapter = RecordingAdapter(
        estimates={"node-1": {"tokens": 5}},
        actuals={"node-1": {"tokens": 4}},
    )
    runner = _build_runner(adapter=adapter, emitter=emitter)

    flow = {
        "run_scope": BudgetSpec(limits={"tokens": 20}, mode=BudgetMode.STOP),
        "nodes": [
            {
                "id": "node-1",
                "tool": "echo",
                "budget": BudgetSpec(limits={"tokens": 10}, mode=BudgetMode.WARN),
            }
        ],
    }

    outputs = runner.run(flow)

    assert adapter.executed == ["node-1"]
    assert outputs["node-1"]["result"] == "executed-node-1"

    policy_events = [e for e in collector.events if e["event"].startswith("policy_")]
    assert any(event["event"] == "policy_resolved" for event in policy_events)

    budget_events = [e for e in collector.events if e["event"].startswith("budget_")]
    assert any(event["event"] == "budget_charge" for event in budget_events)


def test_flow_runner_stops_on_budget_breach() -> None:
    collector = EventCollector()
    emitter = TraceEventEmitter(
        writer=collector,
        flow_id="flow-2",
        run_id="run-2",
        metadata={},
    )
    adapter = RecordingAdapter(
        estimates={"node-1": {"tokens": 9}},
        actuals={"node-1": {"tokens": 15}},
    )
    runner = _build_runner(adapter=adapter, emitter=emitter)

    flow = {
        "run_scope": BudgetSpec(limits={"tokens": 10}, mode=BudgetMode.STOP),
        "nodes": [
            {
                "id": "node-1",
                "tool": "echo",
                "budget": BudgetSpec(limits={"tokens": 8}, mode=BudgetMode.STOP),
            }
        ],
    }

    with pytest.raises(BudgetBreachError):
        runner.run(flow)

    stop_events = [e for e in collector.events if e["event"] == "loop_stop"]
    assert stop_events and stop_events[-1]["stop_reason"] == "budget_breach"


def test_flow_runner_emits_loop_stop_on_policy_violation() -> None:
    collector = EventCollector()
    emitter = TraceEventEmitter(
        writer=collector,
        flow_id="flow-3",
        run_id="run-3",
        metadata={},
    )
    adapter = RecordingAdapter(
        estimates={"node-1": {"tokens": 1}},
        actuals={"node-1": {"tokens": 1}},
    )
    manager = BudgetManager(emitter=emitter)
    policy_stack = PolicyStack(
        tools={"echo": {"tags": []}},
        event_sink=emitter.policy_event_sink,
    )
    policy_stack.push({"deny_tools": ["echo"]}, scope="flow", source="test")

    runner = FlowRunner(
        adapter=adapter,
        policy_stack=policy_stack,
        budget_manager=manager,
        emitter=emitter,
    )

    flow = {
        "run_scope": BudgetSpec(limits={"tokens": 10}, mode=BudgetMode.STOP),
        "nodes": [
            {
                "id": "node-1",
                "tool": "echo",
                "budget": BudgetSpec(limits={"tokens": 5}, mode=BudgetMode.STOP),
            }
        ],
    }

    with pytest.raises(PolicyViolationError):
        runner.run(flow)

    stop_events = [e for e in collector.events if e["event"] == "loop_stop"]
    assert stop_events and stop_events[-1]["stop_reason"] == "policy_violation"

