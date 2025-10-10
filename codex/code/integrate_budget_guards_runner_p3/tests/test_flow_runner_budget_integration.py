"""Integration tests for FlowRunner budget and policy orchestration."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from codex.code.integrate_budget_guards_runner_p3.dsl import budget_models as bm
from codex.code.integrate_budget_guards_runner_p3.dsl.budget_manager import BudgetBreachError, BudgetManager
from codex.code.integrate_budget_guards_runner_p3.dsl.flow_runner import FlowRunner, ToolAdapter
from codex.code.integrate_budget_guards_runner_p3.dsl.trace import TraceEventEmitter
from pkgs.dsl.policy import PolicyStack, PolicyViolationError


class FakeAdapter(ToolAdapter):
    def __init__(self, cost_by_node: Mapping[str, Mapping[str, float]], results: Mapping[str, object]) -> None:
        self._cost_by_node = dict(cost_by_node)
        self._results = dict(results)
        self.executed: list[str] = []

    def estimate_cost(self, node: Mapping[str, object]) -> Mapping[str, float]:  # type: ignore[override]
        return self._cost_by_node[node["id"]]

    def execute(self, node: Mapping[str, object]) -> object:  # type: ignore[override]
        node_id = str(node["id"])
        self.executed.append(node_id)
        return self._results[node_id]


@pytest.fixture()
def trace_emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


@pytest.fixture()
def policy_stack() -> PolicyStack:
    tools = {"echo": {"tags": []}}
    return PolicyStack(tools=tools, trace=None, event_sink=None)


@pytest.fixture()
def budget_manager(trace_emitter: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node-hard",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 50}),
            mode="hard",
            breach_action="stop",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace_emitter)


def test_hard_node_breach_stops_execution(
    budget_manager: BudgetManager,
    policy_stack: PolicyStack,
    trace_emitter: TraceEventEmitter,
) -> None:
    adapter = FakeAdapter(
        cost_by_node={
            "n1": {"time_ms": 40},
            "n2": {"time_ms": 60},
        },
        results={"n1": "ok", "n2": "fail"},
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=budget_manager,
        policy_stack=policy_stack,
        trace=trace_emitter,
    )
    nodes = [
        {"id": "n1", "tool": "echo", "params": {}},
        {"id": "n2", "tool": "echo", "params": {}},
    ]

    with pytest.raises(BudgetBreachError) as excinfo:
        runner.run(flow_id="flow-1", run_id="run-1", nodes=nodes)

    assert excinfo.value.scope.scope_type == "node"
    assert adapter.executed == ["n1"]

    events = trace_emitter.events
    assert any(evt.event == "budget_breach" and evt.scope_id == "n2" for evt in events)
    assert any(evt.event == "run_start" for evt in events)


def test_soft_run_breach_warns_but_completes(trace_emitter: TraceEventEmitter) -> None:
    specs = [
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
            mode="soft",
            breach_action="warn",
        )
    ]
    manager = BudgetManager(specs=specs, trace=trace_emitter)
    policy = PolicyStack(tools={"echo": {"tags": []}}, trace=None, event_sink=None)
    adapter = FakeAdapter(
        cost_by_node={
            "n1": {"time_ms": 60},
            "n2": {"time_ms": 60},
        },
        results={"n1": "ok", "n2": "ok"},
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=policy,
        trace=trace_emitter,
    )
    nodes = [
        {"id": "n1", "tool": "echo", "params": {}},
        {"id": "n2", "tool": "echo", "params": {}},
    ]

    results = runner.run(flow_id="flow-2", run_id="run-2", nodes=nodes)

    assert [record.node_id for record in results] == ["n1", "n2"]
    spent = manager.spent(bm.ScopeKey("run", "run-2"), "run-soft")
    assert spent.time_ms == pytest.approx(120.0)

    events = [evt.event for evt in trace_emitter.events]
    assert "budget_breach" in events
    assert events.count("budget_charge") == 2


def test_policy_violation_propagates_and_traces(policy_stack: PolicyStack, trace_emitter: TraceEventEmitter) -> None:
    policy_stack.push({"deny_tools": ["echo"]}, scope="session")
    specs = [
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 200}),
            mode="hard",
            breach_action="stop",
        )
    ]
    manager = BudgetManager(specs=specs, trace=trace_emitter)
    adapter = FakeAdapter(cost_by_node={"n1": {"time_ms": 10}}, results={"n1": "ok"})
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=policy_stack,
        trace=trace_emitter,
    )

    with pytest.raises(PolicyViolationError):
        runner.run(flow_id="flow-3", run_id="run-3", nodes=[{"id": "n1", "tool": "echo"}])

    events = [evt for evt in trace_emitter.events if evt.event == "policy_violation"]
    assert events
    assert events[0].payload["tool"] == "echo"
