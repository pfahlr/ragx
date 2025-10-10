from collections.abc import Iterable

import pytest

from codex.code.phase3_budget_runner.dsl import budget_manager as m
from codex.code.phase3_budget_runner.dsl import budget_models as bm
from codex.code.phase3_budget_runner.dsl.flow_runner import FlowRunner, ToolAdapter
from codex.code.phase3_budget_runner.dsl.trace import TraceEventEmitter
from pkgs.dsl.policy import PolicyStack, PolicyViolationError


class FakeAdapter(ToolAdapter):
    def __init__(self, costs: Iterable[dict[str, object]], results: Iterable[object]) -> None:
        self._costs = list(costs)
        self._results = list(results)
        self.estimate_calls: list[dict[str, object]] = []
        self.execute_calls: list[dict[str, object]] = []

    def estimate_cost(self, node: dict[str, object]) -> dict[str, object]:
        self.estimate_calls.append(node)
        if not self._costs:
            raise RuntimeError("no more cost estimates")
        return self._costs.pop(0)

    def execute(self, node: dict[str, object]) -> object:
        self.execute_calls.append(node)
        if not self._results:
            raise RuntimeError("no more results")
        return self._results.pop(0)


def build_runner(trace: TraceEventEmitter) -> tuple[FlowRunner, FakeAdapter]:
    specs = [
        bm.BudgetSpec(
            name="run-total",
            scope_type="run",
            limit=bm.CostSnapshot(time_ms=400.0, tokens=400),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node-latency",
            scope_type="node",
            limit=bm.CostSnapshot(time_ms=60.0, tokens=100),
            mode="hard",
            breach_action="stop",
        ),
    ]
    manager = m.BudgetManager(specs=specs, trace=trace)
    tools = {"echo": {"tags": []}}
    policy_stack = PolicyStack(tools=tools)
    adapter = FakeAdapter(
        costs=[{"time_ms": 40.0, "tokens": 10}, {"time_ms": 80.0, "tokens": 5}],
        results=["first", "second"],
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=policy_stack,
        trace=trace,
    )
    return runner, adapter


def test_flow_runner_stops_on_node_budget_breach():
    trace = TraceEventEmitter()
    runner, adapter = build_runner(trace)

    nodes = [
        {"id": "n1", "tool": "echo", "input": "one"},
        {"id": "n2", "tool": "echo", "input": "two"},
    ]

    with pytest.raises(m.BudgetBreachError) as exc:
        runner.run(flow_id="flow", run_id="run", nodes=nodes)

    assert "node-latency" in str(exc.value)
    # first node executed successfully before breach
    assert adapter.execute_calls[0]["id"] == "n1"

    events = trace.events
    assert any(event.event == "budget_breach" and event.scope_id == "n2" for event in events)
    assert any(event.event == "node_start" and event.scope_id == "n1" for event in events)
    assert not any(event.event == "run_complete" for event in events)


def test_flow_runner_emits_policy_violation_trace_and_propagates_error():
    trace = TraceEventEmitter()
    specs = [
        bm.BudgetSpec(
            name="run-total",
            scope_type="run",
            limit=bm.CostSnapshot(time_ms=1000.0, tokens=1000),
        )
    ]
    manager = m.BudgetManager(specs=specs, trace=trace)
    policy_stack = PolicyStack(tools={"echo": {"tags": []}})
    policy_stack.push({"deny_tools": ["echo"]}, scope="test")
    adapter = FakeAdapter(costs=[{"time_ms": 10.0}], results=["ignored"])
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=policy_stack,
        trace=trace,
    )

    nodes = [{"id": "n1", "tool": "echo"}]

    with pytest.raises(PolicyViolationError):
        runner.run(flow_id="flow", run_id="run", nodes=nodes)

    events = trace.events
    assert any(event.event == "policy_violation" and event.scope_id == "n1" for event in events)
    # Ensure node scope cleaned up despite violation
    assert any(event.event == "node_start" for event in events)
    assert all(event.scope_id != "n1" or event.event != "budget_charge" for event in events)

