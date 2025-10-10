"""FlowRunner loop scope regression tests."""

from __future__ import annotations

import pytest

from codex.code.work.dsl import budget_models as bm
from codex.code.work.dsl.budget_manager import BudgetManager
from codex.code.work.dsl.flow_runner import FlowRunner, ToolAdapter
from codex.code.work.dsl.trace import TraceEventEmitter
from pkgs.dsl.policy import PolicyStack


class SequencedAdapter(ToolAdapter):
    """Adapter that returns sequential cost estimates and results."""

    def __init__(self, costs: list[dict[str, float]], results: list[object]) -> None:
        if len(costs) != len(results):
            raise ValueError("costs and results must align")
        self._costs = [dict(cost) for cost in costs]
        self._results = list(results)
        self._estimate_index = 0
        self._execute_index = 0
        self.executed: list[str] = []

    def estimate_cost(self, node: dict[str, object]) -> dict[str, float]:  # type: ignore[override]
        if self._estimate_index >= len(self._costs):
            raise IndexError("cost sequence exhausted")
        cost = dict(self._costs[self._estimate_index])
        self._estimate_index += 1
        return cost

    def execute(self, node: dict[str, object]) -> object:  # type: ignore[override]
        if self._execute_index >= len(self._results):
            raise IndexError("result sequence exhausted")
        self.executed.append(node["id"])
        result = self._results[self._execute_index]
        self._execute_index += 1
        return result


@pytest.fixture()
def base_policy_stack() -> PolicyStack:
    stack = PolicyStack(tools={"echo": {"tags": []}}, trace=None, event_sink=None)
    stack.push({"allow_tools": ["echo"]}, scope="global")
    return stack


def make_loop_budget_manager(trace: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 500}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="loop-hard",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 160}),
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="node-hard",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
            mode="hard",
            breach_action="stop",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace)


def test_loop_scope_budget_stop_emits_summary(base_policy_stack: PolicyStack) -> None:
    trace = TraceEventEmitter()
    manager = make_loop_budget_manager(trace)
    adapter = SequencedAdapter(
        costs=[{"time_ms": 70}, {"time_ms": 70}, {"time_ms": 70}],
        results=["iter-1", "iter-2", "iter-3"],
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=base_policy_stack,
        trace=trace,
    )
    nodes = [
        {
            "id": "loop-1",
            "kind": "loop",
            "target_subgraph": [
                {"id": "loop-node", "kind": "unit", "tool": "echo", "params": {}}
            ],
            "stop": {"max_iterations": 5},
        }
    ]
    executions = runner.run(flow_id="flow-loop", run_id="run-loop", nodes=nodes)
    assert [execution.result for execution in executions] == ["iter-1", "iter-2"]
    assert adapter.executed == ["loop-node", "loop-node"]
    loop_events = [evt for evt in trace.events if evt.scope_type == "loop"]
    assert any(evt.event == "budget_breach" for evt in loop_events)
    summary = next(evt for evt in trace.events if evt.event == "loop_summary")
    assert summary.payload["iterations"] == 2
    assert summary.payload["stop_reason"] == "budget_breach"


def test_loop_respects_max_iterations(base_policy_stack: PolicyStack) -> None:
    trace = TraceEventEmitter()
    # Increase loop budget to avoid breach so max_iterations is the stop reason.
    manager = BudgetManager(
        specs=[
            bm.BudgetSpec(
                name="run-soft",
                scope_type="run",
                limit=bm.CostSnapshot.from_raw({"time_ms": 500}),
                mode="soft",
                breach_action="warn",
            ),
            bm.BudgetSpec(
                name="loop-hard",
                scope_type="loop",
                limit=bm.CostSnapshot.from_raw({"time_ms": 500}),
                mode="hard",
                breach_action="stop",
            ),
            bm.BudgetSpec(
                name="node-hard",
                scope_type="node",
                limit=bm.CostSnapshot.from_raw({"time_ms": 100}),
                mode="hard",
                breach_action="stop",
            ),
        ],
        trace=trace,
    )
    adapter = SequencedAdapter(
        costs=[{"time_ms": 40}, {"time_ms": 40}, {"time_ms": 40}],
        results=["iter-1", "iter-2", "iter-3"],
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=base_policy_stack,
        trace=trace,
    )
    nodes = [
        {
            "id": "loop-iterations",
            "kind": "loop",
            "target_subgraph": [
                {"id": "loop-step", "kind": "unit", "tool": "echo", "params": {}}
            ],
            "stop": {"max_iterations": 2},
        }
    ]
    executions = runner.run(flow_id="flow-loop", run_id="run-loop-2", nodes=nodes)
    assert [execution.result for execution in executions] == ["iter-1", "iter-2"]
    summary = next(evt for evt in trace.events if evt.event == "loop_summary")
    assert summary.payload["iterations"] == 2
    assert summary.payload["stop_reason"] == "max_iterations"
