from __future__ import annotations

from collections.abc import Callable

import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetManager
from pkgs.dsl.flow_runner import FlowRunner, ToolAdapter
from pkgs.dsl.policy import PolicyStack, PolicyViolationError
from pkgs.dsl.trace import TraceEventEmitter


class LoopAwareAdapter(ToolAdapter):
    """Adapter that records iteration-aware executions."""

    def __init__(
        self,
        *,
        cost_ms: float,
        result_factory: Callable[[dict[str, object]], object] | None = None,
    ) -> None:  # type: ignore[override]
        self._cost_snapshot = {"time_ms": cost_ms}
        self._result_factory = result_factory or (lambda node: node["id"])
        self.executed: list[str] = []

    def estimate_cost(self, node: dict[str, object]) -> dict[str, float]:  # type: ignore[override]
        return dict(self._cost_snapshot)

    def execute(self, node: dict[str, object]) -> object:  # type: ignore[override]
        iteration = node.get("iteration")
        token = f"{node['id']}@{iteration}" if iteration is not None else str(node["id"])
        self.executed.append(token)
        return self._result_factory(node)


@pytest.fixture()
def trace_emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


@pytest.fixture()
def policy_stack() -> PolicyStack:
    tools = {"echo": {"tags": []}}
    return PolicyStack(tools=tools, trace=None, event_sink=None)


def _loop_specs(trace: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 500}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="loop-stop",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 80}),
            mode="soft",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="node-hard",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 200}),
            mode="hard",
            breach_action="stop",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace)


def test_loop_scope_stop_emits_trace_and_breaks_iterations(
    trace_emitter: TraceEventEmitter,
    policy_stack: PolicyStack,
) -> None:
    manager = _loop_specs(trace_emitter)
    adapter = LoopAwareAdapter(cost_ms=40.0)
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=policy_stack,
        trace=trace_emitter,
    )

    loop_node = {
        "id": "loop-1",
        "kind": "loop",
        "body": [
            {"id": "node-a", "tool": "echo", "params": {}},
        ],
        "max_iterations": 5,
    }

    results = runner.run(
        flow_id="flow-loop",
        run_id="run-loop",
        nodes=[loop_node],
    )

    assert [exec.node_id for exec in results] == ["node-a", "node-a"]
    assert [exec.iteration for exec in results] == [1, 2]
    assert adapter.executed == ["node-a@1", "node-a@2"]

    events = [(evt.event, evt.scope_type) for evt in trace_emitter.events]
    assert ("loop_start", "loop") in events
    assert any(
        evt.event == "loop_stop" and evt.scope_id == "loop-1"
        for evt in trace_emitter.events
    )
    assert adapter.executed.count("node-a@1") == 1


def test_policy_violation_leaves_budgets_uncharged(
    trace_emitter: TraceEventEmitter,
    policy_stack: PolicyStack,
) -> None:
    manager = _loop_specs(trace_emitter)
    policy_stack.push({"deny_tools": ["echo"]}, scope="test")
    adapter = LoopAwareAdapter(cost_ms=10.0)
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=policy_stack,
        trace=trace_emitter,
    )
    nodes = [{"id": "n1", "tool": "echo", "params": {}}]

    try:
        with pytest.raises(PolicyViolationError):
            runner.run(flow_id="flow-policy", run_id="run-policy", nodes=nodes)
    finally:
        policy_stack.pop("test")

    run_scope = bm.ScopeKey(scope_type="run", scope_id="run-policy")
    assert manager.spent(run_scope, "run-soft") == bm.CostSnapshot.zero()
    assert adapter.executed == []

    events = [evt.event for evt in trace_emitter.events]
    assert "policy_violation" in events
    assert "budget_charge" not in events
