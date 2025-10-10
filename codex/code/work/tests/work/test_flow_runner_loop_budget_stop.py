import pytest

from codex.code.work.dsl import budget_models as bm
from codex.code.work.dsl.budget_manager import BudgetManager
from codex.code.work.dsl.flow_runner import FlowRunner, ToolAdapter
from codex.code.work.dsl.trace import TraceEventEmitter
from pkgs.dsl.policy import PolicyStack


class LoopingAdapter(ToolAdapter):
    """Deterministic adapter that returns pre-seeded results per node id."""

    def __init__(
        self,
        cost_by_node: dict[str, dict[str, float]],
        results_by_node: dict[str, list[object]],
    ) -> None:
        self._cost = cost_by_node
        self._results = {key: list(values) for key, values in results_by_node.items()}
        self._cursors: dict[str, int] = {key: 0 for key in results_by_node}
        self.executed: list[str] = []

    def estimate_cost(self, node: dict[str, object]) -> dict[str, float]:  # type: ignore[override]
        return self._cost[node["id"]]

    def execute(self, node: dict[str, object]) -> object:  # type: ignore[override]
        node_id = node["id"]
        cursor = self._cursors[node_id]
        if cursor >= len(self._results[node_id]):
            raise RuntimeError(f"no result remaining for node {node_id}")
        self._cursors[node_id] = cursor + 1
        self.executed.append(node_id)
        return self._results[node_id][cursor]


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
            limit=bm.CostSnapshot.from_raw({"time_ms": 500}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="loop-stop",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 120}),
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
    return BudgetManager(specs=specs, trace=trace_emitter)


def test_loop_budget_stop_emits_summary_and_continues_run(
    budget_manager: BudgetManager,
    policy_stack: PolicyStack,
    trace_emitter: TraceEventEmitter,
) -> None:
    adapter = LoopingAdapter(
        cost_by_node={
            "inner": {"time_ms": 70.0},
            "after": {"time_ms": 10.0},
        },
        results_by_node={
            "inner": ["loop-1" for _ in range(3)],
            "after": ["done"],
        },
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=budget_manager,
        policy_stack=policy_stack,
        trace=trace_emitter,
    )
    nodes = [
        {
            "id": "loop-1",
            "type": "loop",
            "body": [
                {"id": "inner", "tool": "echo", "params": {}},
            ],
        },
        {"id": "after", "tool": "echo", "params": {}},
    ]

    executions = runner.run(
        flow_id="flow-loop",
        run_id="run-loop",
        nodes=nodes,
        run_policy={"allow_tools": ["echo"]},
    )

    executed_ids = [record.node_id for record in executions]
    assert executed_ids == ["inner", "after"]
    assert adapter.executed == ["inner", "after"]

    loop_events = [evt for evt in trace_emitter.events if evt.event == "loop_summary"]
    assert len(loop_events) == 1
    summary = loop_events[0]
    assert summary.scope_type == "loop"
    assert summary.scope_id == "loop-1"
    assert summary.payload["iterations"] == 1
    assert summary.payload["stop_reason"] == "budget_stop"
    breach = summary.payload.get("breach")
    assert breach is not None
    assert breach["spec_name"] == "loop-stop"

    breach_events = [evt for evt in trace_emitter.events if evt.event == "budget_breach"]
    assert any(evt.scope_type == "loop" and evt.scope_id == "loop-1" for evt in breach_events)

    run_complete = [evt for evt in trace_emitter.events if evt.event == "run_complete"]
    assert len(run_complete) == 1
    assert run_complete[0].payload["flow_id"] == "flow-loop"
