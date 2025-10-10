import pytest

from codex.code.work.dsl import budget_models as bm
from codex.code.work.dsl.budget_manager import BudgetManager
from codex.code.work.dsl.flow_runner import FlowRunner, ToolAdapter
from codex.code.work.dsl.trace import TraceEventEmitter
from pkgs.dsl.policy import PolicyStack, PolicyViolationError


class PolicyAwareAdapter(ToolAdapter):
    def __init__(self, *, cost: dict[str, dict[str, float]], results: dict[str, list[object]]) -> None:
        self._cost = {key: dict(value) for key, value in cost.items()}
        self._results = {key: list(values) for key, values in results.items()}
        self._cursors = {key: 0 for key in results}

    def estimate_cost(self, node: dict[str, object]) -> dict[str, float]:  # type: ignore[override]
        return self._cost[node["id"]]

    def execute(self, node: dict[str, object]) -> object:  # type: ignore[override]
        node_id = node["id"]
        cursor = self._cursors[node_id]
        self._cursors[node_id] = cursor + 1
        return self._results[node_id][cursor]


@pytest.fixture()
def trace_emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


@pytest.fixture()
def budget_manager(trace_emitter: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-soft",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 200}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node-soft",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 150}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace_emitter)


@pytest.fixture()
def policy_stack() -> PolicyStack:
    return PolicyStack(tools={"echo": {"tags": []}}, trace=None, event_sink=None)


def test_policy_events_bridge_and_violation(
    budget_manager: BudgetManager,
    policy_stack: PolicyStack,
    trace_emitter: TraceEventEmitter,
) -> None:
    adapter = PolicyAwareAdapter(
        cost={
            "allow": {"time_ms": 20.0},
            "deny": {"time_ms": 20.0},
        },
        results={
            "allow": ["ok"],
            "deny": ["no"],
        },
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=budget_manager,
        policy_stack=policy_stack,
        trace=trace_emitter,
    )
    nodes = [
        {"id": "allow", "tool": "echo", "params": {}, "policy": {"allow_tools": ["echo"]}},
        {"id": "deny", "tool": "echo", "params": {}, "policy": {"deny_tools": ["echo"]}},
    ]

    with pytest.raises(PolicyViolationError):
        runner.run(
            flow_id="flow-policy",
            run_id="run-policy",
            nodes=nodes,
            run_policy={"allow_tools": ["echo"]},
        )

    events = trace_emitter.events
    policy_events = [evt for evt in events if evt.event.startswith("policy_")]
    emitted_names = [evt.event for evt in policy_events]
    assert emitted_names.count("policy_push") >= 2
    assert "policy_pop" in emitted_names
    assert "policy_resolved" in emitted_names
    assert "policy_violation" in emitted_names

    run_push = next(
        evt for evt in policy_events if evt.event == "policy_push" and evt.scope_id == "run:run-policy"
    )
    assert run_push.payload["stack_depth"] == 1

    node_push = next(
        evt for evt in policy_events if evt.event == "policy_push" and evt.scope_id == "node:allow"
    )
    assert node_push.payload["stack_depth"] == 2

    violation = next(evt for evt in policy_events if evt.event == "policy_violation")
    assert violation.payload["tool"] == "echo"
    assert violation.payload["reasons"]

    pops = [evt for evt in policy_events if evt.event == "policy_pop" and evt.scope_id == "run:run-policy"]
    assert len(pops) == 1

    run_scope_events = [evt for evt in events if evt.scope_type == "run" and evt.event == "run_complete"]
    assert not run_scope_events
