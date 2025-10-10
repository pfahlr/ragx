"""Integration-style coverage for FlowRunner policy/budget interplay."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetBreachError, BudgetManager
from pkgs.dsl.flow_runner import FlowRunner, ToolAdapter
from pkgs.dsl.policy import PolicyStack, PolicyTraceRecorder, PolicyViolationError
from pkgs.dsl.trace import TraceEventEmitter


class RecordingAdapter(ToolAdapter):
    """Adapter that returns canned results and records estimate/execute calls."""

    def __init__(
        self,
        *,
        estimate_costs: Iterable[Mapping[str, float]],
        results: Iterable[Any] | None = None,
    ) -> None:
        self._estimate_costs = deque(dict(cost) for cost in estimate_costs)
        self._results = deque(results or [])
        self.estimated: list[Mapping[str, float]] = []
        self.executed: list[Mapping[str, object]] = []

    def estimate_cost(self, node: Mapping[str, object]) -> Mapping[str, float]:  # type: ignore[override]
        if len(self._estimate_costs) > 1:
            cost = dict(self._estimate_costs.popleft())
        else:
            cost = dict(self._estimate_costs[0])
        self.estimated.append(cost)
        return cost

    def execute(self, node: Mapping[str, object]) -> Any:  # type: ignore[override]
        self.executed.append(dict(node))
        if self._results:
            return self._results.popleft()
        return {"ok": node.get("id")}


def _policy_stack(trace: PolicyTraceRecorder | None = None) -> PolicyStack:
    tools: dict[str, Mapping[str, object]] = {
        "echo": {"tags": ["default"]},
        "alt": {"tags": []},
    }
    return PolicyStack(tools=tools, trace=trace)


def _budget_manager(trace: TraceEventEmitter, *, loop_warn: bool = False) -> BudgetManager:
    loop_limit = 60 if not loop_warn else 30
    loop_spec = bm.BudgetSpec(
        name="loop-soft",
        scope_type="loop",
        limit=bm.CostSnapshot.from_raw({"time_ms": loop_limit}),
        mode="soft",
        breach_action="stop" if not loop_warn else "warn",
    )
    specs = [
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 120}),
            mode="hard",
            breach_action="stop",
        ),
        loop_spec,
        bm.BudgetSpec(
            name="node-soft",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 40}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace)


def _runner(
    *,
    adapter: ToolAdapter,
    trace: TraceEventEmitter,
    loop_warn: bool = False,
    policy_trace: PolicyTraceRecorder | None = None,
) -> FlowRunner:
    manager = _budget_manager(trace, loop_warn=loop_warn)
    stack = _policy_stack(policy_trace)
    return FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=stack,
        trace=trace,
    )


def _loop_node(
    loop_id: str,
    *,
    body: list[Mapping[str, object]],
    max_iterations: int | None = None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {"id": loop_id, "kind": "loop", "body": body}
    if max_iterations is not None:
        payload["max_iterations"] = max_iterations
    return payload


def _unit_node(node_id: str, *, tool: str = "echo") -> Mapping[str, object]:
    return {"id": node_id, "tool": tool, "params": {}}


def test_loop_scope_budget_stop_emits_breach_and_loop_stop() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 35}] * 3)
    runner = _runner(adapter=adapter, trace=trace)

    loop = _loop_node(
        "loop-1",
        body=[_unit_node("node-a"), _unit_node("node-b")],
        max_iterations=3,
    )
    executions = runner.run(flow_id="flow", run_id="run", nodes=[loop])

    # Only the first node completes because the second preview triggers a stop
    assert len(executions) == 1
    events = [evt.event for evt in trace.events if evt.scope_type == "loop"]
    assert "loop_start" in events
    assert "loop_stop" in events
    assert any(
        evt.event == "budget_breach"
        for evt in trace.events
        if evt.scope_type == "loop"
    )


def test_loop_soft_warn_allows_progress() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 20}] * 3)
    runner = _runner(adapter=adapter, trace=trace, loop_warn=True)

    loop = _loop_node(
        "loop-soft",
        body=[_unit_node("node-a")],
        max_iterations=3,
    )
    executions = runner.run(flow_id="flow", run_id="run-soft", nodes=[loop])

    assert len(executions) == 3
    breach_events = [
        evt
        for evt in trace.events
        if evt.event == "budget_breach" and evt.scope_type == "loop"
    ]
    assert breach_events, "soft budget should emit breach events"
    assert not any(
        evt.event == "loop_stop"
        for evt in trace.events
        if evt.scope_type == "loop"
    )


def test_nested_loop_budget_stop_propagation() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 35}] * 6)
    runner = _runner(adapter=adapter, trace=trace)

    first_loop = _loop_node(
        "loop-first",
        body=[_unit_node("node-a")],
        max_iterations=3,
    )
    second_loop = _loop_node(
        "loop-second",
        body=[_unit_node("node-b")],
        max_iterations=2,
    )

    executions = runner.run(flow_id="flow", run_id="run-nested", nodes=[first_loop, second_loop])

    loop_stop_events = [evt for evt in trace.events if evt.event == "loop_stop"]
    assert any(evt.scope_id == "loop-first" for evt in loop_stop_events)
    assert any(evt.scope_id == "loop-second" for evt in loop_stop_events)
    assert any(exec.loop_id == "loop-second" for exec in executions)


def test_policy_violation_prevents_budget_charges_and_emits_ordered_trace() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 5}])
    runner = _runner(adapter=adapter, trace=trace, policy_trace=policy_trace)

    runner._policies.push({"deny_tools": ["echo"]}, scope="deny")
    try:
        with pytest.raises(PolicyViolationError):
            runner.run(flow_id="flow", run_id="run-policy", nodes=[_unit_node("node-a")])
    finally:
        runner._policies.pop("deny")

    assert not any(evt.event == "budget_charge" for evt in trace.events)
    policy_events = [evt.event for evt in policy_trace.events]
    assert "policy_resolved" in policy_events
    assert policy_events.index("policy_resolved") < policy_events.index("policy_violation")


def test_policy_and_budget_violation_coexistence_prioritises_policy() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 90}])
    runner = _runner(adapter=adapter, trace=trace, policy_trace=policy_trace)

    runner._policies.push({"deny_tools": ["echo"]}, scope="deny")
    try:
        with pytest.raises(PolicyViolationError):
            runner.run(flow_id="flow", run_id="run-block", nodes=[_unit_node("node-a")])
    finally:
        runner._policies.pop("deny")

    assert not any(evt.event == "budget_charge" for evt in trace.events)
    assert any(evt.event == "policy_violation" for evt in policy_trace.events)


def test_flow_runner_emits_combined_policy_budget_traces() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 30}, {"time_ms": 50}, {"time_ms": 30}])
    runner = _runner(adapter=adapter, trace=trace, policy_trace=policy_trace)

    nodes = [
        _unit_node("node-a"),
        _unit_node("node-b"),
        _unit_node("node-c"),
    ]
    runner.run(flow_id="flow", run_id="run-trace", nodes=nodes)

    budget_events = [evt for evt in trace.events if evt.event.startswith("budget_")]
    assert any(evt.event == "budget_charge" for evt in budget_events)
    assert any(evt.event == "budget_breach" for evt in budget_events)
    resolved_count = sum(
        1 for evt in policy_trace.events if evt.event == "policy_resolved"
    )
    assert resolved_count == len(nodes)


def test_flow_runner_policy_trace_interleaving_preserves_order() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 5}] * 2)
    runner = _runner(adapter=adapter, trace=trace, policy_trace=policy_trace)

    runner._policies.push({"allow_tools": ["echo"]}, scope="session")
    try:
        executions = runner.run(
            flow_id="flow",
            run_id="run-allow",
            nodes=[_unit_node("node-a"), _unit_node("node-b")],
        )
    finally:
        runner._policies.pop("session")

    assert len(executions) == 2
    policy_events = [
        evt
        for evt in policy_trace.events
        if evt.event in {"policy_resolved", "policy_violation"}
    ]
    # policy_resolved should precede any violation and maintain stable ordering
    resolved_indices = [
        i
        for i, evt in enumerate(policy_events)
        if evt.event == "policy_resolved"
    ]
    assert resolved_indices == sorted(resolved_indices)

    node_charge_events = [
        evt
        for evt in trace.events
        if evt.event == "budget_charge" and evt.scope_type == "node"
    ]
    assert len(node_charge_events) == len(resolved_indices)
    for decision_event, charge_event in zip(
        resolved_indices,
        node_charge_events,
        strict=False,
    ):
        assert policy_events[decision_event].event == "policy_resolved"
        assert charge_event.scope_type == "node"


def test_run_scope_hard_budget_stops_all_loops() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 30}] * 4)
    specs = [
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 50}),
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="loop-soft",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 120}),
            mode="soft",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="node-soft",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 60}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    manager = BudgetManager(specs=specs, trace=trace)
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=_policy_stack(policy_trace),
        trace=trace,
    )

    loop = _loop_node(
        "loop-run",
        body=[_unit_node("node-a")],
        max_iterations=4,
    )
    with pytest.raises(BudgetBreachError):
        runner.run(flow_id="flow", run_id="run-stop", nodes=[loop])

    run_events = [evt for evt in trace.events if evt.scope_type == "run"]
    assert any(evt.event == "budget_breach" for evt in run_events)
    assert not any(evt.event == "loop_stop" for evt in trace.events if evt.scope_type == "loop")
