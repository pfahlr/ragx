"""Phase 6 integration tests for ``pkgs.dsl.flow_runner``."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetBreachError, BudgetManager
from pkgs.dsl.flow_runner import FlowRunner, NodeExecution, ToolAdapter
from pkgs.dsl.policy import PolicyStack, PolicyTraceRecorder, PolicyViolationError
from pkgs.dsl.trace import TraceEventEmitter


class RecordingAdapter(ToolAdapter):
    """Adapter that records estimate/execute calls and returns canned results."""

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
        payload = dict(node)
        self.executed.append(payload)
        if self._results:
            return self._results.popleft()
        return {"ok": node.get("id")}


def _policy_stack(trace: PolicyTraceRecorder | None = None) -> PolicyStack:
    tools: dict[str, Mapping[str, object]] = {
        "echo": {"tags": ["default"]},
        "alt": {"tags": []},
    }
    return PolicyStack(tools=tools, trace=trace)


def _budget_manager(
    trace: TraceEventEmitter, *, loop_warn: bool = False, run_limit: int = 120
) -> BudgetManager:
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
            limit=bm.CostSnapshot.from_raw({"time_ms": run_limit}),
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
    run_limit: int = 120,
) -> FlowRunner:
    manager = _budget_manager(trace, loop_warn=loop_warn, run_limit=run_limit)
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


def _collect_loop_events(trace: TraceEventEmitter, event: str) -> list[str]:
    return [evt.scope_id for evt in trace.events if evt.event == event]


def _executions_for_loop(executions: list[NodeExecution], loop_id: str) -> list[NodeExecution]:
    return [execution for execution in executions if execution.loop_id == loop_id]


def test_loop_scope_budget_stop_emits_breach_and_loop_stop() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 35}] * 3)
    runner = _runner(adapter=adapter, trace=trace)

    loop = _loop_node(
        "loop-hard",
        body=[_unit_node("node-a"), _unit_node("node-b")],
        max_iterations=3,
    )

    executions = runner.run(flow_id="flow", run_id="run", nodes=[loop])

    assert len(executions) == 1
    loop_stops = _collect_loop_events(trace, "loop_stop")
    assert loop_stops == ["loop-hard"]
    breaches = [
        evt
        for evt in trace.events
        if evt.event == "budget_breach" and evt.scope_type == "loop"
    ]
    assert breaches, "Loop breach event expected when stop action triggers"


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
    assert not _collect_loop_events(trace, "loop_stop")
    breaches = [
        evt
        for evt in trace.events
        if evt.event == "budget_breach" and evt.scope_type == "loop"
    ]
    assert breaches, "Soft budget should emit breach events"


def test_nested_loop_budget_stop_propagation() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(
        estimate_costs=[{"time_ms": 15}, {"time_ms": 55}, {"time_ms": 55}]
    )
    runner = _runner(adapter=adapter, trace=trace, run_limit=240)

    inner_loop = _loop_node(
        "inner-loop",
        body=[_unit_node("inner-node")],
        max_iterations=5,
    )
    outer_loop = _loop_node(
        "outer-loop",
        body=[_unit_node("outer-node"), inner_loop],
        max_iterations=2,
    )

    executions = runner.run(flow_id="flow", run_id="run-nested", nodes=[outer_loop])

    stop_events = _collect_loop_events(trace, "loop_stop")
    assert "inner-loop" in stop_events
    assert "outer-loop" in stop_events
    inner_executions = _executions_for_loop(executions, "inner-loop")
    assert inner_executions, "Inner loop nodes should emit execution records"


def test_run_scope_hard_budget_stops_all_loops() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 90}] * 3)
    runner = _runner(adapter=adapter, trace=trace, run_limit=80)

    loops = [
        _loop_node("loop-a", body=[_unit_node("node-a")]),
        _loop_node("loop-b", body=[_unit_node("node-b")]),
    ]

    with pytest.raises(BudgetBreachError):
        runner.run(flow_id="flow", run_id="run-stop", nodes=loops)

    loop_stops = _collect_loop_events(trace, "loop_stop")
    assert not loop_stops
    assert any(
        evt.event == "budget_breach" and evt.scope_type == "run"
        for evt in trace.events
    )


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
    events = [evt.event for evt in policy_trace.events]
    assert "policy_resolved" in events
    assert events.index("policy_resolved") < events.index("policy_violation")


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


def test_flow_runner_policy_trace_interleaving_preserves_order() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(
        estimate_costs=[{"time_ms": 15}, {"time_ms": 15}, {"time_ms": 15}]
    )
    runner = _runner(adapter=adapter, trace=trace, policy_trace=policy_trace)

    nodes = [_unit_node("node-a"), _unit_node("node-b"), _unit_node("node-c")]
    runner.run(flow_id="flow", run_id="run-trace", nodes=nodes)

    budget_events = [evt for evt in trace.events if evt.event.startswith("budget_")]
    policy_events = [evt.event for evt in policy_trace.events]
    assert all(evt.event == "policy_resolved" for evt in policy_trace.events)
    assert len(policy_events) == 3
    for budget_event, policy_event in zip(budget_events, policy_events, strict=False):
        assert policy_event == "policy_resolved"
        assert budget_event.event in {"budget_charge", "budget_breach"}


def test_flow_runner_emits_combined_policy_budget_traces() -> None:
    trace = TraceEventEmitter()
    policy_trace = PolicyTraceRecorder()
    adapter = RecordingAdapter(
        estimate_costs=[{"time_ms": 30}, {"time_ms": 50}, {"time_ms": 30}]
    )
    runner = _runner(adapter=adapter, trace=trace, policy_trace=policy_trace)

    nodes = [
        _unit_node("node-a"),
        _unit_node("node-b"),
        _unit_node("node-c"),
    ]
    runner.run(flow_id="flow", run_id="run-mixed", nodes=nodes)

    assert any(evt.event == "budget_breach" for evt in trace.events)
    assert any(evt.event == "policy_resolved" for evt in policy_trace.events)
