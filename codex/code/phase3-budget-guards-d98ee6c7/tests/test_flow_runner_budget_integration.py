from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from pkgs.dsl.budget import (
    BreachAction,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
)
from pkgs.dsl.policy import PolicyStack
from pkgs.dsl.runner import FlowNode, FlowRunner, RunContext
from pkgs.dsl.trace import TraceEventEmitter


@dataclass
class FakeAdapter:
    estimate_ms: float
    actual_ms: float
    output: Any = None

    def estimate_cost(self, context: RunContext, node: FlowNode) -> CostSnapshot:
        return CostSnapshot(milliseconds=self.estimate_ms)

    def execute(self, context: RunContext, node: FlowNode) -> Any:
        return self.output or {"node": node.node_id, "result": "ok"}

    def actual_cost_snapshot(self) -> CostSnapshot:
        return CostSnapshot(milliseconds=self.actual_ms)


class RecordingPolicyStack(PolicyStack):
    def __init__(self) -> None:
        self.calls: List[str] = []

    def push(self, node_id: str, metadata: Dict[str, Any]) -> None:  # type: ignore[override]
        self.calls.append(f"push:{node_id}")

    def resolve(self, node_id: str) -> None:  # type: ignore[override]
        self.calls.append(f"resolve:{node_id}")

    def pop(self, node_id: str) -> None:  # type: ignore[override]
        self.calls.append(f"pop:{node_id}")


@pytest.fixture()
def runner(trace_collector):
    writer, events = trace_collector
    manager = BudgetManager()
    policy = RecordingPolicyStack()
    emitter = TraceEventEmitter(writer)
    return runner_factory(manager, policy, emitter), manager, policy, events


def runner_factory(manager: BudgetManager, policy: PolicyStack, emitter: TraceEventEmitter) -> FlowRunner:
    return FlowRunner(budget_manager=manager, trace_emitter=emitter, policy_stack=policy)


def test_soft_warn_allows_completion(runner) -> None:
    flow_runner, manager, policy, events = runner
    run_id = "run-soft"
    manager.enter_scope(
        scope_type="run",
        scope_id=run_id,
        spec=BudgetSpec(scope_type="run", limit_ms=100, mode=BudgetMode.SOFT, breach_action=BreachAction.WARN),
    )
    context = RunContext(run_id=run_id)
    flow = [
        FlowNode("node-1", FakeAdapter(estimate_ms=40, actual_ms=40), {}),
        FlowNode("node-2", FakeAdapter(estimate_ms=70, actual_ms=65), {}),
    ]

    result = flow_runner.run(flow, context)

    assert result.completed_nodes == ["node-1", "node-2"]
    assert result.breaches == []
    assert len(result.warnings) == 1
    assert "run-soft" in result.warnings[0]
    assert any(event == "budget_charge" for event, _ in events)


def test_hard_breach_stops_execution(runner) -> None:
    flow_runner, manager, policy, events = runner
    run_id = "run-hard"
    manager.enter_scope(
        "run",
        run_id,
        BudgetSpec(scope_type="run", limit_ms=60, mode=BudgetMode.HARD, breach_action=BreachAction.STOP),
    )
    context = RunContext(run_id=run_id)
    flow = [
        FlowNode("node-1", FakeAdapter(estimate_ms=30, actual_ms=30), {}),
        FlowNode("node-2", FakeAdapter(estimate_ms=40, actual_ms=50), {}),
    ]

    result = flow_runner.run(flow, context)

    assert result.completed_nodes == ["node-1"]
    assert result.breaches != []
    assert result.breaches[0]["scope_id"] == run_id
    assert any(event == "budget_breach" for event, _ in events)


def test_policy_stack_invocation_order(runner) -> None:
    flow_runner, manager, policy, events = runner
    run_id = "run-policy"
    manager.enter_scope(
        "run",
        run_id,
        BudgetSpec(scope_type="run", limit_ms=500, mode=BudgetMode.HARD, breach_action=BreachAction.WARN),
    )
    context = RunContext(run_id=run_id)
    flow = [FlowNode("node-1", FakeAdapter(estimate_ms=10, actual_ms=10), {})]

    flow_runner.run(flow, context)

    assert policy.calls == ["push:node-1", "resolve:node-1", "pop:node-1"]
    node_events = [payload for event, payload in events if event.startswith("policy_")]
    assert node_events[0]["scope"] == "node-1"
    assert node_events[0]["event"] == "policy_push"
