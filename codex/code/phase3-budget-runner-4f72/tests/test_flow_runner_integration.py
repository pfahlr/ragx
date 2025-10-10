from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from dsl.budget import BudgetManager, BudgetMode, BudgetSpec, CostSnapshot
from dsl.policy import PolicyStack, PolicyViolation
from dsl.runner import FlowNode, FlowRunner, RunContext, RunResult
from dsl.trace import InMemoryTraceWriter, TraceEventEmitter


class FakeAdapter:
    def __init__(self, name: str, estimate_ms: float, execute_ms: float | None = None) -> None:
        self.name = name
        self.estimate_ms = estimate_ms
        self.execute_ms = execute_ms if execute_ms is not None else estimate_ms
        self.executions: List[float] = []

    def estimate(self, context: RunContext) -> CostSnapshot:
        return CostSnapshot.from_raw({"time_ms": self.estimate_ms})

    def execute(self, context: RunContext) -> float:
        self.executions.append(self.execute_ms)
        return self.execute_ms


@dataclass
class FakeFlow:
    nodes: list[FlowNode]


def make_soft_flow():
    writer = InMemoryTraceWriter()
    emitter = TraceEventEmitter(writer)
    manager = BudgetManager(emitter)
    policy_stack = PolicyStack(emitter, allowlist={"allowed"})
    runner = FlowRunner(budget_manager=manager, policy_stack=policy_stack, emitter=emitter)

    adapter1 = FakeAdapter("allowed", 800.0)
    adapter2 = FakeAdapter("allowed", 400.0)
    spec = BudgetSpec(scope_id="run", limits={"time_ms": 1000.0}, mode=BudgetMode.SOFT, breach_action="warn")
    flow = FakeFlow(
        nodes=[
            FlowNode(node_id="n1", adapter=adapter1, budget_spec=spec, scope_type="run"),
            FlowNode(node_id="n2", adapter=adapter2, budget_spec=spec, scope_type="run"),
        ]
    )
    return flow, runner, writer, adapter1, adapter2


def test_runner_continues_on_soft_budget_warn():
    flow, runner, writer, adapter1, adapter2 = make_soft_flow()
    context = RunContext(run_id="run-1")

    result = runner.run(flow, context)

    assert isinstance(result, RunResult)
    assert result.stop_reason is None
    assert [node.node_id for node in result.completed_nodes] == ["n1", "n2"]
    assert adapter1.executions and adapter2.executions

    events = writer.snapshot()
    policy_resolved = [event for event in events if event["event"] == "policy_resolved"]
    assert len(policy_resolved) == 2
    breaches = [event for event in events if event["event"] == "budget_breach"]
    assert breaches
    assert breaches[-1]["severity"] == "soft"


def test_runner_stops_on_hard_budget_before_execution():
    writer = InMemoryTraceWriter()
    emitter = TraceEventEmitter(writer)
    manager = BudgetManager(emitter)
    policy_stack = PolicyStack(emitter, allowlist={"allowed"})
    runner = FlowRunner(budget_manager=manager, policy_stack=policy_stack, emitter=emitter)

    adapter1 = FakeAdapter("allowed", 60.0)
    adapter2 = FakeAdapter("allowed", 60.0)
    spec = BudgetSpec(scope_id="run", limits={"time_ms": 100.0}, mode=BudgetMode.HARD, breach_action="stop")
    flow = FakeFlow(
        nodes=[
            FlowNode(node_id="n1", adapter=adapter1, budget_spec=spec, scope_type="run"),
            FlowNode(node_id="n2", adapter=adapter2, budget_spec=spec, scope_type="run"),
        ]
    )
    context = RunContext(run_id="run-2")

    result = runner.run(flow, context)

    assert result.stop_reason == "budget_hard_stop"
    assert [node.node_id for node in result.completed_nodes] == ["n1"]
    assert adapter2.executions == []

    events = writer.snapshot()
    breaches = [event for event in events if event["event"] == "budget_breach"]
    assert breaches[-1]["severity"] == "hard"
    # Policy still resolves for the executed node
    policy_resolved = [event for event in events if event["event"] == "policy_resolved"]
    assert len(policy_resolved) == 1


def test_runner_raises_policy_violation_before_budget_check():
    writer = InMemoryTraceWriter()
    emitter = TraceEventEmitter(writer)
    manager = BudgetManager(emitter)
    policy_stack = PolicyStack(emitter, allowlist={"allowed"})
    runner = FlowRunner(budget_manager=manager, policy_stack=policy_stack, emitter=emitter)

    violating_adapter = FakeAdapter("blocked", 10.0)
    spec = BudgetSpec(scope_id="run", limits={"time_ms": 1000.0}, mode=BudgetMode.SOFT, breach_action="warn")
    flow = FakeFlow([FlowNode(node_id="n1", adapter=violating_adapter, budget_spec=spec, scope_type="run")])

    with pytest.raises(PolicyViolation):
        runner.run(flow, RunContext(run_id="run-3"))

    events = writer.snapshot()
    assert any(event["event"] == "policy_violation" for event in events)
    assert all(event["event"] != "budget_charge" for event in events)
