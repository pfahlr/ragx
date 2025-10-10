from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from codex.code.work.budget import BudgetManager, BudgetMode, BudgetSpec  # noqa: E402
from codex.code.work.runner import (  # noqa: E402
    FlowDefinition,
    FlowRunner,
    NodeDefinition,
    PolicyStack,
    PolicyViolationError,
)
from codex.code.work.trace import TraceEventEmitter  # noqa: E402


class DummyAdapter:
    def __init__(self, estimates: dict[str, dict[str, float]], actuals: dict[str, dict[str, float]]) -> None:
        self._estimates = estimates
        self._actuals = actuals
        self.calls: list[str] = []

    def estimate(self, node: NodeDefinition) -> dict[str, float]:
        return dict(self._estimates[node.id])

    def execute(self, node: NodeDefinition) -> tuple[dict[str, str], dict[str, float]]:
        self.calls.append(node.id)
        return {"node": node.id}, dict(self._actuals[node.id])


@pytest.fixture()
def runner_components() -> tuple[BudgetManager, PolicyStack, TraceEventEmitter]:
    emitter = TraceEventEmitter()
    spec = BudgetSpec(
        scope="run",
        limits={"tokens": 10},
        mode=BudgetMode.HARD,
        breach_action="stop",
    )
    manager = BudgetManager({"run": spec}, trace_emitter=emitter)
    policy_stack = PolicyStack(allow_tools={"alpha", "beta"}, trace_emitter=emitter)
    return manager, policy_stack, emitter


def test_flow_runner_stops_on_budget_breach(runner_components: tuple[BudgetManager, PolicyStack, TraceEventEmitter]) -> None:
    manager, policy_stack, emitter = runner_components
    adapter = DummyAdapter(
        estimates={
            "n1": {"tokens": 4},
            "n2": {"tokens": 7},
        },
        actuals={
            "n1": {"tokens": 4},
            "n2": {"tokens": 7},
        },
    )
    flow = FlowDefinition(run_scope="run", nodes=[NodeDefinition(id="n1", tool="alpha"), NodeDefinition(id="n2", tool="alpha")])
    runner = FlowRunner(adapter=adapter, budget_manager=manager, policy_stack=policy_stack, trace_emitter=emitter)

    result = runner.run(flow)

    assert [execution.node_id for execution in result.executions] == ["n1"]
    assert result.stop_reason == "budget_preflight"
    assert any(event.event == "budget_breach" for event in emitter.events)


def test_flow_runner_emits_policy_events(runner_components: tuple[BudgetManager, PolicyStack, TraceEventEmitter]) -> None:
    manager, policy_stack, emitter = runner_components
    adapter = DummyAdapter(
        estimates={"n1": {"tokens": 2}},
        actuals={"n1": {"tokens": 2}},
    )
    flow = FlowDefinition(run_scope="run", nodes=[NodeDefinition(id="n1", tool="alpha")])
    runner = FlowRunner(adapter=adapter, budget_manager=manager, policy_stack=policy_stack, trace_emitter=emitter)

    result = runner.run(flow)

    assert result.stop_reason is None
    policy_events = [event for event in emitter.events if event.event.startswith("policy_")]
    assert {event.event for event in policy_events} >= {"policy_resolved"}

    charge_events = [event for event in emitter.events if event.event == "budget_charge"]
    assert charge_events, "expected budget_charge trace events"


def test_flow_runner_raises_on_policy_violation(runner_components: tuple[BudgetManager, PolicyStack, TraceEventEmitter]) -> None:
    manager, policy_stack, emitter = runner_components
    adapter = DummyAdapter(
        estimates={"n1": {"tokens": 2}},
        actuals={"n1": {"tokens": 2}},
    )
    violating_flow = FlowDefinition(run_scope="run", nodes=[NodeDefinition(id="n1", tool="gamma")])
    runner = FlowRunner(adapter=adapter, budget_manager=manager, policy_stack=policy_stack, trace_emitter=emitter)

    with pytest.raises(PolicyViolationError):
        runner.run(violating_flow)

    violation_events = [event for event in emitter.events if event.event == "policy_violation"]
    assert violation_events, "policy violation should be traced"
