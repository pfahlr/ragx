from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from dsl.budget import (
    BreachAction,
    BudgetDecisionStatus,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    Cost,
    ScopeKey,
)
from dsl.policy import PolicyStack
from dsl.runner import FlowRunner, FlowSpec, LoopSpec, NodeSpec, RunResult
from dsl.trace import TraceEventEmitter


@dataclass
class FakeAdapter:
    cost_ms: float
    result: str

    def estimate(self, context: dict) -> Cost:
        return Cost(milliseconds=self.cost_ms)

    def execute(self, context: dict) -> str:
        return self.result


def build_runner(run_limit: float, loop_limit: float, node_limit: float, hard_loop: bool = False) -> FlowRunner:
    specs = {
        ScopeKey("run", "run"): BudgetSpec(
            scope_type="run",
            scope_id="run",
            limit_ms=run_limit,
            mode=BudgetMode.SOFT,
            breach_action=BreachAction.WARN,
        ),
        ScopeKey("loop", "loop1"): BudgetSpec(
            scope_type="loop",
            scope_id="loop1",
            limit_ms=loop_limit,
            mode=BudgetMode.HARD if hard_loop else BudgetMode.SOFT,
            breach_action=BreachAction.STOP if hard_loop else BreachAction.WARN,
        ),
        ScopeKey("node", "node1"): BudgetSpec(
            scope_type="node",
            scope_id="node1",
            limit_ms=node_limit,
            mode=BudgetMode.SOFT,
            breach_action=BreachAction.WARN,
        ),
    }
    manager = BudgetManager(specs)
    trace = TraceEventEmitter()
    policy = PolicyStack(trace)
    return FlowRunner(budget_manager=manager, trace_emitter=trace, policy_stack=policy)


def collect_event_names(trace: TraceEventEmitter) -> List[str]:
    return [event.event for event in trace.events]


def test_soft_loop_budget_emits_warning_and_continues():
    runner = build_runner(run_limit=300, loop_limit=120, node_limit=200, hard_loop=False)
    adapter = FakeAdapter(cost_ms=70, result="ok")
    flow = FlowSpec(
        flow_id="flow",
        nodes=[NodeSpec(node_id="node1", adapter=adapter, scope_key=ScopeKey("node", "node1"))],
        loops=[LoopSpec(loop_id="loop1", iterations=2, nodes=["node1"], scope_key=ScopeKey("loop", "loop1"))],
    )

    result = runner.run(flow)

    assert result.completed is True
    assert result.stop_reason is None
    assert len(result.loop_summaries) == 1
    loop_summary = result.loop_summaries[0]
    assert loop_summary.loop_id == "loop1"
    assert loop_summary.iterations_completed == 2
    assert loop_summary.stop_reason == "soft_budget_warn"

    events = collect_event_names(runner.trace_emitter)
    assert events.count("budget_breach") == 1
    assert "policy_push" in events
    assert events.index("policy_push") < events.index("policy_resolved") < events.index("policy_pop")
    assert events.index("budget_charge") > events.index("policy_resolved")

    breach_event = next(event for event in runner.trace_emitter.events if event.event == "budget_breach")
    assert breach_event.payload["breach_kind"] == "soft"
    assert breach_event.payload["action"] == "warn"


def test_hard_loop_budget_stops_runner():
    runner = build_runner(run_limit=300, loop_limit=100, node_limit=200, hard_loop=True)
    adapter = FakeAdapter(cost_ms=70, result="ok")
    flow = FlowSpec(
        flow_id="flow",
        nodes=[NodeSpec(node_id="node1", adapter=adapter, scope_key=ScopeKey("node", "node1"))],
        loops=[LoopSpec(loop_id="loop1", iterations=3, nodes=["node1"], scope_key=ScopeKey("loop", "loop1"))],
    )

    result = runner.run(flow)

    assert result.completed is False
    assert result.stop_reason == "budget_breach(loop:loop1)"
    loop_summary = result.loop_summaries[0]
    assert loop_summary.iterations_completed == 2
    assert loop_summary.stop_reason == "budget_stop"

    events = collect_event_names(runner.trace_emitter)
    assert events.count("budget_breach") == 1
    breach_event = next(event for event in runner.trace_emitter.events if event.event == "budget_breach")
    assert breach_event.payload["breach_kind"] == "hard"
    assert breach_event.payload["action"] == "stop"

    # Node budget should register a stop decision on the second iteration attempt.
    stop_decisions = [
        event
        for event in runner.trace_emitter.events
        if event.event == "budget_charge" and event.payload["decision_status"] == BudgetDecisionStatus.STOP.value
    ]
    assert len(stop_decisions) == 1
