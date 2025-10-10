import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from dataclasses import dataclass
from typing import Dict, List

import pytest

from codex.code.task_07b_budget_guards_and_runner_integration.budget_manager import BudgetManager
from codex.code.task_07b_budget_guards_and_runner_integration.budget_models import (
    BreachAction,
    BudgetDecision,
    BudgetMode,
    BudgetSpec,
    CostAmount,
)
from codex.code.task_07b_budget_guards_and_runner_integration.flow_runner import FlowNode, FlowRunner, RunResult
from codex.code.task_07b_budget_guards_and_runner_integration.policy_stack import PolicyDecision, PolicyStack
from codex.code.task_07b_budget_guards_and_runner_integration.trace_emitter import TraceEventEmitter


@dataclass
class DummyAdapter:
    identifier: str
    estimates: List[str]
    actuals: List[str]
    outputs: List[str]

    def estimate(self, node: FlowNode) -> CostAmount:
        return CostAmount.of(self.estimates.pop(0))

    def execute(self, node: FlowNode):
        amount = CostAmount.of(self.actuals.pop(0))
        output = self.outputs.pop(0)
        return output, amount

    def identify(self, node: FlowNode) -> str:
        return self.identifier

    def describe(self, node: FlowNode) -> Dict[str, str]:
        return {"adapter": self.identifier, "node": node.node_id}


class AllowAllPolicy(PolicyStack):
    def __init__(self, emitter: TraceEventEmitter):
        super().__init__(rules=[], emitter=emitter)

    def evaluate(self, node: FlowNode) -> PolicyDecision:
        decision = PolicyDecision(allowed=True, reason="allow")
        self.record(node, decision, self.emitter)
        return decision


class DenySecondPolicy(PolicyStack):
    def __init__(self, emitter: TraceEventEmitter, deny_node: str):
        super().__init__(rules=[], emitter=emitter)
        self.deny_node = deny_node

    def evaluate(self, node: FlowNode) -> PolicyDecision:
        allowed = node.node_id != self.deny_node
        reason = "deny" if not allowed else "allow"
        decision = PolicyDecision(allowed=allowed, reason=reason)
        self.record(node, decision, self.emitter)
        return decision


def build_runner(policy_stack_cls, **policy_kwargs):
    emitter = TraceEventEmitter()
    manager = BudgetManager(emitter=emitter)
    policy = policy_stack_cls(emitter=emitter, **policy_kwargs)
    return FlowRunner(budget_manager=manager, policy_stack=policy, emitter=emitter), emitter, manager


def test_flow_runner_stops_on_budget_breach():
    runner, emitter, manager = build_runner(AllowAllPolicy)

    run_spec = BudgetSpec(
        scope_id="run",
        limit=CostAmount.of("5"),
        mode=BudgetMode.HARD,
        breach_action=BreachAction.STOP,
    )
    manager.register_scope("run", run_spec)

    adapter = DummyAdapter(
        identifier="dummy",
        estimates=["4", "4"],
        actuals=["4", "4"],
        outputs=["ok-1", "ok-2"],
    )

    nodes = [FlowNode(node_id="n1", adapter=adapter), FlowNode(node_id="n2", adapter=adapter)]

    result = runner.run(nodes, run_scope="run", budget_specs={"run": run_spec})

    assert list(result.executed_nodes) == ["n1"]
    assert result.stop_reason == "budget_stop"

    events = emitter.get_events()
    budget_events = [e for e in events if e.event.startswith("budget_")]
    assert budget_events[0].event == "budget_preview"
    assert budget_events[-1].event == "budget_stop"
    assert budget_events[-1].payload["scope_id"] == "run"


def test_flow_runner_warns_on_soft_budget():
    runner, emitter, manager = build_runner(AllowAllPolicy)

    run_spec = BudgetSpec(
        scope_id="run",
        limit=CostAmount.of("5"),
        mode=BudgetMode.SOFT,
        breach_action=BreachAction.WARN,
    )
    manager.register_scope("run", run_spec)

    adapter = DummyAdapter(
        identifier="dummy",
        estimates=["3", "3"],
        actuals=["3", "3"],
        outputs=["ok-1", "ok-2"],
    )

    nodes = [FlowNode(node_id="n1", adapter=adapter), FlowNode(node_id="n2", adapter=adapter)]

    result = runner.run(nodes, run_scope="run", budget_specs={"run": run_spec})

    assert list(result.executed_nodes) == ["n1", "n2"]
    assert result.stop_reason is None
    assert any(warning.scope_id == "run" for warning in result.warnings)

    warn_events = [e for e in emitter.get_events() if e.event == "budget_warning"]
    assert len(warn_events) == 2


def test_flow_runner_halts_on_policy_violation_before_commit():
    runner, emitter, manager = build_runner(DenySecondPolicy, deny_node="n2")

    run_spec = BudgetSpec(scope_id="run", limit=CostAmount.of("10"))
    manager.register_scope("run", run_spec)

    adapter = DummyAdapter(
        identifier="dummy",
        estimates=["2", "2"],
        actuals=["2", "2"],
        outputs=["ok-1", "ok-2"],
    )

    nodes = [FlowNode(node_id="n1", adapter=adapter), FlowNode(node_id="n2", adapter=adapter)]

    result = runner.run(nodes, run_scope="run", budget_specs={"run": run_spec})

    assert list(result.executed_nodes) == ["n1"]
    assert result.stop_reason == "policy_violation"

    policy_events = [e for e in emitter.get_events() if e.event == "policy_decision"]
    assert policy_events[0].payload["decision"] == "allow"
    assert policy_events[1].payload["decision"] == "deny"

    budget_commits = [e for e in emitter.get_events() if e.event == "budget_commit"]
    assert len(budget_commits) == 1
    assert budget_commits[0].payload["node_id"] == "n1"


def test_trace_events_are_ordered():
    runner, emitter, manager = build_runner(AllowAllPolicy)

    run_spec = BudgetSpec(scope_id="run", limit=CostAmount.of("3"))
    manager.register_scope("run", run_spec)

    adapter = DummyAdapter(
        identifier="dummy",
        estimates=["1", "1", "1"],
        actuals=["1", "1", "1"],
        outputs=["o1", "o2", "o3"],
    )
    nodes = [FlowNode(node_id=f"n{i}", adapter=adapter) for i in range(1, 4)]

    result = runner.run(nodes, run_scope="run", budget_specs={"run": run_spec})

    events = emitter.get_events()
    node_start_indices = [i for i, evt in enumerate(events) if evt.event == "node_start"]
    policy_indices = [i for i, evt in enumerate(events) if evt.event == "policy_decision"]
    commit_indices = [i for i, evt in enumerate(events) if evt.event == "budget_commit"]

    assert all(start < policy_indices[idx] < commit_indices[idx] for idx, start in enumerate(node_start_indices))

    assert list(result.executed_nodes) == ["n1", "n2", "n3"]
    assert result.stop_reason is None
