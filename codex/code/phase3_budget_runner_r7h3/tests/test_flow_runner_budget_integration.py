from dataclasses import dataclass
from typing import Dict, List
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:  # pragma: no cover - import shim for tests
    sys.path.insert(0, str(ROOT))

from codex.code.phase3_budget_runner_r7h3.dsl import budget, costs, runner, trace


class FakeAdapter:
    def __init__(self, estimates: List[Dict[str, float]], results: List[Dict[str, str]]):
        self._estimates = estimates
        self._results = results
        self.executed = []

    def estimate(self, context: Dict[str, str]) -> Dict[str, float]:
        return self._estimates[len(self.executed)]

    def execute(self, context: Dict[str, str]) -> Dict[str, str]:
        result = self._results[len(self.executed)]
        self.executed.append({"context": context, "result": result})
        return result


class RecordingTraceWriter(trace.TraceWriter):
    def __init__(self):
        self.events: List[Dict[str, object]] = []

    def emit(self, event: str, payload: Dict[str, object]) -> None:
        self.events.append({"event": event, **payload})


@dataclass
class DummyPolicyStack:
    resolved: List[str]

    def resolve(self, node_id: str) -> None:
        self.resolved.append(node_id)

    def validate(self, node_id: str) -> None:
        return None


@pytest.fixture
def run_manager():
    specs = {
        "run": budget.BudgetSpec(scope="run", limits={"tokens": 15}, breach_action="warn"),
        "node:loop": budget.BudgetSpec(scope="node", limits={"tokens": 6}, breach_action="stop"),
        "loop:loop": budget.BudgetSpec(scope="loop", limits={"tokens": 6}, breach_action="stop"),
    }
    return budget.BudgetManager(specs)


def test_flow_runner_stops_loop_on_budget_stop(run_manager):
    adapter = FakeAdapter(
        estimates=[{"tokens": 2}, {"tokens": 2}, {"tokens": 3}],
        results=[{"value": "ok1"}, {"value": "ok2"}, {"value": "ok3"}],
    )
    writer = RecordingTraceWriter()
    emitter = trace.TraceEventEmitter(writer, clock=lambda: 123.0)
    policy = DummyPolicyStack([])

    flow = runner.FlowDefinition(
        flow_id="flow",
        nodes=[
            runner.NodeDefinition(
                node_id="loop",
                adapter_id="loop_adapter",
                loop=runner.LoopConfig(max_iterations=5, scope_keys=["loop:loop"]),
                scope_keys=["run", "node:loop"],
            )
        ],
    )

    flow_runner = runner.FlowRunner(
        adapters={"loop_adapter": adapter},
        budget_manager=run_manager,
        trace_emitter=emitter,
        policy_stack=policy,
        flow_scope_keys=["run"],
    )

    result = flow_runner.run(flow, run_id="run-1")

    assert result.status == runner.FlowRunStatus.STOPPED
    assert result.stop_reason == "budget_stop"
    assert len(adapter.executed) == 2
    assert policy.resolved == ["loop", "loop"]

    breach_events = [event for event in writer.events if event["event"] == "budget_breach"]
    assert breach_events, "expected at least one breach event"
    last_breach = breach_events[-1]
    assert last_breach["action"] == "stop"
    assert last_breach["scope"] == "node:loop"


def test_flow_runner_raises_on_hard_breach():
    manager = budget.BudgetManager(
        {
            "run": budget.BudgetSpec(scope="run", limits={"tokens": 2}, breach_action="error"),
        }
    )
    adapter = FakeAdapter(
        estimates=[{"tokens": 5}],
        results=[{"value": "too much"}],
    )
    writer = RecordingTraceWriter()
    emitter = trace.TraceEventEmitter(writer, clock=lambda: 999.0)

    flow = runner.FlowDefinition(
        flow_id="flow",
        nodes=[
            runner.NodeDefinition(
                node_id="n1",
                adapter_id="a1",
                loop=None,
                scope_keys=["run"],
            )
        ],
    )
    flow_runner = runner.FlowRunner(
        adapters={"a1": adapter},
        budget_manager=manager,
        trace_emitter=emitter,
        flow_scope_keys=["run"],
    )

    with pytest.raises(budget.BudgetBreachError):
        flow_runner.run(flow, run_id="run-hard")

    breach_events = [event for event in writer.events if event["event"] == "budget_breach"]
    assert breach_events[0]["action"] == "error"


def test_flow_runner_emits_warn_and_continues():
    manager = budget.BudgetManager(
        {
            "run": budget.BudgetSpec(scope="run", limits={"tokens": 5}, breach_action="warn"),
            "node:n1": budget.BudgetSpec(scope="node", limits={"tokens": 10}, breach_action="warn"),
            "node:n2": budget.BudgetSpec(scope="node", limits={"tokens": 10}, breach_action="warn"),
        }
    )
    adapter1 = FakeAdapter(
        estimates=[{"tokens": 3}],
        results=[{"value": "first"}],
    )
    adapter2 = FakeAdapter(
        estimates=[{"tokens": 4}],
        results=[{"value": "second"}],
    )
    writer = RecordingTraceWriter()
    emitter = trace.TraceEventEmitter(writer, clock=lambda: 42.0)

    flow = runner.FlowDefinition(
        flow_id="flow",
        nodes=[
            runner.NodeDefinition(
                node_id="n1",
                adapter_id="a1",
                loop=None,
                scope_keys=["run", "node:n1"],
            ),
            runner.NodeDefinition(
                node_id="n2",
                adapter_id="a2",
                loop=None,
                scope_keys=["run", "node:n2"],
            ),
        ],
    )
    flow_runner = runner.FlowRunner(
        adapters={"a1": adapter1, "a2": adapter2},
        budget_manager=manager,
        trace_emitter=emitter,
        flow_scope_keys=["run"],
    )

    result = flow_runner.run(flow, run_id="run-soft")

    assert result.status == runner.FlowRunStatus.COMPLETED
    warn_events = [event for event in writer.events if event["event"] == "budget_breach" and event["action"] == "warn"]
    assert warn_events, "expected warn breach events"
    assert warn_events[-1]["scope"] == "run"
    assert result.stop_reason is None
