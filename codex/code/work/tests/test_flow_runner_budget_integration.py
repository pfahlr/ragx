import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from codex.code.work.pkgs.dsl.runner import FlowRunner, RunResult
from codex.code.work.pkgs.dsl.trace import TraceEventEmitter


class DummyTraceWriter:
    def __init__(self):
        self.events = []

    def write(self, event_name: str, payload: dict) -> None:
        self.events.append((event_name, payload))


class FixedCostAdapter:
    def __init__(self, cost, output):
        self.cost = cost
        self.output = output
        self.estimate_calls = 0
        self.execute_calls = 0

    def estimate(self, node_spec: dict) -> dict:
        self.estimate_calls += 1
        return dict(self.cost)

    def execute(self, node_spec: dict) -> tuple:
        self.execute_calls += 1
        return dict(self.output), dict(self.cost)


@pytest.fixture
def trace_emitter():
    writer = DummyTraceWriter()
    return TraceEventEmitter(trace_writer=writer), writer


def build_runner(adapters, trace_emitter):
    emitter, writer = trace_emitter
    return FlowRunner(
        adapters=adapters,
        trace_emitter=emitter,
        id_factory=lambda: "run-fixed",
    ), writer


def test_runner_halts_on_run_budget_breach(trace_emitter):
    adapters = {"expensive": FixedCostAdapter({"calls": 3}, {"result": "x"})}
    runner, writer = build_runner(adapters, trace_emitter)

    spec = {
        "flow_id": "flow",
        "run_budget": {"metric": "calls", "limit": 2, "breach_action": "error", "mode": "hard"},
        "policies": {"allow": ["expensive"]},
        "nodes": [
            {
                "id": "n1",
                "kind": "unit",
                "spec": {"tool_ref": "expensive"},
            }
        ],
    }

    result: RunResult = runner.run(spec, vars={})

    assert result.status == "halted"
    assert result.stop_reason.startswith("budget_breach")
    assert writer.events[-1][0] == "run_end"
    breach_events = [evt for evt in writer.events if evt[0] == "budget_breach"]
    assert breach_events, "budget breach event missing"
    assert breach_events[-1][1]["scope"] == "run"


def test_runner_warns_on_node_soft_budget(trace_emitter):
    adapters = {"echo": FixedCostAdapter({"calls": 2}, {"payload": "value"})}
    runner, writer = build_runner(adapters, trace_emitter)

    spec = {
        "flow_id": "flow",
        "run_budget": {"metric": "calls", "limit": 10, "breach_action": "error", "mode": "hard"},
        "policies": {"allow": ["echo"]},
        "nodes": [
            {
                "id": "n1",
                "kind": "unit",
                "spec": {"tool_ref": "echo"},
                "budget": {"metric": "calls", "limit": 1, "breach_action": "warn", "mode": "soft"},
            }
        ],
    }

    result = runner.run(spec, vars={})

    assert result.status == "ok"
    assert result.outputs["n1"] == {"payload": "value"}
    warn_events = [evt for evt in writer.events if evt[0] == "budget_breach"]
    assert warn_events[0][1]["action"] == "warn"
    assert "n1" in result.warnings[0]


def test_runner_loop_budget_stop(trace_emitter):
    adapters = {"cheap": FixedCostAdapter({"calls": 1}, {"ok": True})}
    runner, writer = build_runner(adapters, trace_emitter)

    spec = {
        "flow_id": "flow",
        "run_budget": {"metric": "calls", "limit": 10, "breach_action": "error", "mode": "hard"},
        "policies": {"allow": ["cheap"]},
        "nodes": [
            {
                "id": "loop",
                "kind": "loop",
                "stop": {"budget": {"metric": "calls", "limit": 2, "breach_action": "stop", "mode": "hard"}},
                "body": [
                    {
                        "id": "task",
                        "kind": "unit",
                        "spec": {"tool_ref": "cheap"},
                    }
                ],
            }
        ],
    }

    result = runner.run(spec, vars={})

    assert result.status == "ok"
    assert len(result.outputs["loop"]) == 2
    stop_events = [evt for evt in writer.events if evt[0] == "budget_breach" and evt[1]["action"] == "stop"]
    assert stop_events, "loop stop event missing"
    assert result.outputs["loop"][0]["task"] == {"ok": True}
