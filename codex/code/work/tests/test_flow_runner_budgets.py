"""Integration tests for FlowRunner budget enforcement and traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytest

from codex.code.work.runner.budgeting import CostSnapshot
from codex.code.work.runner.flow_runner import FlowRunner, RunResult
from codex.code.work.runner.trace import TraceEventEmitter
from pkgs.dsl.policy import PolicyStack


@dataclass
class FakeAdapter:
    name: str
    estimate_costs: Iterable[CostSnapshot]
    execution_costs: Iterable[CostSnapshot]
    outputs: Iterable[dict[str, object]]

    def __post_init__(self) -> None:
        self._estimate_iter = iter(self.estimate_costs)
        self._execute_iter = iter(self.execution_costs)
        self._output_iter = iter(self.outputs)
        self.calls = 0

    def estimate_cost(self, inputs: dict[str, object]) -> CostSnapshot:
        try:
            return next(self._estimate_iter)
        except StopIteration:  # pragma: no cover - safeguard
            return CostSnapshot.zero()

    def execute(self, inputs: dict[str, object]) -> tuple[dict[str, object], CostSnapshot]:
        self.calls += 1
        try:
            result = next(self._output_iter)
        except StopIteration:  # pragma: no cover - safeguard
            result = {}
        try:
            cost = next(self._execute_iter)
        except StopIteration:  # pragma: no cover - safeguard
            cost = CostSnapshot.zero()
        return result, cost


@pytest.fixture()
def emitter() -> TraceEventEmitter:
    return TraceEventEmitter()


@pytest.fixture()
def policy_stack(emitter: TraceEventEmitter) -> PolicyStack:
    tools = {
        "alpha": {"tags": ["unit"]},
        "beta": {"tags": ["unit"]},
    }

    def sink(event) -> None:
        emitter.emit_policy_event(
            event.event,
            scope=event.scope,
            payload={"data": dict(event.data)},
        )

    return PolicyStack(tools=tools, event_sink=sink)


def make_runner(emitter: TraceEventEmitter, policy_stack: PolicyStack, adapters: dict[str, FakeAdapter]) -> FlowRunner:
    return FlowRunner(
        adapters=adapters,
        policy_stack=policy_stack,
        trace_emitter=emitter,
    )


def test_run_halts_on_run_budget_breach(emitter: TraceEventEmitter, policy_stack: PolicyStack) -> None:
    adapters = {
        "alpha": FakeAdapter(
            name="alpha",
            estimate_costs=[CostSnapshot(milliseconds=60, tokens_in=0, tokens_out=0, calls=1)],
            execution_costs=[CostSnapshot(milliseconds=60, tokens_in=0, tokens_out=0, calls=1)],
            outputs=[{"alpha": "alpha"}],
        ),
        "beta": FakeAdapter(
            name="beta",
            estimate_costs=[CostSnapshot(milliseconds=60, tokens_in=0, tokens_out=0, calls=1)],
            execution_costs=[CostSnapshot(milliseconds=60, tokens_in=0, tokens_out=0, calls=1)],
            outputs=[{"beta": "beta"}],
        ),
    }
    runner = make_runner(emitter, policy_stack, adapters)

    spec = {
        "id": "flow-hard-stop",
        "run_budget": {
            "limit": {"milliseconds": 100},
            "mode": "hard",
            "breach_action": "stop",
        },
        "nodes": [
            {
                "id": "alpha-node",
                "kind": "unit",
                "tool": "alpha",
                "outputs": ["alpha"],
            },
            {
                "id": "beta-node",
                "kind": "unit",
                "tool": "beta",
                "outputs": ["beta"],
            },
        ],
    }

    result = runner.run(spec)

    assert isinstance(result, RunResult)
    assert result.status == "halted"
    assert result.outputs == {"alpha-node": {"alpha": "alpha"}}
    assert any(event.event == "budget_breach" for event in emitter.events)
    policy_events = [event for event in emitter.events if event.event == "policy_resolved"]
    assert policy_events, "policy resolution should be traced"


def test_soft_node_budget_warns_but_continues(emitter: TraceEventEmitter, policy_stack: PolicyStack) -> None:
    adapters = {
        "alpha": FakeAdapter(
            name="alpha",
            estimate_costs=[CostSnapshot(milliseconds=20, tokens_in=0, tokens_out=0, calls=1)],
            execution_costs=[CostSnapshot(milliseconds=40, tokens_in=0, tokens_out=0, calls=1)],
            outputs=[{"alpha": "alpha"}],
        ),
    }
    runner = make_runner(emitter, policy_stack, adapters)

    spec = {
        "id": "flow-soft-node",
        "run_budget": {
            "limit": {"milliseconds": 200},
            "mode": "hard",
            "breach_action": "stop",
        },
        "nodes": [
            {
                "id": "alpha-node",
                "kind": "unit",
                "tool": "alpha",
                "outputs": ["alpha"],
                "budget": {
                    "limit": {"milliseconds": 30},
                    "mode": "soft",
                    "breach_action": "warn",
                },
            }
        ],
    }

    result = runner.run(spec)

    assert result.status == "ok"
    assert result.outputs["alpha-node"] == {"alpha": "alpha"}
    warnings = [event for event in emitter.events if event.event == "budget_warning"]
    assert warnings, "soft budget should emit warning"
    assert warnings[-1].payload["scope_id"] == "alpha-node"


def test_loop_budget_stops_iteration(emitter: TraceEventEmitter, policy_stack: PolicyStack) -> None:
    adapters = {
        "alpha": FakeAdapter(
            name="alpha",
            estimate_costs=[
                CostSnapshot(milliseconds=20, tokens_in=0, tokens_out=0, calls=1),
                CostSnapshot(milliseconds=20, tokens_in=0, tokens_out=0, calls=1),
            ],
            execution_costs=[
                CostSnapshot(milliseconds=20, tokens_in=0, tokens_out=0, calls=1),
                CostSnapshot(milliseconds=20, tokens_in=0, tokens_out=0, calls=1),
            ],
            outputs=[{"loop": "loop"}, {"loop": "loop"}],
        ),
    }
    runner = make_runner(emitter, policy_stack, adapters)

    spec = {
        "id": "flow-loop",
        "run_budget": {
            "limit": {"milliseconds": 500},
            "mode": "hard",
            "breach_action": "stop",
        },
        "nodes": [
            {
                "id": "loop-1",
                "kind": "loop",
                "max_iterations": 5,
                "budget": {
                    "limit": {"milliseconds": 30},
                    "mode": "hard",
                    "breach_action": "stop",
                },
                "body": [
                    {
                        "id": "loop-1.unit",
                        "kind": "unit",
                        "tool": "alpha",
                        "outputs": ["loop"],
                    }
                ],
            }
        ],
    }

    result = runner.run(spec)

    assert result.status == "ok"
    assert result.outputs["loop-1.unit"] == {"loop": "loop"}
    stop_events = [event for event in emitter.events if event.event == "loop_stop"]
    assert stop_events, "loop stop event expected"
    assert stop_events[-1].payload["reason"] == "budget_exhausted"
    # Loop should stop after first iteration due to 30ms limit and 20ms per call estimate for second iteration
    assert adapters["alpha"].calls == 1
