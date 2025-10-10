"""FlowRunner integration tests covering budget guards and loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..budget.manager import BudgetManager
from ..budget.models import CostSnapshot
from ..runner.flow_runner import FlowRunner, RunResult
from ..trace.emitter import TraceEventEmitter


class FakeAdapter:
    def __init__(self, *, estimate: CostSnapshot, actual_costs: Iterable[CostSnapshot]) -> None:
        self._estimate = estimate
        self._costs = list(actual_costs)
        self.calls: list[Mapping[str, object]] = []

    def estimate_cost(self, inputs: Mapping[str, object]) -> CostSnapshot:
        return self._estimate

    def execute(self, inputs: Mapping[str, object]) -> tuple[Mapping[str, object], CostSnapshot]:
        self.calls.append(inputs)
        if not self._costs:
            raise RuntimeError("no more cost fixtures")
        cost = self._costs.pop(0)
        return {"echo": inputs.get("value")}, cost


@dataclass
class FakePolicyAdapter:
    resolutions: list[tuple[str, Sequence[str]]]

    def resolve(self, node_id: str, tool_chain: Sequence[str]) -> Sequence[str]:
        self.resolutions.append((node_id, tuple(tool_chain)))
        return tool_chain

    def push(self, policy: Mapping[str, object] | None, scope: str) -> None:  # pragma: no cover - unused in tests
        self.resolutions.append((f"push:{scope}", tuple()))

    def pop(self, expected_scope: str | None = None) -> None:  # pragma: no cover - unused in tests
        self.resolutions.append((f"pop:{expected_scope}", tuple()))


def _runner_with_defaults(*, adapter: FakeAdapter, policy: FakePolicyAdapter, trace: TraceEventEmitter) -> FlowRunner:
    manager = BudgetManager(emitter=trace)
    return FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_adapter=policy,
        trace=trace,
        run_id_factory=lambda: "run-001",
    )


class TestFlowRunnerBudgets:
    def test_preflight_stop_halts_run(self) -> None:
        trace = TraceEventEmitter()
        adapter = FakeAdapter(
            estimate=CostSnapshot.from_mapping({"seconds": 1}),
            actual_costs=[
                CostSnapshot.from_mapping({"seconds": 1.2}),
                CostSnapshot.from_mapping({"seconds": 1.2}),
            ],
        )
        policy = FakePolicyAdapter(resolutions=[])
        runner = _runner_with_defaults(adapter=adapter, policy=policy, trace=trace)

        flow = {
            "id": "demo",
            "budget": {"limit": {"seconds": 2}, "mode": "hard", "breach_action": "stop"},
            "nodes": [
                {"id": "first", "type": "unit", "tool": "echo", "inputs": {"value": "a"}},
                {"id": "second", "type": "unit", "tool": "echo", "inputs": {"value": "b"}},
            ],
        }

        result = runner.run(flow, variables={})
        assert isinstance(result, RunResult)
        assert result.status == "halted"
        assert result.stop_reason == {"scope": "run", "scope_id": "demo"}
        assert result.outputs["first"]["echo"] == "a"
        assert "second" not in result.outputs
        assert len(adapter.calls) == 1
        preflight_events = [event for event in trace.events if event.event == "budget_preflight"]
        assert any(event.payload["stop"] for event in preflight_events)
        assert policy.resolutions == [("first", ("echo",)), ("second", ("echo",))]

    def test_loop_budget_halts_iteration(self) -> None:
        trace = TraceEventEmitter()
        adapter = FakeAdapter(
            estimate=CostSnapshot.from_mapping({"seconds": 1}),
            actual_costs=[
                CostSnapshot.from_mapping({"seconds": 1.1}),
                CostSnapshot.from_mapping({"seconds": 1.1}),
                CostSnapshot.from_mapping({"seconds": 1.1}),
            ],
        )
        policy = FakePolicyAdapter(resolutions=[])
        runner = _runner_with_defaults(adapter=adapter, policy=policy, trace=trace)

        flow = {
            "id": "loop-flow",
            "nodes": [
                {
                    "id": "loop1",
                    "type": "loop",
                    "body": [
                        {
                            "id": "loop-unit",
                            "type": "unit",
                            "tool": "echo",
                            "inputs": {"value": "x"},
                        }
                    ],
                    "stop": {
                        "budget": {
                            "limit": {"seconds": 2},
                            "mode": "hard",
                            "breach_action": "stop",
                        }
                    },
                }
            ],
        }

        result = runner.run(flow, variables={})
        assert result.status == "halted"
        assert result.stop_reason == {"scope": "loop", "scope_id": "loop1"}
        assert result.outputs["loop-unit"] == [
            {"echo": "x"},
            {"echo": "x"},
        ]
        loop_events = [event for event in trace.events if event.event == "loop_halt"]
        assert len(loop_events) == 1
        assert loop_events[0].payload["reason"] == "budget_stop"
        assert len(adapter.calls) == 2
