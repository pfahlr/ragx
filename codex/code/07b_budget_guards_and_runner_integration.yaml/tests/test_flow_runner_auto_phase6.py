"""Phase 6 regression tests for FlowRunner budget/policy integration."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetManager
from pkgs.dsl.flow_runner import FlowRunner, ToolAdapter
from pkgs.dsl.policy import PolicyStack
from pkgs.dsl.trace import TraceEventEmitter


class RecordingAdapter(ToolAdapter):
    """Adapter that replays canned costs/results for deterministic tests."""

    def __init__(
        self,
        *,
        estimate_costs: Iterable[Mapping[str, float]],
        results: Iterable[Any] | None = None,
    ) -> None:
        self._estimate_costs = deque(dict(cost) for cost in estimate_costs)
        self._results = deque(results or [])
        self.executed: list[Mapping[str, object]] = []

    def estimate_cost(self, node: Mapping[str, object]) -> Mapping[str, float]:  # type: ignore[override]
        if len(self._estimate_costs) > 1:
            cost = self._estimate_costs.popleft()
        else:
            cost = self._estimate_costs[0]
        return dict(cost)

    def execute(self, node: Mapping[str, object]) -> Any:  # type: ignore[override]
        payload = dict(node)
        self.executed.append(payload)
        if self._results:
            return self._results.popleft()
        return {"ok": node.get("id")}


def _policy_stack() -> PolicyStack:
    tools: dict[str, Mapping[str, object]] = {
        "echo": {"tags": ["default"]},
    }
    return PolicyStack(tools=tools)


def _budget_manager(trace: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run-hard",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 400}),
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="loop-soft",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 250}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node-soft",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 200}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace)


def _loop(
    loop_id: str,
    body: Iterable[Mapping[str, object]],
    *,
    max_iterations: int | None = 2,
    stop_config: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {"id": loop_id, "kind": "loop", "body": list(body)}
    if max_iterations is not None:
        payload["max_iterations"] = max_iterations
    if stop_config is not None:
        payload["stop"] = dict(stop_config)
    return payload


@pytest.mark.parametrize("config_style", ["direct", "stop"])
def test_flow_runner_executes_nested_loops_without_key_errors(config_style: str) -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 30}] * 8)
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=_budget_manager(trace),
        policy_stack=_policy_stack(),
        trace=trace,
    )

    inner_body = [{"id": "inner-node", "tool": "echo", "params": {}}]
    inner_loop = _loop("inner-loop", inner_body, max_iterations=2)
    outer_body = [inner_loop, {"id": "outer-node", "tool": "echo", "params": {}}]
    if config_style == "direct":
        outer_loop = _loop("outer-loop", outer_body, max_iterations=2)
    else:
        outer_loop = _loop(
            "outer-loop",
            outer_body,
            max_iterations=None,
            stop_config={"max_iterations": "2"},
        )

    executions = runner.run(
        flow_id="flow-nested",
        run_id="run-nested",
        nodes=[outer_loop],
    )

    assert len(executions) == 6
    loop_start_events = [evt.scope_id for evt in trace.events if evt.event == "loop_start"]
    assert "outer-loop" in loop_start_events
    assert "inner-loop" in loop_start_events


def test_run_scope_commit_occurs_after_node_completion() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(estimate_costs=[{"time_ms": 45}], results=[{"ok": True}])
    manager = _budget_manager(trace)
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=_policy_stack(),
        trace=trace,
    )

    executions = runner.run(
        flow_id="flow-commit",
        run_id="run-commit",
        nodes=[{"id": "node-1", "tool": "echo", "params": {}}],
    )

    assert len(executions) == 1
    run_scope = bm.ScopeKey(scope_type="run", scope_id="run-commit")
    spent = manager.spent(run_scope, "run-hard")
    assert spent.time_ms == pytest.approx(45.0, rel=0.01)
    run_events = [evt for evt in trace.events if evt.scope_type == "run" and evt.event == "budget_charge"]
    assert run_events, "run scope charges should emit trace events"
