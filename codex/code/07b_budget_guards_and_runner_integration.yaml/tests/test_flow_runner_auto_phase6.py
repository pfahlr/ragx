"""Phase 6 regression coverage for the FlowRunner."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetManager
from pkgs.dsl.flow_runner import FlowRunner, ToolAdapter
from pkgs.dsl.policy import PolicyStack
from pkgs.dsl.trace import TraceEventEmitter


class EchoAdapter(ToolAdapter):
    """Adapter that records interactions for assertions."""

    def __init__(self, *, estimate_costs: Iterable[Mapping[str, float]]) -> None:
        self._estimates = deque(dict(cost) for cost in estimate_costs)
        self.executed: list[Mapping[str, object]] = []

    def estimate_cost(self, node: Mapping[str, object]) -> Mapping[str, float]:  # type: ignore[override]
        if self._estimates:
            current = self._estimates.popleft()
            self._estimates.append(current)
        else:  # pragma: no cover - defensive guard for misconfigured tests
            current = {"time_ms": 1.0}
        return dict(current)

    def execute(self, node: Mapping[str, object]) -> Any:  # type: ignore[override]
        payload = dict(node)
        self.executed.append(payload)
        return {"ok": payload.get("id")}


def _budget_specs() -> list[bm.BudgetSpec]:
    limit = bm.CostSnapshot.from_raw({"time_ms": 500})
    return [
        bm.BudgetSpec(
            name="run",
            scope_type="run",
            limit=limit,
            mode="hard",
            breach_action="stop",
        ),
        bm.BudgetSpec(
            name="loop",
            scope_type="loop",
            limit=limit,
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node",
            scope_type="node",
            limit=limit,
            mode="soft",
            breach_action="warn",
        ),
    ]


def _runner(trace: TraceEventEmitter) -> tuple[FlowRunner, EchoAdapter]:
    adapter = EchoAdapter(estimate_costs=[{"time_ms": 5.0}])
    manager = BudgetManager(specs=_budget_specs(), trace=trace)
    stack = PolicyStack(tools={"echo": {"tags": ["default"]}})
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=manager,
        policy_stack=stack,
        trace=trace,
    )
    return runner, adapter


def _loop(
    loop_id: str,
    *,
    body: list[Mapping[str, object]],
    max_iterations: int | None = None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {"id": loop_id, "kind": "loop", "body": body}
    if max_iterations is not None:
        payload["max_iterations"] = max_iterations
    return payload


def _unit(node_id: str) -> Mapping[str, object]:
    return {"id": node_id, "tool": "echo", "params": {}}


def test_nested_loops_execute_inner_bodies_and_emit_traces() -> None:
    trace = TraceEventEmitter()
    runner, adapter = _runner(trace)

    inner_loop = _loop(
        "inner-loop",
        body=[_unit("inner-a"), _unit("inner-b")],
        max_iterations=2,
    )
    outer_loop = _loop(
        "outer-loop",
        body=[_unit("outer-a"), inner_loop, _unit("outer-b")],
        max_iterations=2,
    )

    executions = runner.run(
        flow_id="flow-nested",
        run_id="run-nested",
        nodes=[outer_loop],
    )

    executed_ids = [execution.node_id for execution in executions]
    assert executed_ids.count("outer-a") == 2
    assert executed_ids.count("outer-b") == 2
    assert executed_ids.count("inner-a") == 4
    assert executed_ids.count("inner-b") == 4
    assert any(execution.loop_id == "inner-loop" for execution in executions)

    loop_start_ids = [
        event.scope_id
        for event in trace.events
        if event.event == "loop_start" and event.scope_type == "loop"
    ]
    assert loop_start_ids.count("inner-loop") == 2
    assert loop_start_ids.count("outer-loop") == 1
    assert len(adapter.executed) == len(executions)


def test_missing_tool_field_surfaces_helpful_error() -> None:
    trace = TraceEventEmitter()
    runner, _ = _runner(trace)

    with pytest.raises(KeyError) as exc:
        runner.run(
            flow_id="flow-error",
            run_id="run-error",
            nodes=[{"id": "node-without-tool", "params": {}}],
        )

    assert "missing required field 'tool'" in str(exc.value)
