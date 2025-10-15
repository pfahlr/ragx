"""Phase 6 regression tests for FlowRunner nested-loop behaviours."""

from __future__ import annotations

from collections.abc import Iterable

from pkgs.dsl import budget_models as bm
from pkgs.dsl.budget_manager import BudgetManager
from pkgs.dsl.flow_runner import FlowRunner
from pkgs.dsl.policy import PolicyStack
from pkgs.dsl.trace import TraceEventEmitter


class RecordingAdapter:
    """Tool adapter that records invocation order for assertions."""

    def __init__(
        self,
        *,
        estimate_costs: Iterable[dict[str, object]],
        results: Iterable[object],
    ) -> None:
        self._estimate_iter = iter(estimate_costs)
        self._result_iter = iter(results)
        self.estimate_order: list[str] = []
        self.execute_order: list[str] = []

    def estimate_cost(self, node: dict[str, object]) -> dict[str, object]:
        self.estimate_order.append(str(node["id"]))
        return dict(next(self._estimate_iter))

    def execute(self, node: dict[str, object]) -> object:
        self.execute_order.append(str(node["id"]))
        return next(self._result_iter)


def _budget_manager(trace: TraceEventEmitter) -> BudgetManager:
    specs = [
        bm.BudgetSpec(
            name="run",
            scope_type="run",
            limit=bm.CostSnapshot.from_raw({"time_ms": 10_000}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="loop",
            scope_type="loop",
            limit=bm.CostSnapshot.from_raw({"time_ms": 5_000}),
            mode="soft",
            breach_action="warn",
        ),
        bm.BudgetSpec(
            name="node",
            scope_type="node",
            limit=bm.CostSnapshot.from_raw({"time_ms": 2_000}),
            mode="soft",
            breach_action="warn",
        ),
    ]
    return BudgetManager(specs=specs, trace=trace)


def _policy_stack() -> PolicyStack:
    return PolicyStack(tools={"echo": {"tags": []}})


def _unit(node_id: str) -> dict[str, object]:
    return {"id": node_id, "tool": "echo"}


def _loop(node_id: str, body: list[dict[str, object]], *, max_iterations: int) -> dict[str, object]:
    return {
        "id": node_id,
        "kind": "loop",
        "body": body,
        "max_iterations": max_iterations,
    }


def test_flow_runner_handles_nested_loops_without_scope_leakage() -> None:
    trace = TraceEventEmitter()
    adapter = RecordingAdapter(
        estimate_costs=[{"time_ms": 10}] * 6,
        results=[{"ok": True}] * 6,
    )
    runner = FlowRunner(
        adapters={"echo": adapter},
        budget_manager=_budget_manager(trace),
        policy_stack=_policy_stack(),
        trace=trace,
    )

    inner_loop = _loop(
        "loop-inner",
        body=[_unit("inner-node")],
        max_iterations=2,
    )
    outer_loop = _loop(
        "loop-outer",
        body=[_unit("outer-node"), inner_loop],
        max_iterations=2,
    )

    executions = runner.run(
        flow_id="flow-nested",
        run_id="run-nested",
        nodes=[outer_loop],
    )

    assert [exec.node_id for exec in executions] == [
        "outer-node",
        "inner-node",
        "inner-node",
        "outer-node",
        "inner-node",
        "inner-node",
    ]
    assert [exec.loop_id for exec in executions] == [
        "loop-outer",
        "loop-inner",
        "loop-inner",
        "loop-outer",
        "loop-inner",
        "loop-inner",
    ]
    assert [exec.iteration for exec in executions] == [
        1,
        1,
        2,
        2,
        1,
        2,
    ]

    loop_complete = [
        evt
        for evt in trace.events
        if evt.event == "loop_complete"
    ]
    assert [evt.scope_id for evt in loop_complete if evt.scope_id == "loop-outer"] == [
        "loop-outer"
    ]
    inner_completions = [
        evt
        for evt in loop_complete
        if evt.scope_id == "loop-inner"
    ]
    # Inner loop executes once per outer iteration -> two completion events
    assert len(inner_completions) == 2
    assert all(evt.payload["iterations"] == 2 for evt in inner_completions)

    # Adapter should see interleaved estimate/execute calls mirroring node order
    assert adapter.estimate_order == [
        "outer-node",
        "inner-node",
        "inner-node",
        "outer-node",
        "inner-node",
        "inner-node",
    ]
    assert adapter.execute_order == adapter.estimate_order
