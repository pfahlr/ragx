import pytest

from pkgs.dsl.policy import PolicyStack

from phase3_budget_runner_71d5.adapters import ToolAdapter, ToolExecutionResult
from phase3_budget_runner_71d5.budgeting import (
    BreachAction,
    BudgetContext,
    BudgetManager,
    BudgetMode,
    BudgetSpec,
    CostSnapshot,
    ScopeKey,
    ScopeType,
)
from phase3_budget_runner_71d5.runner import FlowPlan, FlowRunner, LoopPlan, NodeExecution, NodePlan
from phase3_budget_runner_71d5.trace import TraceEventEmitter, TraceRecorder


class DeterministicAdapter(ToolAdapter):
    def __init__(self, name: str, cost_sequence: list[float]) -> None:
        self.name = name
        self._costs = list(cost_sequence)

    def estimate(self, node: NodePlan, context: dict[str, object] | None = None) -> CostSnapshot:
        return CostSnapshot.from_raw({"tokens": self._costs[0]})

    def execute(self, node: NodePlan, context: dict[str, object] | None = None) -> ToolExecutionResult:
        cost_value = self._costs.pop(0)
        return ToolExecutionResult(result={"ok": True}, cost=CostSnapshot.from_raw({"tokens": cost_value}))


@pytest.fixture()
def runner_setup():
    recorder = TraceRecorder()
    emitter = TraceEventEmitter(recorder=recorder)
    manager = BudgetManager(trace_emitter=emitter)
    tools = {"echo": {"tags": []}}
    policy = PolicyStack(tools=tools, event_sink=emitter.policy_sink())
    policy.push({"allow_tools": ["echo"]}, scope="run")
    adapters = {"echo": DeterministicAdapter("echo", [30, 70, 30, 30, 30])}
    runner = FlowRunner(
        adapters=adapters,
        policy_stack=policy,
        budget_manager=manager,
        trace_emitter=emitter,
    )
    return runner, recorder, policy


def test_runner_enforces_budget_and_stops_on_run_breach(runner_setup):
    runner, recorder, policy = runner_setup
    plan = FlowPlan(
        flow_id="flow-1",
        run_budget=BudgetSpec("run", CostSnapshot.from_raw({"tokens": 90})),
        steps=(
            NodePlan(node_id="n1", tool="echo", budget=None),
            NodePlan(node_id="n2", tool="echo", budget=None),
        ),
    )

    result = runner.run(plan)
    assert result.stop_reason == "budget_breach:run"
    assert len(result.executions) == 2
    assert result.executions[-1].stopped is True

    breach_events = [evt for evt in recorder.events if evt.event == "budget_breach"]
    assert breach_events
    assert any(evt.payload["scope_type"] == "run" for evt in breach_events)

    policy.pop("run")


def test_loop_summary_and_soft_warning(runner_setup):
    runner, recorder, policy = runner_setup
    plan = FlowPlan(
        flow_id="flow-soft",
        run_budget=BudgetSpec("run", CostSnapshot.from_raw({"tokens": 400})),
        steps=(
            LoopPlan(
                loop_id="loop-1",
                iterations=3,
                budget=BudgetSpec(
                    "loop",
                    CostSnapshot.from_raw({"tokens": 60}),
                    mode=BudgetMode.SOFT,
                    breach_action=BreachAction.WARN,
                ),
                body=(NodePlan(node_id="loop-node", tool="echo", budget=None),),
            ),
        ),
    )

    result = runner.run(plan)
    assert result.stop_reason is None
    assert len(result.executions) == 3
    assert all(isinstance(execution, NodeExecution) for execution in result.executions)
    assert any("breached" in warning for warning in result.warnings)

    summaries = [evt for evt in recorder.events if evt.event == "loop_summary"]
    assert summaries
    summary_payload = summaries[0].payload
    assert summary_payload["iterations"] == 3
    assert summary_payload["warnings"]

    policy.pop("run")
