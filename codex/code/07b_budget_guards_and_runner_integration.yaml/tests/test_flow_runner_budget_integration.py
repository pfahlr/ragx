import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkgs.dsl.policy import PolicyStack, PolicyTraceRecorder


def _load_module(name: str):
    module_path = pathlib.Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"task07b_{name}", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"unable to import {name}")
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


budget_mod = _load_module("budget")
trace_mod = _load_module("trace")
runner_mod = _load_module("runner")
adapters_mod = _load_module("adapters")

CostSnapshot = budget_mod.CostSnapshot
BudgetSpec = budget_mod.BudgetSpec
BudgetMode = budget_mod.BudgetMode
BudgetManager = budget_mod.BudgetManager
BudgetBreach = budget_mod.BudgetBreach

TraceEventEmitter = trace_mod.TraceEventEmitter
TraceEvent = trace_mod.TraceEvent

FlowRunner = runner_mod.FlowRunner
RunPlan = runner_mod.RunPlan
LoopPlan = runner_mod.LoopPlan
NodePlan = runner_mod.NodePlan
RunResult = runner_mod.RunResult

AdapterContext = adapters_mod.AdapterContext
AdapterResult = adapters_mod.AdapterResult
ToolAdapter = adapters_mod.ToolAdapter


def make_cost(value: int) -> CostSnapshot:
    return CostSnapshot({"time_ms": value})


class DeterministicAdapter:
    def __init__(self, name: str, *, execute_costs: list[int]) -> None:
        self.name = name
        self._costs = list(execute_costs)

    def estimate(self, context: AdapterContext) -> AdapterResult:
        if not self._costs:
            raise RuntimeError("no more costs available for estimate")
        return AdapterResult(output={"preview": True}, cost=make_cost(self._costs[0]))

    def execute(self, context: AdapterContext) -> AdapterResult:
        if not self._costs:
            raise RuntimeError("no more costs available for execute")
        value = self._costs.pop(0)
        return AdapterResult(output={"executed": context.node_id}, cost=make_cost(value))


class FakeTraceWriter:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


def test_flow_runner_stops_loop_on_budget_breach_and_emits_traces() -> None:
    policy_recorder = PolicyTraceRecorder()
    policy_stack = PolicyStack(
        tools={
            "tool.echo": {"tags": ["default"]},
        },
        trace=policy_recorder,
    )
    adapters = {"tool.echo": DeterministicAdapter("tool.echo", execute_costs=[400, 300, 300])}
    budget_manager = BudgetManager()
    trace_writer = FakeTraceWriter()
    trace_emitter = TraceEventEmitter(trace_writer)

    runner = FlowRunner(
        adapters=adapters,
        policy_stack=policy_stack,
        budget_manager=budget_manager,
        trace_emitter=trace_emitter,
    )

    run_plan = RunPlan(
        run_id="run-1",
        run_budget=BudgetSpec(
            scope_type="run",
            scope_id="run-1",
            limit=make_cost(1000),
            mode=BudgetMode.HARD,
            breach_action="stop",
        ),
        loops=(
            LoopPlan(
                loop_id="loop-1",
                iterations=3,
                budget=BudgetSpec(
                    scope_type="loop",
                    scope_id="loop-1",
                    limit=make_cost(600),
                    mode=BudgetMode.HARD,
                    breach_action="stop",
                ),
                nodes=(
                    NodePlan(
                        node_id="node-1",
                        tool="tool.echo",
                        budget=BudgetSpec(
                            scope_type="node",
                            scope_id="node-1",
                            limit=make_cost(400),
                            mode=BudgetMode.HARD,
                            breach_action="stop",
                        ),
                    ),
                ),
            ),
        ),
    )

    result = runner.run(run_plan)

    assert isinstance(result, RunResult)
    assert result.stop_reason == "loop budget exceeded"
    assert result.breaches
    assert any(b.scope_type == "loop" for b in result.breaches)
    assert len(result.outputs) == 2

    assert any(event.event == "budget_breach" for event in trace_writer.events)
    assert any(event.event == "loop_summary" for event in trace_writer.events)
    assert any(event.event == "budget_charge" for event in trace_writer.events)

    assert any(evt.event == "policy_resolved" for evt in policy_recorder.events)


def test_flow_runner_raises_policy_violation_before_budget_charge() -> None:
    policy_recorder = PolicyTraceRecorder()
    policy_stack = PolicyStack(
        tools={},
        trace=policy_recorder,
    )
    adapters = {"tool.echo": DeterministicAdapter("tool.echo", execute_costs=[100])}
    budget_manager = BudgetManager()
    trace_writer = FakeTraceWriter()
    trace_emitter = TraceEventEmitter(trace_writer)
    runner = FlowRunner(
        adapters=adapters,
        policy_stack=policy_stack,
        budget_manager=budget_manager,
        trace_emitter=trace_emitter,
    )

    run_plan = RunPlan(
        run_id="run-viol",
        run_budget=None,
        loops=(
            LoopPlan(
                loop_id="loop-viol",
                iterations=1,
                budget=None,
                nodes=(
                    NodePlan(
                        node_id="node-viol",
                        tool="tool.echo",
                        budget=None,
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(Exception):
        runner.run(run_plan)
    assert all(event.event != "budget_charge" for event in trace_writer.events)
