from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str):
    module_path = MODULE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


budget_models = load_module("budget_models")
BudgetSpec = budget_models.BudgetSpec
BudgetMode = budget_models.BudgetMode
BudgetBreachError = budget_models.BudgetBreachError
CostSnapshot = budget_models.CostSnapshot

budget_manager_mod = load_module("budget_manager")
BudgetManager = budget_manager_mod.BudgetManager

trace_mod = load_module("trace")
TraceEventEmitter = trace_mod.TraceEventEmitter

flow_runner_mod = load_module("flow_runner")
FlowRunner = flow_runner_mod.FlowRunner
FlowNode = flow_runner_mod.FlowNode
LoopConfig = flow_runner_mod.LoopConfig
FlowPlan = flow_runner_mod.FlowPlan
ExecutionReport = flow_runner_mod.ExecutionReport


class SequencedAdapter:
    def __init__(self, costs: Iterable[CostSnapshot]) -> None:
        self._costs = list(costs)
        self._index = 0

    def estimate_cost(self) -> CostSnapshot:
        return self._costs[self._index]

    def execute(self) -> ExecutionReport:
        cost = self._costs[self._index]
        self._index += 1
        return ExecutionReport(output={"iteration": self._index}, cost=cost, metadata={})


def build_runner(run_budget: BudgetSpec | None = None) -> tuple[FlowRunner, BudgetManager, TraceEventEmitter]:
    emitter = TraceEventEmitter()
    manager = BudgetManager(trace_emitter=emitter.emit)
    runner = FlowRunner(budget_manager=manager, trace_emitter=emitter)
    if run_budget is not None:
        manager.register_scope("run", run_budget)
    return runner, manager, emitter


def test_flow_runner_stops_loop_on_budget_stop():
    run_budget = BudgetSpec(scope="run", mode=BudgetMode.HARD, limits={"calls": 10}, breach_action="error")
    loop_budget = BudgetSpec(scope="loop:collect", mode=BudgetMode.SOFT, limits={"calls": 2}, breach_action="stop")
    node_budget = BudgetSpec(scope="node:collect", mode=BudgetMode.SOFT, limits={"calls": 5}, breach_action="warn")

    runner, manager, emitter = build_runner(run_budget)
    manager.register_scope("loop:collect", loop_budget, parent="run")
    manager.register_scope("node:collect", node_budget, parent="loop:collect")

    costs = [CostSnapshot.from_values(calls=1) for _ in range(5)]
    adapters = {"node:collect": SequencedAdapter(costs)}

    plan = FlowPlan(
        run_id="run-123",
        nodes=(
            LoopConfig(
                id="loop:collect",
                nodes=(FlowNode(id="node:collect", kind="unit", adapter_key="node:collect"),),
                max_iterations=5,
                budget_scope="loop:collect",
            ),
        ),
    )

    result = runner.run(plan=plan, adapters=adapters)

    summary = result.loop_summaries["loop:collect"]
    assert summary.iterations == 2
    assert summary.stop_reason == "budget_stop"

    loop_events = [event for event in emitter.events if event.event == "budget_charge" and event.scope == "loop:collect"]
    assert len(loop_events) == 2
    with pytest.raises(TypeError):
        loop_events[0].payload["remaining"]["calls"] = 99  # type: ignore[index]

    stop_events = [event for event in emitter.events if event.event == "loop_stop"]
    assert stop_events and stop_events[0].payload["reason"] == "budget_stop"


def test_flow_runner_raises_on_hard_run_breach():
    run_budget = BudgetSpec(scope="run", mode=BudgetMode.HARD, limits={"usd": 3.0}, breach_action="error")
    node_budget = BudgetSpec(scope="node:expensive", mode=BudgetMode.HARD, limits={"usd": 2.0}, breach_action="error")

    runner, manager, emitter = build_runner(run_budget)
    manager.register_scope("node:expensive", node_budget, parent="run")

    adapters = {
        "node:expensive": SequencedAdapter([CostSnapshot.from_values(usd=4.5)])
    }

    plan = FlowPlan(
        run_id="run-456",
        nodes=(FlowNode(id="node:expensive", kind="unit", adapter_key="node:expensive"),),
    )

    with pytest.raises(BudgetBreachError):
        runner.run(plan=plan, adapters=adapters)

    assert any(event.event == "budget_breach" and event.scope == "run" for event in emitter.events)


def test_preflight_blocks_iteration_before_execute():
    run_budget = BudgetSpec(scope="run", mode=BudgetMode.HARD, limits={"tokens": 100}, breach_action="error")
    loop_budget = BudgetSpec(scope="loop:writer", mode=BudgetMode.SOFT, limits={"tokens": 3}, breach_action="stop")
    node_budget = BudgetSpec(scope="node:writer", mode=BudgetMode.SOFT, limits={"tokens": 5}, breach_action="warn")

    runner, manager, emitter = build_runner(run_budget)
    manager.register_scope("loop:writer", loop_budget, parent="run")
    manager.register_scope("node:writer", node_budget, parent="loop:writer")

    adapters = {"node:writer": SequencedAdapter([CostSnapshot.from_values(tokens=1) for _ in range(4)])}

    plan = FlowPlan(
        run_id="run-789",
        nodes=(
            LoopConfig(
                id="loop:writer",
                nodes=(FlowNode(id="node:writer", kind="unit", adapter_key="node:writer"),),
                max_iterations=10,
                budget_scope="loop:writer",
            ),
        ),
    )

    result = runner.run(plan=plan, adapters=adapters)

    summary = result.loop_summaries["loop:writer"]
    assert summary.iterations == 3
    assert summary.stop_reason == "budget_stop"

    loop_charges = [event for event in emitter.events if event.event == "budget_charge" and event.scope == "loop:writer"]
    assert len(loop_charges) == 3
