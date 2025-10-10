import decimal
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List

sys.path.append(str(pathlib.Path(__file__).resolve().parents[4]))

import pytest

from codex.code.work.runner.budget_manager import BudgetManager
from codex.code.work.runner.budget_models import BudgetScope, BudgetSpec, CostSnapshot
from codex.code.work.runner.flow_runner import FlowDefinition, FlowNode, FlowRunner, LoopDefinition, RunContext
from codex.code.work.runner.trace import InMemoryTraceWriter, TraceEventEmitter


@dataclass
class CountingAdapter:
    costs: List[CostSnapshot]
    executed: int = 0

    def estimate(self, *, node_id: str, iteration: int) -> CostSnapshot:
        return self.costs[min(iteration, len(self.costs) - 1)]

    def execute(self, *, node_id: str, iteration: int) -> Dict[str, str]:
        self.executed += 1
        return {"node_id": node_id, "iteration": str(iteration)}


@pytest.fixture
def adapters() -> Dict[str, CountingAdapter]:
    costs = [
        CostSnapshot.from_costs({"usd": "0.40", "tokens": 30, "calls": 1}),
        CostSnapshot.from_costs({"usd": "0.40", "tokens": 30, "calls": 1}),
        CostSnapshot.from_costs({"usd": "0.40", "tokens": 30, "calls": 1}),
    ]
    return {"completion": CountingAdapter(costs=costs)}


@pytest.fixture
def flow_definition() -> FlowDefinition:
    run_scope = BudgetScope.run("run-budget")
    loop_scope = BudgetScope.loop("loop-budget")
    node_scope = BudgetScope.node("node-alpha")
    spec_scope = BudgetScope.spec("node-alpha", "completion")

    node = FlowNode(node_id="node-alpha", adapter_id="completion", scopes=[node_scope, spec_scope])
    loop = LoopDefinition(loop_id="loop-budget", scope=loop_scope, iterations=4, nodes=[node])
    return FlowDefinition(flow_id="demo-flow", run_scope=run_scope, loop=loop)


@pytest.fixture
def budget_manager(flow_definition: FlowDefinition) -> BudgetManager:
    run_spec = BudgetSpec.from_dict({"mode": "hard", "max_calls": 6, "max_usd": "3.00"})
    loop_spec = BudgetSpec.from_dict({"mode": "hard", "max_calls": 2, "breach_action": "stop"})
    node_spec = BudgetSpec.from_dict({"mode": "hard", "max_calls": 4})
    spec_soft = BudgetSpec.from_dict({"mode": "soft", "max_tokens": 50, "breach_action": "warn"})
    budgets = {
        flow_definition.run_scope: run_spec,
        flow_definition.loop.scope: loop_spec,
        flow_definition.loop.nodes[0].scopes[0]: node_spec,
        flow_definition.loop.nodes[0].scopes[1]: spec_soft,
    }
    return BudgetManager(budgets=budgets)


def test_runner_stops_on_loop_budget_and_warns_on_spec(
    adapters: Dict[str, CountingAdapter],
    flow_definition: FlowDefinition,
    budget_manager: BudgetManager,
) -> None:
    writer = InMemoryTraceWriter()
    emitter = TraceEventEmitter(writer)
    runner = FlowRunner(budget_manager=budget_manager, trace_emitter=emitter, adapters=adapters)

    context = RunContext(flow_id="demo-flow", run_id="run-001")
    result = runner.run(flow_definition, context)

    assert result.status == "stopped"
    assert result.iterations_executed == 2
    assert result.stop_reason is not None
    assert result.stop_reason["scope"]["level"] == "loop"
    assert result.stop_reason["action"] == "stop"

    warnings = result.warnings
    assert len(warnings) == 1
    assert warnings[0]["scope"]["level"] == "spec"
    assert decimal.Decimal(warnings[0]["overages"]["tokens"]) == decimal.Decimal(10)

    # Trace should contain ordered budget charges and breach events
    events = writer.events
    budget_charge_events = [event for event in events if event["event"] == "budget_charge"]
    budget_breach_events = [event for event in events if event["event"] == "budget_breach"]

    assert any(event["scope"]["level"] == "loop" for event in budget_charge_events)
    assert any(event["scope"]["level"] == "spec" for event in budget_breach_events)
    assert any(event["action"] == "stop" for event in budget_breach_events)

    # Adapter should only execute for iterations that completed
    assert adapters["completion"].executed == 2
