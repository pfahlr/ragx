from importlib import util
from pathlib import Path
import sys
import types

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str):
    if "codex" not in sys.modules:
        codex_pkg = types.ModuleType("codex")
        codex_pkg.__path__ = [str(MODULE_DIR.parent)]
        sys.modules["codex"] = codex_pkg
    if "codex.07b" not in sys.modules:
        pkg = types.ModuleType("codex.07b")
        pkg.__path__ = [str(MODULE_DIR)]
        sys.modules["codex.07b"] = pkg
    spec = util.spec_from_file_location(
        f"codex.07b.{name}", MODULE_DIR / f"{name}.py"
    )
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


bm = load_module("budget_models")
manager_mod = load_module("budget_manager")
trace_mod = load_module("trace_emitter")
runner_mod = load_module("flow_runner")


class FakeAdapter:
    def estimate_cost(self, context):
        return {"duration_ms": context["node"]["estimate_ms"]}

    def execute(self, context):
        return {
            "output": f"ran:{context['node']['id']}",
            "cost": {"duration_ms": context["node"]["execute_ms"]},
        }


@pytest.fixture
def adapter_registry():
    return {
        "alpha": FakeAdapter(),
        "beta": FakeAdapter(),
    }


def test_flow_runner_stops_on_spec_budget_breach(adapter_registry):
    manager = manager_mod.BudgetManager()
    emitter = trace_mod.TraceEventEmitter()
    runner = runner_mod.FlowRunner(budget_manager=manager, trace_emitter=emitter)

    flow_spec = {
        "run_id": "run-hard-stop",
        "run_budget": {"limit_ms": 150, "breach_action": "stop"},
        "spec_budgets": {
            "embedding": {"limit_ms": 100, "breach_action": "stop"},
        },
        "nodes": [
                {
                    "id": "node-1",
                    "adapter": "alpha",
                    "policy": "policy-a",
                    "spec_id": "embedding",
                    "budget": {"limit_ms": 50, "breach_action": "warn"},
                    "estimate_ms": 45,
                    "execute_ms": 60,
                },
            {
                "id": "node-2",
                "adapter": "alpha",
                "policy": "policy-a",
                "spec_id": "embedding",
                "budget": {"limit_ms": 70, "breach_action": "stop"},
                "estimate_ms": 55,
                "execute_ms": 65,
            },
        ],
    }

    result = runner.run(flow_spec, adapter_registry)

    assert result.stop_reason == "budget_breach:spec:embedding"
    assert [n.node_id for n in result.executions] == ["node-1"]
    assert len(result.warnings) == 1
    assert any("node:node-1" in warning for warning in result.warnings)

    event_types = [event["type"] for event in emitter.events]
    assert event_types[0] == "policy_push"
    breach_events = [event for event in emitter.events if event["type"] == "budget_breach"]
    assert any(event["payload"]["scope"] == "spec" for event in breach_events)
    spec_breach = next(event for event in breach_events if event["payload"]["scope"] == "spec")
    assert spec_breach["payload"]["scope_id"] == "embedding"
    assert spec_breach["payload"]["stage"] == "preflight"
    assert any(event["type"] == "policy_violation" for event in emitter.events)


def test_flow_runner_accumulates_run_and_loop_warnings(adapter_registry):
    manager = manager_mod.BudgetManager()
    emitter = trace_mod.TraceEventEmitter()
    runner = runner_mod.FlowRunner(budget_manager=manager, trace_emitter=emitter)

    flow_spec = {
        "run_id": "run-soft",
        "run_budget": {"limit_ms": 60, "breach_action": "warn"},
        "loop_budgets": {
            "loop-1": {"limit_ms": 50, "breach_action": "warn"},
        },
        "nodes": [
                {
                    "id": "node-a",
                    "adapter": "alpha",
                    "policy": "policy-a",
                    "loop_id": "loop-1",
                    "budget": {"limit_ms": 30, "breach_action": "warn"},
                    "estimate_ms": 30,
                    "execute_ms": 35,
                },
                {
                    "id": "node-b",
                    "adapter": "beta",
                    "policy": "policy-b",
                    "loop_id": "loop-1",
                    "budget": {"limit_ms": 20, "breach_action": "warn"},
                    "estimate_ms": 25,
                    "execute_ms": 30,
                },
        ],
    }

    result = runner.run(flow_spec, adapter_registry)

    assert result.stop_reason is None
    assert [n.node_id for n in result.executions] == ["node-a", "node-b"]
    assert len(result.warnings) >= 3
    expected_prefixes = {"node:node-a", "node:node-b", "loop:loop-1", "run:run-soft"}
    for prefix in expected_prefixes:
        assert any(warning.startswith(prefix) for warning in result.warnings)

    run_warnings = [w for w in result.warnings if w.startswith("run:run-soft")]
    assert len(run_warnings) == 1

    charge_events = [event for event in emitter.events if event["type"] == "budget_charge"]
    assert len(charge_events) >= 2
    assert all(event["payload"]["stage"] == "commit" for event in charge_events)
