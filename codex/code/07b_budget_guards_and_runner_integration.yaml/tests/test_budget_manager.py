from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "budget_integration.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("budget_integration", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    sys.modules[spec.name] = module
    loader.exec_module(module)  # type: ignore[assignment]
    return module


@pytest.fixture(scope="module")
def budget_module():
    return load_module()


@pytest.fixture()
def manager(budget_module):
    TraceWriter = budget_module.ListTraceWriter
    BudgetManager = budget_module.BudgetManager

    trace = TraceWriter()
    return BudgetManager(trace_writer=trace), trace


def test_preflight_warns_on_soft_breach(manager, budget_module):
    manager_obj, trace = manager
    BudgetSpec = budget_module.BudgetSpec
    CostSnapshot = budget_module.CostSnapshot

    spec = BudgetSpec(scope="node", limits={"time_ms": 100}, breach_action="warn")
    attempt = CostSnapshot.from_mapping({"time_ms": 150})

    outcome = manager_obj.preflight("node:1", spec, attempt)

    assert outcome.allowed is True
    assert outcome.breaches
    assert outcome.remaining.metrics["time_ms"] == 0
    assert trace.events[0]["event"] == "budget_preflight"
    assert trace.events[-1]["event"] == "budget_breach"
    assert trace.events[-1]["data"]["breach_action"] == "warn"


def test_commit_blocks_hard_breach_and_sets_stop(manager, budget_module):
    manager_obj, trace = manager
    BudgetSpec = budget_module.BudgetSpec
    CostSnapshot = budget_module.CostSnapshot

    spec = BudgetSpec(scope="run", limits={"time_ms": 200}, breach_action="stop")
    manager_obj.preflight("run", spec, CostSnapshot.from_mapping({"time_ms": 0}))

    outcome = manager_obj.commit("run", CostSnapshot.from_mapping({"time_ms": 250}))

    assert outcome.allowed is False
    assert manager_obj.should_stop("run") is True
    breach = outcome.breaches[0]
    assert breach.metric == "time_ms"
    assert breach.attempted == 250
    assert breach.limit == 200
    assert trace.events[-1]["event"] == "budget_breach"
    assert trace.events[-1]["data"]["breach_action"] == "stop"


def test_commit_accumulates_spend_across_calls(manager, budget_module):
    manager_obj, _ = manager
    BudgetSpec = budget_module.BudgetSpec
    CostSnapshot = budget_module.CostSnapshot

    spec = BudgetSpec(scope="loop", limits={"tokens": 10}, breach_action="warn")
    manager_obj.preflight("loop", spec, CostSnapshot.from_mapping({"tokens": 5}))

    first = manager_obj.commit("loop", CostSnapshot.from_mapping({"tokens": 4}))
    second = manager_obj.commit("loop", CostSnapshot.from_mapping({"tokens": 7}))

    assert first.allowed is True
    assert not first.breaches
    assert second.breaches
    assert second.overage.metrics["tokens"] == 1
    assert second.remaining.metrics["tokens"] == 0
