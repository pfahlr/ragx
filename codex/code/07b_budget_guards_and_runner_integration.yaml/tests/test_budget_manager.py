from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
BudgetDecision = budget_manager_mod.BudgetDecision


class DummyEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def emit(self, event: str, scope: str, payload: dict[str, object]) -> None:
        self.events.append((event, scope, payload))


@pytest.fixture()
def manager() -> tuple[BudgetManager, DummyEmitter]:
    emitter = DummyEmitter()
    mgr = BudgetManager(trace_emitter=emitter.emit)
    return mgr, emitter


def test_commit_updates_parent_and_child_scopes(manager: tuple[BudgetManager, DummyEmitter]) -> None:
    mgr, emitter = manager

    run_spec = BudgetSpec(scope="run", mode=BudgetMode.HARD, limits={"tokens": 150}, breach_action="error")
    node_spec = BudgetSpec(scope="node:alpha", mode=BudgetMode.SOFT, limits={"tokens": 80}, breach_action="warn")

    mgr.register_scope("run", run_spec)
    mgr.register_scope("node:alpha", node_spec, parent="run")

    decision = mgr.commit("node:alpha", CostSnapshot.from_values(tokens=60), label="node-alpha")
    assert isinstance(decision, BudgetDecision)
    assert len(decision.charges) == 2
    child_charge = decision.charges[0]
    run_charge = decision.charges[1]
    assert dict(child_charge.remaining)["tokens"] == 20
    assert dict(run_charge.remaining)["tokens"] == 90
    assert not decision.stop_requested
    assert len(emitter.events) == 2

    with pytest.raises(BudgetBreachError):
        mgr.commit("node:alpha", CostSnapshot.from_values(tokens=100), label="breach")


def test_stop_request_propagates_from_loop_budget(manager: tuple[BudgetManager, DummyEmitter]) -> None:
    mgr, _ = manager

    run_spec = BudgetSpec(scope="run", mode=BudgetMode.HARD, limits={"calls": 10}, breach_action="error")
    loop_spec = BudgetSpec(scope="loop:collect", mode=BudgetMode.SOFT, limits={"calls": 2}, breach_action="stop")
    node_spec = BudgetSpec(scope="node:collect", mode=BudgetMode.SOFT, limits={"calls": 3}, breach_action="warn")

    mgr.register_scope("run", run_spec)
    mgr.register_scope("loop:collect", loop_spec, parent="run")
    mgr.register_scope("node:collect", node_spec, parent="loop:collect")

    first = mgr.commit("node:collect", CostSnapshot.from_values(calls=1), label="iter1")
    assert not first.stop_requested

    second = mgr.commit("node:collect", CostSnapshot.from_values(calls=2), label="iter2")
    assert second.stop_requested
    assert any(charge.breach_action == "stop" for charge in second.breaches)


def test_preflight_does_not_mutate_scope_state(manager: tuple[BudgetManager, DummyEmitter]) -> None:
    mgr, _ = manager

    run_spec = BudgetSpec(scope="run", mode=BudgetMode.HARD, limits={"tokens": 50}, breach_action="error")
    node_spec = BudgetSpec(scope="node:beta", mode=BudgetMode.SOFT, limits={"tokens": 30}, breach_action="warn")

    mgr.register_scope("run", run_spec)
    mgr.register_scope("node:beta", node_spec, parent="run")

    preview = mgr.preflight("node:beta", CostSnapshot.from_values(tokens=80))
    assert preview.stop_requested
    assert any(charge.scope == "node:beta" for charge in preview.breaches)

    decision = mgr.commit("node:beta", CostSnapshot.from_values(tokens=10), label="post-preview")
    assert not decision.breached
