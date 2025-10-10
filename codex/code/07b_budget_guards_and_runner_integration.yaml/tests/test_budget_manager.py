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


@pytest.fixture
def manager():
    return manager_mod.BudgetManager()


def test_preflight_and_commit_enforce_hard_budget(manager):
    run_scope = bm.ScopeKey("run", "run-1")
    spec = bm.BudgetSpec.from_config(scope=run_scope, config={"limit_ms": 100, "breach_action": "stop"})
    manager.register_scope(spec)

    first_cost = bm.CostSnapshot.from_inputs(duration_ms=60)
    decision = manager.preflight(run_scope, first_cost)
    assert decision.allowed is True
    assert decision.breach is None

    commit_outcome = manager.commit(run_scope, first_cost)
    assert commit_outcome.allowed is True
    assert commit_outcome.remaining.milliseconds == pytest.approx(40)

    second_cost = bm.CostSnapshot.from_inputs(duration_ms=50)
    second_decision = manager.preflight(run_scope, second_cost)
    assert second_decision.allowed is False
    assert second_decision.breach is not None
    assert second_decision.breach.limit_ms == 100

    # Hard stop should not mutate spent totals on failed commit.
    with pytest.raises(manager_mod.BudgetBreachError):
        manager.commit(run_scope, second_cost)

    # Remaining budget should still reflect the first commit only.
    post_state = manager.snapshot(run_scope)
    assert post_state.spent.milliseconds == pytest.approx(60)


def test_soft_budget_allows_commit_but_records_warning(manager):
    loop_scope = bm.ScopeKey("loop", "loop-a")
    spec = bm.BudgetSpec.from_config(scope=loop_scope, config={"limit_ms": 90, "breach_action": "warn"})
    manager.register_scope(spec)

    cost = bm.CostSnapshot.from_inputs(duration_ms=70)
    manager.commit(loop_scope, cost)

    over_cost = bm.CostSnapshot.from_inputs(duration_ms=40)
    decision = manager.preflight(loop_scope, over_cost)
    assert decision.allowed is True
    assert decision.breach is not None
    assert decision.breach.action == bm.BreachAction.WARN

    outcome = manager.commit(loop_scope, over_cost)
    assert outcome.allowed is True
    assert outcome.breach is not None
    assert outcome.remaining.milliseconds == pytest.approx(0.0)

    warnings = manager.drain_warnings()
    assert len(warnings) == 1
    assert loop_scope.identifier in warnings[0]


def test_registering_scope_twice_raises(manager):
    spec = bm.BudgetSpec.from_config(scope=bm.ScopeKey("spec", "embedding"), config={"limit_ms": 30})
    manager.register_scope(spec)
    with pytest.raises(ValueError):
        manager.register_scope(spec)


def test_snapshot_returns_immutable_view(manager):
    node_scope = bm.ScopeKey("node", "node-1")
    spec = bm.BudgetSpec.from_config(scope=node_scope, config={"limit_ms": 50})
    manager.register_scope(spec)
    manager.commit(node_scope, bm.CostSnapshot.from_inputs(duration_ms=20))

    snapshot = manager.snapshot(node_scope)
    payload = snapshot.to_payload()
    assert payload["spent_ms"] == pytest.approx(20.0)
    with pytest.raises(TypeError):
        payload["spent_ms"] = 10  # type: ignore[index]
