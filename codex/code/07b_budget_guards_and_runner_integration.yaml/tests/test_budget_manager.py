import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_budget_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "budget.py"
    spec = importlib.util.spec_from_file_location("task07b_budget", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("unable to load budget module specification")
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


budget_mod = _load_budget_module()
CostSnapshot = budget_mod.CostSnapshot
BudgetSpec = budget_mod.BudgetSpec
BudgetMode = budget_mod.BudgetMode
BudgetManager = budget_mod.BudgetManager


def make_cost(time_ms: int) -> CostSnapshot:
    return CostSnapshot({"time_ms": time_ms})


def test_hard_budget_breach_stops_scope_and_reports_overage() -> None:
    manager = BudgetManager()
    run_spec = BudgetSpec(
        scope_type="run",
        scope_id="run-1",
        limit=make_cost(1000),
        mode=BudgetMode.HARD,
        breach_action="stop",
    )

    with manager.open_scope(run_spec) as run_scope:
        first = run_scope.charge(make_cost(400))
        run_outcome = first.outcome_for("run", "run-1")
        assert run_outcome.remaining.metrics["time_ms"] == 600
        assert run_outcome.overages.metrics["time_ms"] == 0
        assert not first.should_stop

        second = run_scope.charge(make_cost(700))
        run_outcome = second.outcome_for("run", "run-1")
        assert run_outcome.remaining.metrics["time_ms"] == 0
        assert run_outcome.overages.metrics["time_ms"] == 100
        assert second.should_stop
        assert second.breached is not None
        assert second.breached.kind == "hard"
        assert second.breached.action == "stop"


def test_charge_propagates_to_parent_scopes() -> None:
    manager = BudgetManager()
    run_spec = BudgetSpec(
        scope_type="run",
        scope_id="run-1",
        limit=make_cost(1200),
        mode=BudgetMode.HARD,
        breach_action="stop",
    )
    loop_spec = BudgetSpec(
        scope_type="loop",
        scope_id="loop-1",
        limit=make_cost(700),
        mode=BudgetMode.HARD,
        breach_action="stop",
    )

    with manager.open_scope(run_spec):
        with manager.open_scope(loop_spec) as loop_scope:
            result = loop_scope.charge(make_cost(300))
            run_outcome = result.outcome_for("run", "run-1")
            loop_outcome = result.outcome_for("loop", "loop-1")
            assert run_outcome.spent.metrics["time_ms"] == 300
            assert loop_outcome.spent.metrics["time_ms"] == 300
            assert run_outcome.remaining.metrics["time_ms"] == 900
            assert loop_outcome.remaining.metrics["time_ms"] == 400

            breach = loop_scope.charge(make_cost(500))
            loop_outcome = breach.outcome_for("loop", "loop-1")
            run_outcome = breach.outcome_for("run", "run-1")
            assert breach.should_stop
            assert breach.breached is not None
            assert breach.breached.scope_type == "loop"
            assert loop_outcome.overages.metrics["time_ms"] == 100
            assert run_outcome.overages.metrics["time_ms"] == 0


def test_soft_budget_warns_without_stopping() -> None:
    manager = BudgetManager()
    soft_spec = BudgetSpec(
        scope_type="loop",
        scope_id="loop-soft",
        limit=make_cost(300),
        mode=BudgetMode.SOFT,
        breach_action="warn",
    )

    with manager.open_scope(soft_spec) as scope:
        result = scope.charge(make_cost(350))
        loop_outcome = result.outcome_for("loop", "loop-soft")
        assert not result.should_stop
        assert result.breached is not None
        assert result.breached.kind == "soft"
        assert loop_outcome.overages.metrics["time_ms"] == 50
        assert loop_outcome.remaining.metrics["time_ms"] == 0
        assert "warn" in result.breached.action
