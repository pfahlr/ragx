import math
import types
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


def test_budget_spec_normalizes_seconds_to_milliseconds():
    spec = bm.BudgetSpec.from_config(
        scope=bm.ScopeKey("run", "run-123"),
        config={"limit_seconds": 0.25, "breach_action": "warn"},
    )
    assert spec.limit_ms == 250
    assert spec.breach_action == bm.BreachAction.WARN
    assert spec.scope.identifier == "run-123"


def test_budget_spec_requires_positive_limit():
    with pytest.raises(ValueError):
        bm.BudgetSpec.from_config(
            scope=bm.ScopeKey("node", "n1"),
            config={"limit_ms": 0},
        )


def test_cost_snapshot_supports_mixed_units_and_arithmetic():
    base = bm.CostSnapshot.from_inputs(duration_ms=125.5)
    increment = bm.CostSnapshot.from_inputs(duration_seconds=0.120)
    total = base + increment
    assert math.isclose(total.milliseconds, 245.5, rel_tol=1e-9)
    remaining = total - bm.CostSnapshot.from_inputs(duration_ms=30.5)
    assert math.isclose(remaining.milliseconds, 215.0, rel_tol=1e-9)


def test_cost_snapshot_prevents_negative_values():
    with pytest.raises(ValueError):
        bm.CostSnapshot.from_inputs(duration_ms=-1)


def test_budget_breach_payload_is_immutable_mapping():
    spec = bm.BudgetSpec.from_config(
        scope=bm.ScopeKey("spec", "embedding"),
        config={"limit_ms": 90, "breach_action": "stop"},
    )
    breach = bm.BudgetBreach(
        scope=spec.scope,
        attempted=bm.CostSnapshot.from_inputs(duration_ms=120),
        limit_ms=spec.limit_ms,
        remaining=bm.CostSnapshot.zero(),
        action=spec.breach_action,
    )
    payload = breach.to_payload()
    assert isinstance(payload, types.MappingProxyType)
    assert payload["scope"] == "spec"
    with pytest.raises(TypeError):
        payload["scope"] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize(
    "config,expected_action",
    [
        ({"limit_ms": 10, "breach_action": "warn"}, bm.BreachAction.WARN),
        ({"limit_ms": 10, "breach_action": "stop"}, bm.BreachAction.STOP),
        ({"limit_ms": 10}, bm.BreachAction.STOP),
    ],
)
def test_budget_spec_breach_actions_normalized(config, expected_action):
    spec = bm.BudgetSpec.from_config(scope=bm.ScopeKey("loop", "loop-1"), config=config)
    assert spec.breach_action == expected_action


def test_budget_decision_to_payload_reflects_stage_and_breach():
    spec = bm.BudgetSpec.from_config(scope=bm.ScopeKey("run", "run"), config={"limit_ms": 100})
    decision = bm.BudgetDecision(
        scope=spec.scope,
        stage="preflight",
        attempted=bm.CostSnapshot.from_inputs(duration_ms=120),
        remaining=bm.CostSnapshot.from_inputs(duration_ms=0),
        allowed=False,
        action=spec.breach_action,
        breach=bm.BudgetBreach(
            scope=spec.scope,
            attempted=bm.CostSnapshot.from_inputs(duration_ms=120),
            limit_ms=spec.limit_ms,
            remaining=bm.CostSnapshot.from_inputs(duration_ms=0),
            action=spec.breach_action,
        ),
    )
    payload = decision.to_payload()
    assert payload["stage"] == "preflight"
    assert payload["allowed"] is False
    assert payload["breach"]["limit_ms"] == 100
