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
BudgetMeter = budget_models.BudgetMeter
CostSnapshot = budget_models.CostSnapshot


def test_cost_snapshot_normalizes_seconds_to_milliseconds():
    snapshot = CostSnapshot.from_values(time_seconds=1.25)
    assert snapshot.time_ms == 1250
    assert snapshot.calls == 0
    assert snapshot.tokens == 0
    assert snapshot.usd == 0.0


def test_budget_meter_soft_budget_records_overage_without_stop():
    spec = BudgetSpec(
        scope="node:alpha",
        mode=BudgetMode.SOFT,
        limits={"tokens": 100},
        breach_action="warn",
    )
    meter = BudgetMeter(spec)

    first = meter.charge(CostSnapshot.from_values(tokens=80), label="first")
    assert not first.breached
    assert dict(first.remaining)["tokens"] == 20
    assert dict(first.overages)["tokens"] == 0

    second = meter.charge(CostSnapshot.from_values(tokens=30), label="second")
    assert second.breached
    assert second.breach_kind == BudgetMode.SOFT
    assert not second.should_stop
    assert dict(second.overages)["tokens"] == 10

    with pytest.raises(TypeError):
        second.remaining["tokens"] = 5  # type: ignore[index]


def test_budget_meter_hard_budget_flags_stop():
    spec = BudgetSpec(
        scope="run",
        mode=BudgetMode.HARD,
        limits={"usd": 5.0},
        breach_action="error",
    )
    meter = BudgetMeter(spec)

    meter.charge(CostSnapshot.from_values(usd=3.5), label="setup")
    result = meter.charge(CostSnapshot.from_values(usd=2.25), label="breach")

    assert result.breached
    assert result.breach_kind == BudgetMode.HARD
    assert result.should_stop
    assert pytest.approx(dict(result.overages)["usd"], rel=1e-6) == 0.75
