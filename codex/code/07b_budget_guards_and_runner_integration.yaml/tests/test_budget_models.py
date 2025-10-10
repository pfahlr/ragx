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
    # Ensure we reload a fresh module copy for isolation between test modules.
    return load_module()


def test_budget_spec_normalizes_time_units(budget_module):
    BudgetSpec = budget_module.BudgetSpec

    spec = BudgetSpec(
        scope="run",
        limits={"time_s": 2, "time_ms": 500, "tokens": 50},
        breach_action="warn",
    )

    assert spec.limits["time_ms"] == 2500
    assert spec.limits["tokens"] == 50
    # Internal representation must be immutable
    with pytest.raises(TypeError):
        spec.limits["time_ms"] = 0  # type: ignore[index]


def test_budget_spec_rejects_negative_limits(budget_module):
    BudgetSpec = budget_module.BudgetSpec

    with pytest.raises(ValueError):
        BudgetSpec(scope="node", limits={"time_ms": -1})


def test_cost_snapshot_arithmetic_helpers(budget_module):
    CostSnapshot = budget_module.CostSnapshot

    base = CostSnapshot.from_mapping({"time_ms": 100, "tokens": 5})
    delta = CostSnapshot.from_mapping({"time_ms": 25, "tokens": 3})

    combined = base + delta
    assert combined.metrics["time_ms"] == 125
    assert combined.metrics["tokens"] == 8

    remaining = combined.clamped_non_negative(delta * -1)
    assert remaining.metrics["time_ms"] == 100
    assert remaining.metrics["tokens"] == 5


def test_cost_snapshot_serialization(budget_module):
    CostSnapshot = budget_module.CostSnapshot

    snap = CostSnapshot.from_mapping({"time_ms": 123.4, "tokens": 7})
    payload = snap.to_payload()

    assert payload == {"time_ms": 123.4, "tokens": 7}
    # Ensure mapping is a shallow copy and external mutation does not affect snapshot
    payload["tokens"] = 99
    assert snap.metrics["tokens"] == 7
