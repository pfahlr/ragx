import math
import pathlib
import sys
from types import MappingProxyType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:  # pragma: no cover - import shim for tests
    sys.path.insert(0, str(ROOT))

from codex.code.phase3_budget_runner_r7h3.dsl import budget, costs


class DummyTraceWriter:
    def __init__(self):
        self.events = []

    def emit(self, event: str, payload):
        self.events.append((event, payload))


def test_normalize_cost_converts_seconds_and_validates_keys():
    normalized = costs.normalize_cost({"time_s": 1.5, "tokens": 12, "requests": 1})

    assert math.isclose(normalized["time_ms"], 1500.0)
    assert normalized["tokens"] == 12.0
    assert normalized["requests"] == 1.0
    # Unknown keys are preserved for forward compatibility.
    normalized_extra = costs.normalize_cost({"custom": 2})
    assert normalized_extra["custom"] == 2.0


def test_budget_manager_warns_and_returns_immutable_snapshots():
    manager = budget.BudgetManager(
        {
            "run": budget.BudgetSpec(scope="run", limits={"tokens": 100}, breach_action="warn"),
        }
    )

    first_charge = manager.commit(["run"], {"tokens": 80})
    assert first_charge.action is budget.BudgetAction.ALLOW
    assert first_charge.allowed is True
    assert isinstance(first_charge.remaining, MappingProxyType)

    preview = manager.preview(["run"], {"tokens": 30})
    assert preview.allowed is True
    assert preview.action is budget.BudgetAction.WARN
    assert preview.scope_statuses[0].breached is True
    assert isinstance(preview.scope_statuses[0].remaining, MappingProxyType)
    assert isinstance(preview.scope_statuses[0].overages, MappingProxyType)

    # Mapping proxies should be immutable
    with pytest.raises(TypeError):
        preview.scope_statuses[0].remaining["tokens"] = 1

    outcome = manager.commit(["run"], {"tokens": 30})
    assert outcome.action is budget.BudgetAction.WARN
    assert outcome.allowed is True
    assert outcome.overages["tokens"] == pytest.approx(10.0)


def test_budget_manager_stop_blocks_commit_and_reports_scope():
    manager = budget.BudgetManager(
        {
            "node:alpha": budget.BudgetSpec(scope="node", limits={"time_ms": 5}, breach_action="stop"),
        }
    )

    preview = manager.preview(["node:alpha"], {"time_ms": 6})
    assert preview.allowed is False
    assert preview.action is budget.BudgetAction.STOP
    assert preview.scope_statuses[0].scope == "node:alpha"

    with pytest.raises(budget.BudgetBreachError):
        manager.commit(["node:alpha"], {"time_ms": 6})


def test_budget_manager_hard_breach_raises_immediately():
    manager = budget.BudgetManager(
        {
            "run": budget.BudgetSpec(scope="run", limits={"tokens": 10}, breach_action="error"),
            "node:beta": budget.BudgetSpec(scope="node", limits={"tokens": 5}, breach_action="warn"),
        }
    )

    preview = manager.preview(["run", "node:beta"], {"tokens": 12})
    assert preview.allowed is False
    assert preview.action is budget.BudgetAction.ERROR
    dominant = preview.scope_statuses[0]
    assert dominant.scope == "run"
    assert dominant.breached is True

    with pytest.raises(budget.BudgetBreachError):
        manager.commit(["run", "node:beta"], {"tokens": 12})


def test_budget_manager_combines_multiple_scopes():
    manager = budget.BudgetManager(
        {
            "run": budget.BudgetSpec(scope="run", limits={"tokens": 200}, breach_action="warn"),
            "node:gamma": budget.BudgetSpec(scope="node", limits={"tokens": 50}, breach_action="stop"),
        }
    )

    manager.commit(["run", "node:gamma"], {"tokens": 30})
    preview = manager.preview(["run", "node:gamma"], {"tokens": 25})

    assert preview.allowed is False
    assert preview.action is budget.BudgetAction.STOP
    statuses = {status.scope: status for status in preview.scope_statuses}
    assert statuses["node:gamma"].breached is True
    assert statuses["run"].breached is False
