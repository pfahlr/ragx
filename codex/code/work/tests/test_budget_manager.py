"""Unit tests for the canonical BudgetManager orchestration layer."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from pkgs.dsl.budget import BudgetManager, BudgetScope, BudgetSpec


@pytest.fixture()
def run_budget_manager() -> tuple[BudgetManager, BudgetScope]:
    manager = BudgetManager()
    run_scope = BudgetScope.run("run")
    run_spec = BudgetSpec.from_mapping(
        {
            "name": "run",
            "limit": {"calls": 5},
            "mode": "hard",
            "breach_action": "stop",
        }
    )
    manager.register(run_scope, run_spec)
    return manager, run_scope


def test_preview_and_commit_hard_budget_stop(run_budget_manager: tuple[BudgetManager, BudgetScope]) -> None:
    manager, run_scope = run_budget_manager

    initial_commit = manager.commit({"calls": 3}, [run_scope])
    first_charge = initial_commit.charges[0]
    assert first_charge.remaining["calls"] == 2
    assert not initial_commit.should_stop

    stop_preview = manager.preview({"calls": 3}, [run_scope])
    charge = stop_preview.charges[0]
    assert stop_preview.should_stop
    assert charge.overages["calls"] == 1
    assert charge.remaining["calls"] == 0
    assert isinstance(charge.remaining, MappingProxyType)
    assert isinstance(charge.overages, MappingProxyType)

    confirm_no_mutation = manager.preview({"calls": 1}, [run_scope])
    assert not confirm_no_mutation.should_stop
    assert confirm_no_mutation.charges[0].remaining["calls"] == 1

    post_commit = manager.commit({"calls": 1}, [run_scope])
    assert post_commit.charges[0].remaining["calls"] == 1


def test_soft_budget_warn_allows_commit() -> None:
    manager = BudgetManager()
    node_scope = BudgetScope.node("node-1")
    spec = BudgetSpec.from_mapping(
        {
            "name": "node-1",
            "limit": {"calls": 2},
            "mode": "soft",
            "breach_action": "warn",
        }
    )
    manager.register(node_scope, spec)

    first = manager.commit({"calls": 2}, [node_scope])
    assert not first.should_stop
    assert first.warnings == ()

    preview = manager.preview({"calls": 1}, [node_scope])
    assert not preview.should_stop
    assert preview.warnings == (preview.charges[0],)

    commit = manager.commit({"calls": 1}, [node_scope])
    assert commit.charges[0].overages["calls"] == 1


def test_preview_does_not_mutate_spent() -> None:
    manager = BudgetManager()
    scope = BudgetScope.loop("loop-1")
    spec = BudgetSpec.from_mapping(
        {
            "name": "loop-1",
            "limit": {"tokens": 100},
            "mode": "hard",
            "breach_action": "stop",
        }
    )
    manager.register(scope, spec)

    preview = manager.preview({"tokens": 120}, [scope])
    assert preview.should_stop

    safe_commit = manager.commit({"tokens": 20}, [scope])
    assert not safe_commit.should_stop
    assert safe_commit.charges[0].remaining["tokens"] == 80

    second_preview = manager.preview({"tokens": 90}, [scope])
    assert second_preview.should_stop
