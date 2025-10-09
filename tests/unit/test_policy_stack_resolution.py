from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pkgs.dsl.policy import PolicyEvent, PolicyStack, PolicyViolationError

SPEC_PATH = Path("codex/specs/ragx_master_spec.yaml")
RegistryAndSets = tuple[dict[str, Any], dict[str, list[str]]]


def _load_spec() -> dict[str, Any]:
    data = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture()
def registry_and_sets() -> RegistryAndSets:
    spec = _load_spec()
    registry = spec.get("tool_registry")
    tool_sets = spec.get("tool_sets")
    assert isinstance(registry, dict)
    assert isinstance(tool_sets, dict)
    return registry, tool_sets


def test_effective_allowlist_merges_allow_and_deny(registry_and_sets: RegistryAndSets) -> None:
    registry, tool_sets = registry_and_sets
    events: list[PolicyEvent] = []

    stack = PolicyStack(
        tool_registry=registry,
        tool_sets=tool_sets,
        event_sink=events.append,
    )

    stack.push({"allow_tools": ["safe_internal"], "deny_tags": ["external"]}, scope="flow")
    stack.push({"allow_tags": ["retrieve"]}, scope="branch:react")

    snapshot = stack.effective_allowlist()

    assert snapshot.allowed_tools == frozenset({"vector_query"})
    assert snapshot.denied_tools["gpt"].reason == "allow_tags_exclusion"
    assert snapshot.denied_tools["web_search"].reason in {
        "allow_tools_exclusion",
        "allowlist_restriction",
        "deny_tags:external",
    }

    assert [event.kind for event in events] == ["policy_push", "policy_push"]
    assert all(event.scope for event in events)


def test_enforce_emits_violation_event_and_raises(registry_and_sets: RegistryAndSets) -> None:
    registry, tool_sets = registry_and_sets
    emitted: list[PolicyEvent] = []

    stack = PolicyStack(
        tool_registry=registry,
        tool_sets=tool_sets,
        event_sink=emitted.append,
    )

    stack.push({"allow_tools": ["analysis_only"], "deny_tags": ["external"]}, scope="root")

    snapshot = stack.effective_allowlist()
    assert "web_search" not in snapshot.allowed_tools

    with pytest.raises(PolicyViolationError) as exc_info:
        stack.enforce("web_search", scope="node:do_search")

    assert "web_search" in str(exc_info.value)
    assert any(event.kind == "policy_violation" for event in emitted)
    violation = [event for event in emitted if event.kind == "policy_violation"][0]
    assert violation.detail["tool"] == "web_search"
    assert violation.detail["scope"] == "node:do_search"
    assert violation.detail["reason"] in {
        "allow_tools_exclusion",
        "allowlist_restriction",
        "deny_tags:external",
    }

    # Allowed path should not emit further violations
    assert stack.enforce("gpt", scope="node:planner", raise_on_violation=False)
