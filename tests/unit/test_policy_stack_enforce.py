from __future__ import annotations

import pytest

from pkgs.dsl.policy import (
    PolicyError,
    PolicyStack,
    PolicyTraceEvent,
    PolicyViolationError,
)


def _build_stack() -> PolicyStack:
    registry = {
        "gpt": {"tags": ["llm", "analysis"]},
        "web_search": {"tags": ["search", "external"]},
        "vector_query": {"tags": ["retrieve", "internal"]},
    }
    tool_sets = {
        "analysis_only": ["gpt"],
        "safe_internal": ["vector_query", "analysis_only"],
        "search_only": ["web_search"],
    }
    return PolicyStack(registry, tool_sets)


def test_effective_allowlist_honors_nearest_scope_and_traces() -> None:
    stack = _build_stack()
    stack.push("globals", {"allow_tools": ["safe_internal"], "deny_tags": ["external"]})
    stack.push("decision:search", {"allow_tags": ["search"]})

    snapshot = stack.effective_allowlist(candidates=["gpt", "web_search", "vector_query"])

    assert set(snapshot.allowed) == {"gpt", "web_search", "vector_query"}
    assert snapshot.decisions["web_search"].allowed is True
    assert snapshot.decisions["web_search"].deciding_scope == "decision:search"
    assert "allow_tags" in snapshot.decisions["web_search"].reasons
    assert snapshot.decisions["gpt"].deciding_scope == "globals"
    assert snapshot.decisions["vector_query"].allowed is True

    resolved_event = stack.recorder.events[-1]
    assert isinstance(resolved_event, PolicyTraceEvent)
    assert resolved_event.event == "policy_resolved"
    assert resolved_event.data["candidates"] == ["gpt", "vector_query", "web_search"]
    assert sorted(resolved_event.data["allowed"]) == ["gpt", "vector_query", "web_search"]
    assert resolved_event.data["stack_depth"] == 2


def test_enforce_blocks_not_in_allowlist_and_emits_violation() -> None:
    stack = _build_stack()
    stack.push("globals", {"allow_tools": ["analysis_only"], "deny_tags": ["external"]})

    with pytest.raises(PolicyViolationError) as excinfo:
        stack.enforce("web_search")

    error = excinfo.value
    assert error.denial.tool == "web_search"
    assert "deny_tags" in error.denial.reasons
    assert error.denial.deciding_scope == "globals"

    violation_event = stack.recorder.events[-1]
    assert violation_event.event == "policy_violation"
    assert violation_event.data["tool"] == "web_search"
    assert "deny_tags" in violation_event.data["reasons"]

    # The policy_resolved event should precede the violation event.
    assert stack.recorder.events[-2].event == "policy_resolved"


def test_pop_scope_validation() -> None:
    stack = _build_stack()
    stack.push("globals", None)
    stack.push("node:a", {"deny_tools": ["search_only"]})

    with pytest.raises(PolicyError):
        stack.pop("globals")

    stack.pop("node:a")
    stack.pop("globals")

    events = [event.event for event in stack.recorder.events]
    assert events.count("policy_pop") == 2

