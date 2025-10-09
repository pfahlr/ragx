"""Unit tests for the DSL policy stack resolution logic.

These tests were written first to drive out the PolicyStack implementation
specified in ``codex/specs/ragx_master_spec.yaml``.  The goal is to
exercise the behaviour of hierarchical allow/deny rules, tool set
resolution, candidate filtering and trace emission.
"""
from __future__ import annotations

from pkgs.dsl.policy import PolicyEvent, PolicyStack

TOOL_REGISTRY = {
    "gpt": {"type": "llm", "tags": ["analysis", "internal"]},
    "web_search": {"type": "tool", "tags": ["search", "external"]},
    "vector_query": {
        "type": "tool",
        "tags": ["retrieval", "internal", "search"],
    },
}

TOOL_SETS = {
    "analysis_only": ["gpt"],
    "safe_internal": ["vector_query", "gpt"],
    "search_only": ["web_search"],
    "retrieval_only": ["vector_query"],
}


def test_effective_allowlist_respects_nested_scopes() -> None:
    """Nested policies should intersect to produce the effective allowlist."""

    stack = PolicyStack(tools=TOOL_REGISTRY, tool_sets=TOOL_SETS)

    stack.push(
        {"allow_tools": ["analysis_only", "safe_internal"], "deny_tags": ["external"]},
        scope="graph",
    )

    decision = stack.effective_allowlist()
    assert decision.allowed == {"gpt", "vector_query"}
    assert decision.denied["web_search"].startswith("denied because tag")

    stack.push({"allow_tags": ["search"]}, scope="branch:search")

    nested = stack.effective_allowlist()
    assert nested.allowed == {"vector_query"}
    assert nested.denied["gpt"].startswith("filtered by allow_tags")

    event_types = [event.event for event in stack.events]
    assert event_types.count("policy_push") == 2
    assert event_types[-1] == "policy_allowlist"


def test_effective_allowlist_filters_candidates_and_records_denials() -> None:
    """Candidates are filtered and denial reasons are surfaced for debugging."""

    stack = PolicyStack(tools=TOOL_REGISTRY, tool_sets=TOOL_SETS)
    stack.push(
        {"allow_tools": ["analysis_only", "safe_internal"], "deny_tags": ["external"]},
        scope="graph",
    )

    stack.push({"deny_tools": ["vector_query"]}, scope="node:search")

    decision = stack.effective_allowlist(["vector_query", "gpt"])
    assert decision.allowed == {"gpt"}
    assert decision.denied["vector_query"].startswith("denied by deny_tools")

    # The trace should capture the stack depth and candidate evaluation.
    last_event = stack.events[-1]
    assert isinstance(last_event, PolicyEvent)
    assert last_event.payload["candidates"] == ["vector_query", "gpt"]
    assert last_event.payload["allowed"] == sorted(decision.allowed)


def test_pop_emits_trace_event() -> None:
    """Popping a policy should emit a policy_pop trace event."""

    stack = PolicyStack(tools=TOOL_REGISTRY, tool_sets=TOOL_SETS)
    stack.push({"allow_tools": ["analysis_only"]}, scope="graph")

    stack.pop()

    assert [event.event for event in stack.events][-1] == "policy_pop"
