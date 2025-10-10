from __future__ import annotations

import pytest

from pkgs.dsl.policy import PolicyError, PolicyStack, PolicyTraceRecorder

TOOLS = {
    "analysis_llm": {"tags": ["analysis", "llm"]},
    "search_api": {"tags": ["search", "external"]},
    "vector_query": {"tags": ["retrieval", "internal"]},
    "metrics": {"tags": ["internal"]},
}

TOOL_SETS = {
    "analysis_only": ["analysis_llm"],
    "search_bundle": ["search_api", "vector_query"],
    "hybrid": ["analysis_only", "metrics"],
}


def _stack(trace: PolicyTraceRecorder | None = None) -> PolicyStack:
    return PolicyStack(tools=TOOLS, tool_sets=TOOL_SETS, trace=trace)


def test_policy_resolution_respects_nearest_scope() -> None:
    trace = PolicyTraceRecorder()
    stack = _stack(trace)
    stack.push({"allow_tools": ["analysis_only"]}, scope="global")
    resolution = stack.effective_allowlist()
    assert resolution.allowed == frozenset({"analysis_llm"})
    assert resolution.denied["search_api"] == ("not_in_allowlist",)

    stack.push({"allow_tools": ["search_bundle"]}, scope="branch")
    resolution = stack.effective_allowlist()
    assert resolution.allowed == frozenset({"search_api", "vector_query"})
    search_decision = resolution.decisions["search_api"]
    assert search_decision.allowed is True
    assert search_decision.granted_by == "branch"
    assert search_decision.reasons == ("allow:tool:search_api",)

    stack.push({"deny_tags": ["external"]}, scope="node")
    resolution = stack.effective_allowlist()
    assert resolution.allowed == frozenset({"vector_query"})
    denied = resolution.denied["search_api"]
    assert "deny:tag:external" in denied
    assert trace.events[-1].event == "policy_resolved"


def test_policy_resolution_includes_trace_payload() -> None:
    trace = PolicyTraceRecorder()
    stack = _stack(trace)
    stack.push(None, scope="root")
    stack.effective_allowlist()
    event = trace.events[-1]
    assert event.event == "policy_resolved"
    assert event.data["allowed"] == sorted(TOOLS)
    assert event.data["stack_depth"] == 1


def test_policy_stack_validates_pop_scope() -> None:
    stack = _stack()
    stack.push({}, scope="outer")
    stack.push({}, scope="inner")
    with pytest.raises(PolicyError, match="scope mismatch"):
        stack.pop(expected_scope="outer")
    stack.pop(expected_scope="inner")
    stack.pop(expected_scope="outer")


def test_policy_stack_rejects_unknown_set() -> None:
    stack = _stack()
    with pytest.raises(PolicyError, match="unknown tool or tool_set"):
        stack.push({"allow_tools": ["missing"]}, scope="root")


def test_policy_stack_detects_cycles() -> None:
    cyclic_sets = {"a": ["b"], "b": ["a"]}
    with pytest.raises(PolicyError, match="cycle detected"):
        PolicyStack(tools=TOOLS, tool_sets=cyclic_sets)
