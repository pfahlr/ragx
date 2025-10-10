from __future__ import annotations

from pkgs.dsl.policy import PolicyError, PolicyStack, PolicyTraceRecorder

TOOL_REGISTRY = {
    "gpt": {"tags": ["llm", "analysis"]},
    "web_search": {"tags": ["search", "external"]},
    "vector_query": {"tags": ["retrieve", "internal"]},
}

TOOL_SETS = {
    "analysis_only": ["gpt"],
    "search_pair": ["web_search", "vector_query"],
}


def _stack() -> PolicyStack:
    return PolicyStack(tool_registry=TOOL_REGISTRY, tool_sets=TOOL_SETS)


def test_policy_stack_resolves_allow_and_deny() -> None:
    stack = _stack()
    stack.push(
        {
            "allow_tools": ["analysis_only"],
            "allow_tags": ["internal"],
            "deny_tags": ["external"],
        },
        scope="globals",
        source="flow.yaml",
    )

    snapshot = stack.effective_allowlist()

    assert snapshot.allowed == {"gpt", "vector_query"}
    assert snapshot.denied == {"web_search"}

    gpt_decision = snapshot.decisions["gpt"]
    assert gpt_decision.allowed is True
    assert gpt_decision.reason == "allowed"
    assert gpt_decision.allow_scope == "globals"

    web_decision = snapshot.decisions["web_search"]
    assert web_decision.allowed is False
    assert web_decision.reason == "denied:tag"
    assert web_decision.deny_scope == "globals"
    assert web_decision.matched_tags == frozenset({"external"})


def test_nearest_scope_overrides_global_directives() -> None:
    stack = _stack()
    stack.push({"deny_tools": ["gpt"]}, scope="globals", source="flow.yaml")
    stack.push({"deny_tools": []}, scope="node:answer", source="node")
    stack.push({"allow_tools": ["gpt"]}, scope="branch:answer", source="branch")

    snapshot = stack.effective_allowlist(candidates=["gpt"])
    decision = snapshot.decisions["gpt"]

    assert decision.allowed is True
    assert decision.allow_scope == "branch:answer"
    assert decision.deny_scope is None


def test_policy_stack_rejects_unknown_tool_ref() -> None:
    stack = _stack()

    try:
        stack.push({"allow_tools": ["missing_tool"]}, scope="globals")
    except PolicyError as exc:  # pragma: no cover - exercised below
        assert "unknown tool" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("PolicyError expected for unknown tool reference")


def test_trace_records_candidates_and_depth() -> None:
    recorder = PolicyTraceRecorder()
    stack = PolicyStack(tool_registry=TOOL_REGISTRY, tool_sets=TOOL_SETS, recorder=recorder)
    stack.push({"allow_tags": ["internal"]}, scope="globals")

    stack.effective_allowlist(candidates=["vector_query"])
    stack.pop(scope="globals")

    events = recorder.events
    assert [event.event for event in events] == ["policy_push", "policy_resolved", "policy_pop"]
    resolved_event = events[1]
    assert resolved_event.data["allowed"] == ["vector_query"]
    assert resolved_event.data["stack_depth"] == 1

