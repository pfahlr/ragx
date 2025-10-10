from __future__ import annotations

import pytest

from pkgs.dsl import (
    PolicyStack,
    PolicyTraceEvent,
    PolicyViolationError,
    ToolDescriptor,
)


def _build_registry() -> dict[str, ToolDescriptor]:
    return {
        "gpt": ToolDescriptor.from_spec("gpt", tags=["analysis", "internal"]),
        "vector_query": ToolDescriptor.from_spec("vector_query", tags=["retrieval", "internal"]),
        "web_search": ToolDescriptor.from_spec("web_search", tags=["external", "search"]),
    }


def test_effective_allowlist_prefers_nearest_scope() -> None:
    events: list[PolicyTraceEvent] = []
    stack = PolicyStack(
        tool_registry=_build_registry(),
        tool_sets={"analysis_only": ["gpt"], "safe_internal": ["analysis_only", "vector_query"]},
        event_sink=events.append,
    )

    stack.push(
        "global",
        {
            "allow_tools": ["safe_internal"],
            "deny_tags": ["external"],
        },
    )
    stack.push(
        "node:search",
        {
            "allow_tools": ["web_search"],
            "deny_tools": ["gpt"],
        },
    )

    resolution = stack.effective_allowlist()

    assert resolution.decisions["vector_query"].allowed is True
    assert resolution.decisions["vector_query"].scope == "global"
    assert resolution.decisions["gpt"].allowed is False
    assert resolution.decisions["gpt"].scope == "node:search"
    assert resolution.decisions["web_search"].allowed is True
    assert resolution.decisions["web_search"].scope == "node:search"

    assert events[-1].event == "policy_resolved"
    assert set(events[-1].data["allowed"]) == {"vector_query", "web_search"}
    assert "gpt" in resolution.denied_tools


def test_push_unknown_tool_set_raises() -> None:
    stack = PolicyStack(tool_registry=_build_registry(), tool_sets={})
    with pytest.raises(ValueError, match="unknown tool or tool_set 'analysis_only'"):
        stack.push("global", {"allow_tools": ["analysis_only"]})


def test_tool_set_cycle_detection() -> None:
    tool_sets = {"cycle_a": ["cycle_b"], "cycle_b": ["cycle_a"]}
    with pytest.raises(ValueError, match="cycle detected"):
        PolicyStack(tool_registry=_build_registry(), tool_sets=tool_sets)


def test_enforce_emits_violation_event_and_error() -> None:
    events: list[PolicyTraceEvent] = []
    stack = PolicyStack(tool_registry=_build_registry(), tool_sets={}, event_sink=events.append)
    stack.push("global", {"allow_tools": ["vector_query"]})

    with pytest.raises(PolicyViolationError) as excinfo:
        stack.enforce("gpt")

    assert excinfo.value.denial.decision.tool == "gpt"
    assert events[-1].event == "violation"
    assert events[-1].data["tool"] == "gpt"


def test_enforce_returns_snapshot_when_allowed() -> None:
    stack = PolicyStack(tool_registry=_build_registry(), tool_sets={})
    stack.push("global", {"allow_tools": ["gpt"]})

    snapshot = stack.enforce("gpt")

    assert snapshot.stack_depth == stack.stack_depth
    assert snapshot.resolution.decisions["gpt"].allowed is True

