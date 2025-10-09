"""PolicyStack allow/deny resolution behavior tests.

These tests implement the Red stage for task 07a (Policy Engine).
"""

from __future__ import annotations

import pytest

from pkgs.dsl.policy import PolicyError, PolicyStack, PolicyTraceRecorder


@pytest.fixture()
def sample_tools() -> dict[str, dict[str, list[str]]]:
    return {
        "analysis_only": {"tags": ["internal"]},
        "bing_search": {"tags": ["search", "external"]},
        "wikipedia": {"tags": ["search"]},
        "gpt-4": {"tags": ["llm"]},
    }


@pytest.fixture()
def sample_tool_sets() -> dict[str, list[str]]:
    return {"search_tools": ["bing_search", "wikipedia"]}


@pytest.fixture()
def stack(
    sample_tools: dict[str, dict[str, list[str]]],
    sample_tool_sets: dict[str, list[str]],
) -> PolicyStack:
    trace = PolicyTraceRecorder()
    policy_stack = PolicyStack(
        tools=sample_tools,
        tool_sets=sample_tool_sets,
        trace=trace,
    )
    policy_stack.push(
        {
            "allow_tools": ["analysis_only"],
            "allow_tags": ["search"],
            "deny_tags": ["external"],
        },
        scope="globals",
        source="globals.policy",
    )
    return policy_stack


def test_effective_allowlist_merges_allow_tags_and_blocks_denied_tags(
    stack: PolicyStack,
) -> None:
    resolution = stack.effective_allowlist()

    expected_allowed = {"analysis_only", "wikipedia"}
    assert resolution.allowed == expected_allowed, (
        "Resolution allowed tools mismatch.\n"
        f"Expected: {sorted(expected_allowed)}\n"
        f"Actual:   {sorted(resolution.allowed)}"
    )

    # bing_search is blocked because of the 'external' tag from the global policy deny list.
    assert resolution.reasons.get("bing_search") == "denied:tag:external"
    # The LLM tool is absent from the allow lists entirely.
    assert resolution.reasons.get("gpt-4") == "not_in_allowlist"

    events = stack.trace.events
    assert events[0].event == "policy_push"
    assert events[-1].event == "policy_resolved"
    assert events[-1].data["allowed"] == sorted(expected_allowed)


def test_nearest_policy_overrides_allow_and_deny_settings(stack: PolicyStack) -> None:
    stack.push(
        {
            "allow_tools": ["search_tools"],
            "deny_tags": [],
        },
        scope="node",
        source="graph.nodes[1].policy",
    )

    resolution = stack.effective_allowlist()
    assert resolution.allowed == {"bing_search", "wikipedia"}, (
        "Node policy should expand allowlist via tool set and clear deny tags."
    )
    assert "bing_search" not in resolution.blocked
    assert resolution.reasons.get("bing_search") is None

    stack.pop(scope="node")
    back_to_global = stack.effective_allowlist()
    assert back_to_global.allowed == {"analysis_only", "wikipedia"}


def test_invalid_tool_set_reference_raises_policy_error(
    sample_tools: dict[str, dict[str, list[str]]]
) -> None:
    stack = PolicyStack(
        tools=sample_tools,
        tool_sets={},
    )

    with pytest.raises(PolicyError, match="Unknown policy reference 'unknown-set'"):
        stack.push(
            {
                "allow_tools": ["unknown-set"],
            },
            scope="node",
            source="graph.nodes[99].policy",
        )
