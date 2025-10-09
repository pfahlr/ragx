from __future__ import annotations

import pytest

from pkgs.dsl.models import ToolDescriptor
from pkgs.dsl.policy import PolicyStack


@pytest.fixture()
def tool_registry() -> dict[str, ToolDescriptor]:
    return {
        "gpt": ToolDescriptor(name="gpt", tags=("llm", "analysis")),
        "web_search": ToolDescriptor(
            name="web_search", tags=("search", "external")
        ),
        "vector_query": ToolDescriptor(
            name="vector_query", tags=("retrieve", "internal")
        ),
    }


@pytest.fixture()
def tool_sets() -> dict[str, tuple[str, ...]]:
    return {
        "analysis_only": ("gpt",),
        "search_only": ("web_search",),
        "safe_internal": ("vector_query", "gpt"),
    }


def test_policy_resolution_respects_hierarchy(tool_registry, tool_sets) -> None:
    stack = PolicyStack(tool_sets=tool_sets)
    stack.push(
        {
            "allow_tools": ["analysis_only"],
            "deny_tags": ["external"],
        },
        scope="global",
    )

    resolution = stack.effective_allowlist(tool_registry)
    assert resolution.allowed == {"gpt"}
    assert resolution.denied == {"web_search", "vector_query"}

    search_decision = resolution.decisions["web_search"]
    assert not search_decision.allowed
    assert search_decision.source == "deny_tags"
    assert search_decision.scope == "global"
    assert "external" in (search_decision.detail or "")

    loop_policy = {"allow_tools": ["safe_internal"]}
    stack.push(loop_policy, scope="loop:react")
    loop_resolution = stack.effective_allowlist(tool_registry)

    assert loop_resolution.decisions["vector_query"].allowed
    assert loop_resolution.decisions["vector_query"].scope == "loop:react"
    assert stack.trace[-1].event == "policy_push"

    stack.push({"allow_tools": ["search_only"]}, scope="decision:search")
    branch_resolution = stack.effective_allowlist(tool_registry)

    # Branch policy should override earlier deny_tags and allow the search tool.
    assert branch_resolution.decisions["web_search"].allowed
    assert branch_resolution.decisions["web_search"].scope == "decision:search"

    stack.push({"deny_tools": ["analysis_only"]}, scope="node:block-gpt")
    deny_resolution = stack.effective_allowlist(tool_registry)

    gpt_decision = deny_resolution.decisions["gpt"]
    assert not gpt_decision.allowed
    assert gpt_decision.scope == "node:block-gpt"
    assert gpt_decision.source == "deny_tools"


def test_policy_resolution_implicit_deny_when_allowlist_present(
    tool_registry, tool_sets
) -> None:
    stack = PolicyStack(tool_sets=tool_sets)
    stack.push({"allow_tags": ["analysis"]}, scope="global")
    resolution = stack.effective_allowlist(tool_registry)

    assert resolution.decisions["gpt"].allowed
    assert not resolution.decisions["web_search"].allowed
    assert resolution.decisions["web_search"].source == "implicit_deny"

    stack.push({"allow_tags": ["search"]}, scope="decision:search")
    branch_resolution = stack.effective_allowlist(tool_registry)
    assert branch_resolution.decisions["web_search"].allowed
