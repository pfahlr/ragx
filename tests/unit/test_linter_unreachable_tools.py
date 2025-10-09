"""Unit tests for the DSL linter rule that detects unreachable tools/branches."""
from __future__ import annotations

from pkgs.dsl.linter import lint_unreachable_tools

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


def build_flow(nodes: list[dict], policy: dict | None = None) -> dict:
    return {
        "globals": {"tools": TOOL_REGISTRY, "tool_sets": TOOL_SETS},
        "graph": {"nodes": nodes, "policy": policy or {}},
    }


def test_linter_flags_unit_with_no_allowed_tool() -> None:
    flow = build_flow(
        [
            {
                "id": "search",
                "kind": "unit",
                "spec": {"type": "tool", "tool_ref": "web_search"},
            }
        ],
        policy={"allow_tools": ["analysis_only"]},
    )

    issues = lint_unreachable_tools(flow)

    assert [issue.code for issue in issues] == ["policy.unreachable_tool"]
    assert "search" in issues[0].msg
    assert issues[0].path == "nodes.search"


def test_linter_accepts_node_with_allowed_fallback() -> None:
    flow = build_flow(
        [
            {
                "id": "search",
                "kind": "unit",
                "spec": {
                    "type": "tool",
                    "tool_ref": "web_search",
                    "fallback": {"try": ["vector_query", "gpt"]},
                },
            }
        ],
        policy={"allow_tools": ["safe_internal"]},
    )

    issues = lint_unreachable_tools(flow)

    assert issues == []


def test_linter_detects_unreachable_decision_branch() -> None:
    nodes = [
        {
            "id": "route",
            "kind": "decision",
            "spec": {
                "options": [
                    {
                        "key": "search",
                        "goto": "do_search",
                        "policy": {"allow_tools": ["retrieval_only"]},
                    },
                    {"key": "retrieve", "goto": "do_retrieve"},
                ],
                "default": "search",
            },
        },
        {
            "id": "do_search",
            "kind": "unit",
            "spec": {"type": "tool", "tool_ref": "web_search"},
        },
        {
            "id": "do_retrieve",
            "kind": "unit",
            "spec": {"type": "tool", "tool_ref": "vector_query"},
        },
    ]

    flow = build_flow(nodes, policy={"allow_tools": ["safe_internal", "analysis_only"]})

    issues = lint_unreachable_tools(flow)

    assert any(
        issue.code == "policy.unreachable_tool" and issue.path == "nodes.route.options[0]"
        for issue in issues
    ), issues
