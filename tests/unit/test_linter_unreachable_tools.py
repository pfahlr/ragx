"""Static linter tests for unreachable tool nodes due to policy restrictions."""

from __future__ import annotations

from pkgs.dsl.linter import FlowLinter


def _base_flow() -> dict:
    return {
        "version": "0.1",
        "globals": {
            "tools": {
                "web_search": {"tags": ["search", "external"]},
                "internal_search": {"tags": ["search"]},
                "analysis_only": {"tags": ["internal"]},
            },
            "tool_sets": {
                "search": ["web_search", "internal_search"],
            },
            "policy": {
                "allow_tools": ["analysis_only"],
                "allow_tags": ["search"],
                "deny_tags": ["external"],
            },
        },
        "graph": {
            "nodes": [
                {
                    "id": "blocked_search",
                    "kind": "unit",
                    "spec": {
                        "type": "tool",
                        "tool_ref": "web_search",
                    },
                    "outputs": [],
                },
                {
                    "id": "allowed_search",
                    "kind": "unit",
                    "spec": {
                        "type": "tool",
                        "tool_ref": "internal_search",
                    },
                    "outputs": [],
                },
                {
                    "id": "with_fallback",
                    "kind": "unit",
                    "spec": {
                        "type": "tool",
                        "tool_ref": "web_search",
                        "fallback": {"try": ["internal_search"]},
                    },
                    "outputs": [],
                },
                {
                    "id": "override",
                    "kind": "unit",
                    "spec": {
                        "type": "tool",
                        "tool_ref": "web_search",
                    },
                    "policy": {
                        "allow_tools": ["web_search"],
                        "deny_tags": [],
                    },
                    "outputs": [],
                },
            ],
        },
    }


def test_linter_flags_nodes_without_allowed_tools() -> None:
    flow = _base_flow()
    issues = FlowLinter().find_unreachable_tools(flow)

    assert len(issues) == 1, (
        "Expected a single unreachable node, got "
        f"{[issue.path for issue in issues]}"
    )
    issue = issues[0]
    assert issue.code == "policy.unreachable_tool"
    assert issue.path == "graph.nodes[0].spec.tool_ref"
    assert "web_search" in issue.msg
    assert "denied:tag:external" in issue.msg
