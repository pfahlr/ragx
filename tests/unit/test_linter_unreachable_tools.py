from __future__ import annotations

from pkgs.dsl.linter import LinterIssue, lint_unreachable_tools


def _base_flow() -> dict:
    return {
        "version": 0.1,
        "globals": {
            "tools": {
                "gpt": {
                    "type": "llm",
                    "tags": ["llm", "analysis"],
                },
                "web_search": {
                    "type": "mcp",
                    "tags": ["search", "external"],
                },
            },
            "tool_sets": {
                "analysis_only": ["gpt"],
                "search_only": ["web_search"],
            },
        },
        "graph": {
            "policy": {"allow_tools": ["analysis_only"], "deny_tags": ["external"]},
            "nodes": [
                {
                    "id": "plan",
                    "kind": "unit",
                    "spec": {"type": "llm", "tool_ref": "gpt"},
                },
                {
                    "id": "search",
                    "kind": "unit",
                    "spec": {"type": "tool", "tool_ref": "web_search"},
                },
                {
                    "id": "router",
                    "kind": "decision",
                    "spec": {
                        "options": [
                            {
                                "key": "answer",
                                "goto": "plan",
                            }
                        ],
                        "default": "answer",
                    },
                },
            ],
        },
    }


def test_linter_flags_unreachable_tool_due_to_policy() -> None:
    flow = _base_flow()
    issues = lint_unreachable_tools(flow)

    assert issues
    issue = issues[0]
    assert isinstance(issue, LinterIssue)
    assert issue.code == "policy.unreachable_tool"
    assert issue.level == "error"
    assert issue.path == "graph.nodes[1]"
    assert "search" in issue.msg
    assert "web_search" in issue.msg


def test_linter_accepts_branch_policy_allowing_tool() -> None:
    flow = _base_flow()
    # Attach a branch policy that allows the search tool explicitly.
    for node in flow["graph"]["nodes"]:
        if node["id"] == "router":
            node["spec"]["options"].append(
                {
                    "key": "search",
                    "goto": "search",
                    "policy": {"allow_tools": ["search_only"]},
                }
            )

    issues = lint_unreachable_tools(flow)
    assert not issues
