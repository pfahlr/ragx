from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pkgs.dsl.linter import Issue, PolicyLinter

SPEC_PATH = Path("codex/specs/ragx_master_spec.yaml")


def _load_spec() -> dict[str, Any]:
    data = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _build_flow(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "0.1",
        "globals": {"policy": policy},
        "graph": {
            "nodes": [
                {
                    "id": "planner",
                    "kind": "unit",
                    "spec": {"type": "llm", "tool_ref": "gpt"},
                },
                {
                    "id": "do_search",
                    "kind": "unit",
                    "spec": {
                        "type": "tool",
                        "tool_ref": "web_search",
                        "fallback": {"try": ["vector_query"]},
                    },
                },
                {
                    "id": "safety_audit",
                    "kind": "unit",
                    "policy": {"allow_tags": ["quality"]},
                    "spec": {"type": "tool", "tool_ref": "citations_audit"},
                },
            ]
        },
    }


def test_linter_flags_unreachable_tools_due_to_policy() -> None:
    spec = _load_spec()
    registry = spec.get("tool_registry")
    tool_sets = spec.get("tool_sets")
    assert isinstance(registry, dict)
    assert isinstance(tool_sets, dict)

    policy = {
        "allow_tools": ["safe_internal", "publishing_set"],
        "deny_tags": ["external"],
    }
    linter = PolicyLinter(tool_registry=registry, tool_sets=tool_sets)

    flow = _build_flow(policy)
    issues = linter.find_unreachable_nodes(flow)

    assert issues, "Expected at least one policy issue"
    assert all(isinstance(issue, Issue) for issue in issues)

    unreachable = [issue for issue in issues if issue.code == "policy.unreachable_tool"]
    assert len(unreachable) == 1
    issue = unreachable[0]
    assert issue.path == "graph.nodes[1].spec.tool_ref"
    assert "web_search" in issue.msg
    assert any(
        marker in issue.msg for marker in ("deny_tags", "allow_tools_exclusion", "allowlist")
    )

    # Safety audit node should become reachable with its own policy override
    override_flow = _build_flow(policy)
    issues_override = linter.find_unreachable_nodes(override_flow)
    assert any(issue.path == "graph.nodes[2].spec.tool_ref" for issue in issues_override) is False

    # Removing deny should clear all issues
    permissive_flow = _build_flow({"allow_tags": ["search", "retrieve", "quality", "analysis"]})
    assert not linter.find_unreachable_nodes(permissive_flow)
