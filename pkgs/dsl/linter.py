"""Static linter utilities for DSL policy reachability checks."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .policy import PolicyDecision, PolicyStack


@dataclass(slots=True)
class Issue:
    """Structured linter issue description."""

    level: str
    code: str
    msg: str
    path: str


def lint_unreachable_tools(flow: Mapping[str, Any]) -> list[Issue]:
    """Detect nodes or decision branches that cannot execute any allowed tool."""

    globals_cfg = flow.get("globals") or {}
    tools = globals_cfg.get("tools") or {}
    tool_sets = globals_cfg.get("tool_sets") or {}

    if not tools:
        return []

    stack = PolicyStack(tools=tools, tool_sets=tool_sets)
    graph = flow.get("graph") or {}
    issues: list[Issue] = []

    root_policy = graph.get("policy")
    if root_policy:
        stack.push(root_policy, scope="graph")

    nodes: list[Mapping[str, Any]] = list(graph.get("nodes") or [])
    nodes_by_id: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id:
            nodes_by_id[node_id] = node

    for node in nodes:
        kind = node.get("kind")
        if kind == "unit":
            decision = _evaluate_unit(node, stack)
            if not decision.allowed:
                issues.append(
                    Issue(
                        level="error",
                        code="policy.unreachable_tool",
                        msg=_format_unit_message(node, decision),
                        path=f"nodes.{node.get('id')}",
                    )
                )
        elif kind == "decision":
            issues.extend(_lint_decision(node, stack, nodes_by_id))

    if root_policy:
        stack.pop()

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluate_unit(node: Mapping[str, Any], stack: PolicyStack) -> PolicyDecision:
    scope = f"node:{node.get('id')}"
    pushed = False
    policy = node.get("policy")
    if policy:
        stack.push(policy, scope=scope)
        pushed = True

    try:
        candidates = _candidate_tools(node.get("spec") or {})
        return stack.effective_allowlist(candidates)
    finally:
        if pushed:
            stack.pop()


def _candidate_tools(spec: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    tool_ref = spec.get("tool_ref")
    if tool_ref:
        candidates.append(tool_ref)

    fallback = spec.get("fallback") or {}
    for name in fallback.get("try", []):
        if name not in candidates:
            candidates.append(name)
    return candidates


def _format_unit_message(node: Mapping[str, Any], decision: PolicyDecision) -> str:
    reasons = "; ".join(f"{tool}: {reason}" for tool, reason in decision.denied.items())
    if not reasons:
        reasons = "no tools allowed by policy stack"
    return f"Node '{node.get('id')}' cannot call any allowed tool ({reasons})."


def _lint_decision(
    node: Mapping[str, Any],
    stack: PolicyStack,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    scope = f"node:{node.get('id')}"
    pushed_node_policy = False
    if node.get("policy"):
        stack.push(node["policy"], scope=scope)
        pushed_node_policy = True

    try:
        options = list((node.get("spec") or {}).get("options") or [])
        for idx, option in enumerate(options):
            option_scope = f"decision:{node.get('id')}:{option.get('key', idx)}"
            pushed_option_policy = False
            if option.get("policy"):
                stack.push(option["policy"], scope=option_scope)
                pushed_option_policy = True

            try:
                target_id = option.get("goto")
                target = nodes_by_id.get(target_id or "")
                if not target or target.get("kind") != "unit":
                    continue

                preview = _evaluate_unit(target, stack)
                if not preview.allowed:
                    message = _format_branch_message(node, option, preview)
                    issues.append(
                        Issue(
                            level="error",
                            code="policy.unreachable_tool",
                            msg=message,
                            path=f"nodes.{node.get('id')}.options[{idx}]",
                        )
                    )
            finally:
                if pushed_option_policy:
                    stack.pop()
    finally:
        if pushed_node_policy:
            stack.pop()

    return issues


def _format_branch_message(
    node: Mapping[str, Any],
    option: Mapping[str, Any],
    decision: PolicyDecision,
) -> str:
    reasons = "; ".join(
        f"{tool}: {reason}" for tool, reason in decision.denied.items()
    ) or "no tools allowed"
    return (
        f"Branch '{option.get('key')}' of decision '{node.get('id')}' cannot execute "
        f"an allowed tool ({reasons})."
    )


__all__ = ["Issue", "lint_unreachable_tools"]

