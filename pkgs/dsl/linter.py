"""Static linter checks for RAGX DSL flows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .policy import PolicyResolution, PolicyStack


@dataclass(slots=True)
class Issue:
    """Structured linter issue."""

    level: str
    code: str
    msg: str
    path: str


class FlowLinter:
    """Collection of linter rules for DSL flows."""

    def find_unreachable_tools(self, flow: Mapping[str, object]) -> list[Issue]:
        globals_section = flow.get("globals", {}) if isinstance(flow, Mapping) else {}
        if isinstance(globals_section, Mapping):
            tools = globals_section.get("tools", {})
            tool_sets = globals_section.get("tool_sets", {})
            global_policy = globals_section.get("policy")
        else:
            tools = {}
            tool_sets = {}
            global_policy = None

        policy_stack = PolicyStack(tools=tools, tool_sets=tool_sets)

        pushed_global = False
        if isinstance(global_policy, Mapping):
            policy_stack.push(global_policy, scope="globals", source="globals.policy")
            pushed_global = True

        issues: list[Issue] = []
        graph = flow.get("graph", {}) if isinstance(flow, Mapping) else {}
        if isinstance(graph, Mapping):
            nodes: Sequence[Mapping[str, object]] = graph.get("nodes", [])
        else:
            nodes = []

        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                continue
            scope = f"graph.nodes[{index}]"
            node_policy = node.get("policy") if isinstance(node, Mapping) else None
            pushed_node = False
            if isinstance(node_policy, Mapping):
                policy_stack.push(node_policy, scope=scope, source=f"{scope}.policy")
                pushed_node = True

            resolution = policy_stack.effective_allowlist()
            issue = self._evaluate_node(node, index, resolution)
            if issue is not None:
                issues.append(issue)

            if pushed_node:
                policy_stack.pop(scope=scope)

        if pushed_global:
            policy_stack.pop(scope="globals")

        return issues

    def _evaluate_node(
        self,
        node: Mapping[str, object],
        index: int,
        resolution: PolicyResolution,
    ) -> Issue | None:
        kind = node.get("kind") if isinstance(node, Mapping) else None
        if kind != "unit":
            return None

        spec = node.get("spec") if isinstance(node, Mapping) else None
        if not isinstance(spec, Mapping):
            return None

        unit_type = spec.get("type")
        if unit_type not in {"tool", "llm"}:
            return None

        tool_ref = spec.get("tool_ref")
        if not isinstance(tool_ref, str):
            return None

        candidates: list[str] = [tool_ref]
        fallback = spec.get("fallback") if isinstance(spec, Mapping) else None
        if isinstance(fallback, Mapping):
            tries = fallback.get("try")
            if isinstance(tries, Iterable) and not isinstance(tries, str | bytes):
                for entry in tries:
                    if isinstance(entry, str):
                        candidates.append(entry)

        reachable = [tool for tool in candidates if resolution.is_allowed(tool)]
        if reachable:
            return None

        reasons: dict[str, str] = {}
        for tool in candidates:
            if tool in resolution.reasons:
                reasons[tool] = resolution.reasons[tool]
            else:
                reasons[tool] = "unknown_tool"

        node_id = node.get("id", f"graph.nodes[{index}]")
        path = f"graph.nodes[{index}].spec.tool_ref"
        msg = (
            f"Node '{node_id}' has no allowed tools under policy. Candidates: {reasons}."
        )
        return Issue(level="error", code="policy.unreachable_tool", msg=msg, path=path)
