from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .policy import PolicyDenial, PolicyStack


@dataclass(frozen=True)
class Issue:
    """Structured linter issue following master spec contract."""

    level: str
    code: str
    msg: str
    path: str


class PolicyLinter:
    """Policy-aware DSL linter focusing on unreachable tools."""

    def __init__(
        self,
        *,
        tool_registry: Mapping[str, Mapping[str, Any]],
        tool_sets: Mapping[str, Sequence[str]],
    ) -> None:
        self._tool_registry = tool_registry
        self._tool_sets = tool_sets

    def find_unreachable_nodes(self, flow: Mapping[str, Any]) -> list[Issue]:
        stack = PolicyStack(tool_registry=self._tool_registry, tool_sets=self._tool_sets)
        issues: list[Issue] = []

        base_policy = self._extract_policy(flow)
        if base_policy:
            stack.push(base_policy, scope="flow")

        graph = flow.get("graph") if isinstance(flow, Mapping) else None
        nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
        if not isinstance(nodes, Iterable):
            return issues

        for index, raw_node in enumerate(nodes):
            if not isinstance(raw_node, Mapping):
                continue
            node_scope = f"node:{raw_node.get('id', index)}"
            node_policy = raw_node.get("policy") if isinstance(raw_node, Mapping) else None
            pushed = False
            if isinstance(node_policy, Mapping) and node_policy:
                stack.push(node_policy, scope=node_scope)
                pushed = True

            try:
                if raw_node.get("kind") != "unit":
                    continue
                spec = raw_node.get("spec")
                if not isinstance(spec, Mapping):
                    continue
                tool_ref = spec.get("tool_ref")
                if isinstance(tool_ref, str):
                    snapshot = stack.effective_allowlist()
                    if tool_ref not in snapshot.allowed_tools:
                        denial = snapshot.denied_tools.get(tool_ref)
                        issues.append(
                            self._build_tool_issue(
                                idx=index,
                                tool_ref=tool_ref,
                                denial=denial,
                            )
                        )

                fallback = spec.get("fallback")
                if isinstance(fallback, Mapping):
                    tries = fallback.get("try")
                    if isinstance(tries, Iterable):
                        try_tools = [tool for tool in tries if isinstance(tool, str)]
                        if try_tools:
                            snapshot = stack.effective_allowlist()
                            if not any(tool in snapshot.allowed_tools for tool in try_tools):
                                issues.append(
                                    Issue(
                                        level="error",
                                        code="policy.unreachable_fallback",
                                        msg=(
                                            "All fallback tools are blocked by policy at "
                                            f"scope '{node_scope}': {try_tools}"
                                        ),
                                        path=f"graph.nodes[{index}].spec.fallback.try",
                                    )
                                )
            finally:
                if pushed:
                    stack.pop()

        return issues

    def _build_tool_issue(
        self,
        *,
        idx: int,
        tool_ref: str,
        denial: PolicyDenial | None,
    ) -> Issue:
        scope = denial.scope if denial else "unknown"
        reason = denial.reason if denial else "blocked_by_policy"
        return Issue(
            level="error",
            code="policy.unreachable_tool",
            msg=f"Tool '{tool_ref}' is unreachable under policy scope '{scope}': {reason}",
            path=f"graph.nodes[{idx}].spec.tool_ref",
        )

    @staticmethod
    def _extract_policy(flow: Mapping[str, Any]) -> Mapping[str, Any] | None:
        for key in ("policy",):
            candidate = flow.get(key)
            if isinstance(candidate, Mapping) and candidate:
                return candidate
        globals_section = flow.get("globals") if isinstance(flow, Mapping) else None
        if isinstance(globals_section, Mapping):
            policy = globals_section.get("policy")
            if isinstance(policy, Mapping) and policy:
                return policy
        return None
