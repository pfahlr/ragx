from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .models import ToolDescriptor
from .policy import PolicyDecision, PolicyStack


@dataclass(frozen=True, slots=True)
class LinterIssue:
    level: str
    code: str
    msg: str
    path: str


def lint_unreachable_tools(flow: Mapping[str, object]) -> list[LinterIssue]:
    """Detect nodes whose required tools cannot be reached under any policy path."""

    globals_section = _as_mapping(flow.get("globals"))
    tools_section = _as_mapping(globals_section.get("tools"))
    tool_sets_section = globals_section.get("tool_sets", {})
    tool_sets: dict[str, Sequence[str]] = {}
    for set_name, members in _iterate_items(tool_sets_section):
        tool_sets[set_name] = tuple(str(member) for member in _ensure_iterable(members))

    tools: dict[str, ToolDescriptor] = {}
    for name, descriptor in _iterate_items(tools_section):
        if isinstance(descriptor, ToolDescriptor):
            tools[name] = descriptor
        elif isinstance(descriptor, Mapping):
            tools[name] = ToolDescriptor.from_mapping(name, descriptor)

    graph_section = _as_mapping(flow.get("graph"))
    graph_policy = _as_mapping(graph_section.get("policy"))
    nodes = list(_ensure_iterable(graph_section.get("nodes", ())))
    control_nodes = list(_ensure_iterable(graph_section.get("control", ())))

    loop_policies = _index_loop_policies(control_nodes)
    branch_policies = _index_branch_policies(nodes)

    issues: list[LinterIssue] = []
    for index, node in enumerate(nodes):
        node_map = _as_mapping(node)
        if node_map.get("kind") != "unit":
            continue
        spec = _as_mapping(node_map.get("spec"))
        tool_ref = spec.get("tool_ref")
        if not isinstance(tool_ref, str) or tool_ref not in tools:
            continue

        contexts = _build_policy_contexts(
            node_id=str(node_map.get("id", f"<node-{index}>")),
            graph_policy=graph_policy,
            loop_policies=loop_policies,
            branch_policies=branch_policies,
            node_policy=_as_mapping(node_map.get("policy")),
            tool_sets=tool_sets,
        )

        allowed = False
        denial_reasons: list[str] = []
        for stack in contexts:
            resolution = stack.effective_allowlist(tools)
            decision = resolution.decisions.get(tool_ref)
            if decision is None:
                denial_reasons.append("tool-not-registered")
                continue
            if decision.allowed:
                allowed = True
                break
            denial_reasons.append(_format_decision(decision))

        if allowed:
            continue

        reason_text = ", ".join(denial_reasons) or "no-policy-path"
        node_id = node_map.get("id", f"nodes[{index}]")
        msg = (
            f"Node '{node_id}' tool '{tool_ref}' is unreachable under active policies"
            f" ({reason_text})."
        )
        path = f"graph.nodes[{index}]"
        issues.append(LinterIssue("error", "policy.unreachable_tool", msg, path))

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_policy_contexts(
    *,
    node_id: str,
    graph_policy: Mapping[str, object] | None,
    loop_policies: Mapping[str, list[tuple[Mapping[str, object], str]]],
    branch_policies: Mapping[str, list[tuple[Mapping[str, object], str]]],
    node_policy: Mapping[str, object] | None,
    tool_sets: Mapping[str, Sequence[str]],
) -> list[PolicyStack]:
    base_stack = PolicyStack(tool_sets=tool_sets)
    base_stack.push(graph_policy, scope="graph")
    for policy, scope in loop_policies.get(node_id, []):
        base_stack.push(policy, scope=scope)
    base_stack.push(node_policy, scope=f"node:{node_id}")

    contexts: list[PolicyStack] = [base_stack]
    for policy, scope in branch_policies.get(node_id, []):
        branch_stack = base_stack.clone()
        branch_stack.push(policy, scope=scope)
        contexts.append(branch_stack)
    return contexts


def _index_loop_policies(
    control_nodes: Sequence[object],
) -> dict[str, list[tuple[Mapping[str, object], str]]]:
    loop_map: dict[str, list[tuple[Mapping[str, object], str]]] = defaultdict(list)
    for loop in control_nodes:
        loop_map_entry = _as_mapping(loop)
        if loop_map_entry.get("kind") != "loop":
            continue
        policy = _as_mapping(loop_map_entry.get("policy"))
        scope = f"loop:{loop_map_entry.get('id', '<loop>')}"
        targets = _ensure_iterable(loop_map_entry.get("target_subgraph", ()))
        for target in targets:
            target_id = str(target)
            loop_map[target_id].append((policy, scope))
    return loop_map


def _index_branch_policies(
    nodes: Sequence[object],
) -> dict[str, list[tuple[Mapping[str, object], str]]]:
    branch_map: dict[str, list[tuple[Mapping[str, object], str]]] = defaultdict(list)
    for node in nodes:
        node_map = _as_mapping(node)
        if node_map.get("kind") != "decision":
            continue
        decision_id = str(node_map.get("id", "<decision>"))
        options = _ensure_iterable(_as_mapping(node_map.get("spec")).get("options", ()))
        for option in options:
            option_map = _as_mapping(option)
            policy = _as_mapping(option_map.get("policy"))
            goto = option_map.get("goto")
            if not goto:
                continue
            key = option_map.get("key", goto)
            scope = f"decision:{decision_id}:{key}"
            branch_map[str(goto)].append((policy, scope))
    return branch_map


def _format_decision(decision: PolicyDecision) -> str:
    detail = f"/{decision.detail}" if decision.detail else ""
    return f"{decision.source}{detail}@{decision.scope}"


def _iterate_items(obj: object | None) -> Iterable[tuple[str, object]]:
    mapping = _as_mapping(obj)
    return mapping.items()


def _ensure_iterable(obj: object | None) -> Iterable[object]:
    if obj is None:
        return ()
    if isinstance(obj, list | tuple | set | frozenset):
        return obj
    if isinstance(obj, dict):
        return obj.values()
    return (obj,)


def _as_mapping(obj: object | None) -> Mapping[str, object]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return obj
    raise TypeError(f"Expected mapping, received {type(obj)!r}")
