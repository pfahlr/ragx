"""Policy engine primitives for the DSL runner.

This module implements the ``PolicyStack`` class described in
``codex/specs/ragx_master_spec.yaml``.  The stack maintains a sequence of
policy scopes (graph → decisions → nodes) and can derive an effective
allowlist of tools respecting allow/deny directives and tags.  Trace
events are captured in-memory to support deterministic testing and future
integration with the runner's structured logging.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

PolicyDict = Mapping[str, Any]


@dataclass(slots=True)
class PolicyEvent:
    """Structured trace emitted by ``PolicyStack`` operations."""

    event: str
    scope: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyDecision:
    """Result of evaluating the effective allowlist for a scope."""

    allowed: set[str]
    denied: dict[str, str]
    candidates: list[str]
    stack_depth: int

    def __post_init__(self) -> None:  # pragma: no cover - defensive guard
        # Ensure deterministic iteration for payload comparisons in tests.
        self.candidates = list(self.candidates)


@dataclass(slots=True)
class _PolicyFrame:
    """Internal representation of a policy pushed on the stack."""

    scope: str
    policy: dict[str, Any]


class PolicyStack:
    """Maintain hierarchical DSL policies and compute effective allowlists."""

    def __init__(
        self,
        *,
        tools: Mapping[str, Mapping[str, Any]],
        tool_sets: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if not tools:
            raise ValueError("PolicyStack requires at least one registered tool")

        self._tools: dict[str, Mapping[str, Any]] = dict(tools)
        self._tool_sets: dict[str, Sequence[str]] = dict(tool_sets or {})
        self.stack: list[_PolicyFrame] = []
        self.events: list[PolicyEvent] = []

    # ------------------------------------------------------------------
    # Stack manipulation
    # ------------------------------------------------------------------
    def push(self, policy: PolicyDict | None, *, scope: str) -> None:
        """Push a new policy frame onto the stack and emit a trace event."""

        normalized = self._normalize_policy(policy)
        self.stack.append(_PolicyFrame(scope=scope, policy=normalized))
        self._emit("policy_push", scope, {"policy": normalized})

    def pop(self) -> dict[str, Any]:
        """Pop the most recent policy frame and return it."""

        if not self.stack:
            raise RuntimeError("PolicyStack.pop() called on empty stack")

        frame = self.stack.pop()
        self._emit("policy_pop", frame.scope, {"policy": frame.policy})
        return frame.policy

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def effective_allowlist(self, candidates: Sequence[str] | None = None) -> PolicyDecision:
        """Compute the effective allowlist for the current stack state.

        Args:
            candidates: Optional iterable of tool names to filter against the
                allowlist.  When ``None`` all registered tools are
                considered.

        Returns:
            ``PolicyDecision`` capturing allowed tool names, denial reasons
            for candidates and metadata useful for trace payloads.
        """

        all_tools = sorted(self._tools)
        state: dict[str, bool] = {tool: True for tool in all_tools}
        denial_reasons: dict[str, tuple[int, str]] = {}

        for frame in self.stack:
            policy = frame.policy
            scope = frame.scope

            self._apply_allow_tools(policy, scope, state, denial_reasons)
            self._apply_allow_tags(policy, scope, state, denial_reasons)
            self._apply_deny_tools(policy, scope, state, denial_reasons)
            self._apply_deny_tags(policy, scope, state, denial_reasons)

        candidate_list = list(candidates) if candidates is not None else list(all_tools)
        allowed_set = {tool for tool, is_allowed in state.items() if is_allowed}
        allowed_candidates = allowed_set & set(candidate_list)
        denied_candidates: dict[str, str] = {}

        for tool in candidate_list:
            if tool not in self._tools:
                denied_candidates[tool] = "unknown tool"
            elif tool not in allowed_set:
                reason = denial_reasons.get(tool)
                denied_candidates[tool] = reason[1] if reason else "blocked by upstream policy"

        decision = PolicyDecision(
            allowed=allowed_candidates,
            denied=denied_candidates,
            candidates=candidate_list,
            stack_depth=len(self.stack),
        )

        self._emit(
            "policy_allowlist",
            self.stack[-1].scope if self.stack else "global",
            {
                "candidates": candidate_list,
                "allowed": sorted(allowed_candidates),
                "denied": denied_candidates,
                "stack_depth": len(self.stack),
            },
        )
        return decision

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------
    def _normalize_policy(self, policy: PolicyDict | None) -> dict[str, Any]:
        if policy is None:
            return {"allow_tools": [], "deny_tools": [], "allow_tags": [], "deny_tags": []}

        normalized: dict[str, Any] = {
            "allow_tools": list(policy.get("allow_tools", []) or []),
            "deny_tools": list(policy.get("deny_tools", []) or []),
            "allow_tags": list(policy.get("allow_tags", []) or []),
            "deny_tags": list(policy.get("deny_tags", []) or []),
        }
        return normalized

    def _resolve_tool_refs(self, names: Sequence[str], scope: str) -> set[str]:
        resolved: set[str] = set()
        for name in names:
            if name in self._tool_sets:
                resolved.update(self._tool_sets[name])
            else:
                if name not in self._tools:
                    raise KeyError(
                        "Unknown tool or tool_set "
                        f"'{name}' referenced in policy scope '{scope}'"
                    )
                resolved.add(name)
        return resolved

    def _apply_allow_tools(
        self,
        policy: Mapping[str, Any],
        scope: str,
        state: MutableMapping[str, bool],
        denial_reasons: MutableMapping[str, tuple[int, str]],
    ) -> None:
        allow_tools = policy.get("allow_tools") or []
        if not allow_tools:
            return

        resolved = self._resolve_tool_refs(allow_tools, scope)
        for tool in state:
            if tool not in resolved:
                state[tool] = False
                self._record_reason(
                    tool,
                    priority=1,
                    message=f"filtered by allow_tools {allow_tools} in scope '{scope}'",
                    denial_reasons=denial_reasons,
                )

    def _apply_allow_tags(
        self,
        policy: Mapping[str, Any],
        scope: str,
        state: MutableMapping[str, bool],
        denial_reasons: MutableMapping[str, tuple[int, str]],
    ) -> None:
        allow_tags = set(policy.get("allow_tags") or [])
        if not allow_tags:
            return

        for tool in state:
            tags = set(self._tools[tool].get("tags", []))
            if allow_tags & tags:
                continue
            state[tool] = False
            self._record_reason(
                tool,
                priority=1,
                message=f"filtered by allow_tags {sorted(allow_tags)} in scope '{scope}'",
                denial_reasons=denial_reasons,
            )

    def _apply_deny_tools(
        self,
        policy: Mapping[str, Any],
        scope: str,
        state: MutableMapping[str, bool],
        denial_reasons: MutableMapping[str, tuple[int, str]],
    ) -> None:
        deny_tools = policy.get("deny_tools") or []
        if not deny_tools:
            return

        resolved = self._resolve_tool_refs(deny_tools, scope)
        for tool in resolved:
            state[tool] = False
            self._record_reason(
                tool,
                priority=2,
                message=f"denied by deny_tools {deny_tools} in scope '{scope}'",
                denial_reasons=denial_reasons,
            )

    def _apply_deny_tags(
        self,
        policy: Mapping[str, Any],
        scope: str,
        state: MutableMapping[str, bool],
        denial_reasons: MutableMapping[str, tuple[int, str]],
    ) -> None:
        deny_tags = set(policy.get("deny_tags") or [])
        if not deny_tags:
            return

        for tool in state:
            tags = set(self._tools[tool].get("tags", []))
            if not (deny_tags & tags):
                continue
            state[tool] = False
            self._record_reason(
                tool,
                priority=2,
                message=f"denied because tag(s) {sorted(deny_tags)} are blocked in scope '{scope}'",
                denial_reasons=denial_reasons,
            )

    def _record_reason(
        self,
        tool: str,
        *,
        priority: int,
        message: str,
        denial_reasons: MutableMapping[str, tuple[int, str]],
    ) -> None:
        existing = denial_reasons.get(tool)
        if existing is None or priority >= existing[0]:
            denial_reasons[tool] = (priority, message)

    # ------------------------------------------------------------------
    def _emit(self, event: str, scope: str, payload: Mapping[str, Any]) -> None:
        self.events.append(PolicyEvent(event=event, scope=scope, payload=dict(payload)))


__all__ = ["PolicyDecision", "PolicyEvent", "PolicyStack"]

