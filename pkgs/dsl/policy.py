"""Policy stack implementation for the RAGX DSL."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .models import (
    PolicyDecision,
    PolicyDenial,
    PolicyResolution,
    PolicySnapshot,
    PolicyTraceEvent,
    PolicyTraceRecorder,
    ToolDescriptor,
)


class PolicyViolationError(RuntimeError):
    """Raised when policy enforcement blocks a tool invocation."""

    def __init__(self, denial: PolicyDenial) -> None:
        super().__init__(f"Tool '{denial.tool}' blocked by policy: {denial.decision.reason}")
        self.denial = denial


def emit_policy_event(
    recorder: PolicyTraceRecorder,
    sink: Callable[[PolicyTraceEvent], None] | None,
    *,
    event: str,
    scope: str,
    payload: Mapping[str, object],
) -> None:
    record = PolicyTraceEvent(event=event, scope=scope, data=MappingProxyType(dict(payload)))
    recorder.record(record)
    if sink is not None:
        sink(record)


@dataclass(slots=True)
class _PolicyDirectives:
    scope: str
    allow_tools: frozenset[str]
    deny_tools: frozenset[str]
    allow_tags: frozenset[str]
    deny_tags: frozenset[str]
    raw_policy: Mapping[str, Any]

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "allow_tools": tuple(sorted(self.allow_tools)),
                "deny_tools": tuple(sorted(self.deny_tools)),
                "allow_tags": tuple(sorted(self.allow_tags)),
                "deny_tags": tuple(sorted(self.deny_tags)),
            }
        )


class PolicyStack:
    """LIFO policy stack supporting scoped allow/deny directives."""

    def __init__(
        self,
        *,
        tool_registry: Mapping[str, ToolDescriptor],
        tool_sets: Mapping[str, Sequence[str]] | None = None,
        event_sink: Callable[[PolicyTraceEvent], None] | None = None,
    ) -> None:
        if not tool_registry:
            raise ValueError("tool_registry must not be empty")

        self._registry = {name: descriptor for name, descriptor in tool_registry.items()}
        self._tool_sets = {name: tuple(values) for name, values in (tool_sets or {}).items()}
        self._stack: list[_PolicyDirectives] = []
        self._recorder = PolicyTraceRecorder()
        self._event_sink = event_sink

        # Validate tool set contents upfront.
        for set_name in self._tool_sets:
            self._expand_tool_set(set_name, set())

    @property
    def stack_depth(self) -> int:
        return len(self._stack)

    @property
    def recorder(self) -> PolicyTraceRecorder:
        return self._recorder

    def push(self, scope: str, policy: Mapping[str, Any] | None = None) -> None:
        normalized = self._normalize_policy(scope, policy or {})
        self._stack.append(normalized)
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="push",
            scope=scope,
            payload={"directives": normalized.payload, "stack_depth": self.stack_depth},
        )

    def pop(self, scope: str | None = None) -> None:
        if not self._stack:
            raise RuntimeError("policy stack underflow")
        top = self._stack[-1]
        if scope is not None and top.scope != scope:
            raise RuntimeError(f"attempted to pop scope '{scope}' but top is '{top.scope}'")
        self._stack.pop()
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="pop",
            scope=top.scope,
            payload={"stack_depth": self.stack_depth},
        )

    def effective_allowlist(self, *, tools: Iterable[str] | None = None) -> PolicyResolution:
        candidates = self._candidate_tools(tools)
        decisions: dict[str, PolicyDecision] = {}
        for tool_name in sorted(candidates):
            descriptor = self._registry[tool_name]
            decisions[tool_name] = self._resolve_tool(descriptor)

        resolution = PolicyResolution(decisions)
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_resolved",
            scope=self._stack[-1].scope if self._stack else "<root>",
            payload={
                "allowed": tuple(sorted(resolution.allowed_tools)),
                "denied": tuple(sorted(resolution.denied_tools)),
                "stack_depth": self.stack_depth,
            },
        )
        return resolution

    def enforce(self, tool: str, *, raise_error: bool = True) -> PolicySnapshot:
        if tool not in self._registry:
            raise ValueError(f"unknown tool '{tool}'")

        resolution = self.effective_allowlist(tools=[tool])
        decision = resolution.decisions[tool]
        snapshot = PolicySnapshot(stack_depth=self.stack_depth, resolution=resolution)

        if decision.allowed:
            return snapshot

        denial = PolicyDenial(tool=tool, decision=decision, resolution=resolution)
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="violation",
            scope=decision.scope or "<unspecified>",
            payload={
                "tool": tool,
                "reason": decision.reason,
                "stack_depth": self.stack_depth,
            },
        )
        if raise_error:
            raise PolicyViolationError(denial)
        return snapshot

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _candidate_tools(self, tools: Iterable[str] | None) -> Sequence[str]:
        if tools is None:
            return tuple(sorted(self._registry))
        normalized = []
        for tool in tools:
            if tool not in self._registry:
                raise ValueError(f"unknown tool '{tool}'")
            normalized.append(tool)
        return tuple(normalized)

    def _resolve_tool(self, descriptor: ToolDescriptor) -> PolicyDecision:
        for directives in reversed(self._stack):
            if descriptor.name in directives.deny_tools:
                return PolicyDecision(
                    tool=descriptor.name,
                    allowed=False,
                    scope=directives.scope,
                    reason="deny_tools",
                )
            if directives.deny_tags & descriptor.tags:
                return PolicyDecision(
                    tool=descriptor.name,
                    allowed=False,
                    scope=directives.scope,
                    reason="deny_tags",
                )
            if descriptor.name in directives.allow_tools:
                return PolicyDecision(
                    tool=descriptor.name,
                    allowed=True,
                    scope=directives.scope,
                    reason="allow_tools",
                )
            if directives.allow_tags & descriptor.tags:
                return PolicyDecision(
                    tool=descriptor.name,
                    allowed=True,
                    scope=directives.scope,
                    reason="allow_tags",
                )

        return PolicyDecision(
            tool=descriptor.name,
            allowed=False,
            scope=None,
            reason="implicit_deny",
        )

    def _normalize_policy(self, scope: str, policy: Mapping[str, Any]) -> _PolicyDirectives:
        allow_tools = self._expand_directive(policy.get("allow_tools", ()))
        deny_tools = self._expand_directive(policy.get("deny_tools", ()))
        allow_tags = frozenset(policy.get("allow_tags", ()))
        deny_tags = frozenset(policy.get("deny_tags", ()))

        unknown_tags = [tag for tag in allow_tags | deny_tags if not isinstance(tag, str)]
        if unknown_tags:
            raise TypeError("tags must be strings")

        return _PolicyDirectives(
            scope=scope,
            allow_tools=allow_tools,
            deny_tools=deny_tools,
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            raw_policy=MappingProxyType(dict(policy)),
        )

    def _expand_directive(self, values: Iterable[str]) -> frozenset[str]:
        expanded: set[str] = set()
        for value in values:
            expanded.update(self._expand_value(value, set()))
        return frozenset(expanded)

    def _expand_value(self, value: str, seen: set[str]) -> set[str]:
        if value in self._registry:
            return {value}
        if value in self._tool_sets:
            return set(self._expand_tool_set(value, seen))
        raise ValueError(f"unknown tool or tool_set '{value}'")

    def _expand_tool_set(self, set_name: str, seen: set[str]) -> frozenset[str]:
        if set_name in seen:
            raise ValueError(f"cycle detected in tool_sets expansion: {set_name}")
        seen.add(set_name)
        tools: set[str] = set()
        for entry in self._tool_sets.get(set_name, ()):  # type: ignore[arg-type]
            if entry in self._registry:
                tools.add(entry)
            elif entry in self._tool_sets:
                tools.update(self._expand_tool_set(entry, seen))
            else:
                raise ValueError(f"unknown tool or tool_set '{entry}' in set '{set_name}'")
        seen.remove(set_name)
        return frozenset(tools)

