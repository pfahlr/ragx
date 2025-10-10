from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pkgs.dsl.models import (
    PolicyDecision,
    PolicyDenial,
    PolicySnapshot,
    PolicyTraceEvent,
    PolicyViolationError,
    ToolDescriptor,
)


class PolicyError(ValueError):
    """Raised when policies reference unknown tools or violate stack rules."""


@dataclass(frozen=True)
class _PolicyFrame:
    scope: str
    source: str | None
    allow_tools: frozenset[str] | None
    allow_tags: frozenset[str] | None
    deny_tools: frozenset[str] | None
    deny_tags: frozenset[str] | None
    raw: Mapping[str, Sequence[str]]


class PolicyTraceRecorder:
    """In-memory collector for policy trace events."""

    def __init__(self) -> None:
        self._events: list[PolicyTraceEvent] = []

    def record(self, event: PolicyTraceEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[PolicyTraceEvent, ...]:
        return tuple(self._events)


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


class PolicyStack:
    """Hierarchical allow/deny resolver for DSL tool policies."""

    def __init__(
        self,
        *,
        tool_registry: Mapping[str, Mapping[str, Any]],
        tool_sets: Mapping[str, Sequence[str]] | None = None,
        recorder: PolicyTraceRecorder | None = None,
        event_sink: Callable[[PolicyTraceEvent], None] | None = None,
    ) -> None:
        if not tool_registry:
            raise PolicyError("tool_registry must not be empty")

        self._tool_descriptors = {
            tool_id: ToolDescriptor(tool_id=tool_id, tags=_normalize_tags(meta))
            for tool_id, meta in tool_registry.items()
        }
        self._tool_sets = {name: tuple(entries) for name, entries in (tool_sets or {}).items()}
        self._validate_tool_sets()
        self._recorder = recorder or PolicyTraceRecorder()
        self._event_sink = event_sink
        self._stack: list[_PolicyFrame] = []

    @property
    def recorder(self) -> PolicyTraceRecorder:
        return self._recorder

    @property
    def stack_depth(self) -> int:
        return len(self._stack)

    def push(
        self,
        policy: Mapping[str, Sequence[str]] | None,
        *,
        scope: str,
        source: str | None = None,
    ) -> None:
        normalized = self._normalize_policy(policy or {})
        frame = _PolicyFrame(
            scope=scope,
            source=source,
            allow_tools=normalized.get("allow_tools"),
            allow_tags=normalized.get("allow_tags"),
            deny_tools=normalized.get("deny_tools"),
            deny_tags=normalized.get("deny_tags"),
            raw=_serialize_policy(normalized),
        )
        self._stack.append(frame)
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_push",
            scope=scope,
            payload={
                "source": source,
                "policy": frame.raw,
                "stack_depth": self.stack_depth,
            },
        )

    def pop(self, *, scope: str) -> None:
        if not self._stack:
            raise PolicyError("PolicyStack.pop() called on empty stack")

        frame = self._stack.pop()
        if frame.scope != scope:
            self._stack.append(frame)
            raise PolicyError(f"policy scope mismatch: expected '{frame.scope}', got '{scope}'")

        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_pop",
            scope=scope,
            payload={
                "source": frame.source,
                "policy": frame.raw,
                "stack_depth": self.stack_depth,
            },
        )

    def effective_allowlist(self, *, candidates: Iterable[str] | None = None) -> PolicySnapshot:
        candidate_set = _resolve_candidates(candidates, self._tool_descriptors)
        directives = self._resolve_directives()

        allowed: set[str] = set()
        denied: set[str] = set()
        decisions: dict[str, PolicyDecision] = {}
        denials: list[PolicyDenial] = []

        for tool_id in candidate_set:
            descriptor = self._tool_descriptors[tool_id]
            tags = descriptor.tags

            allowed_by_tools = False
            allow_scope = None
            if directives.allow_tools is None:
                allowed_by_tools = True
            else:
                allowed_by_tools = tool_id in directives.allow_tools.values
                allow_scope = directives.allow_tools.scope if allowed_by_tools else None

            allowed_by_tags = False
            if directives.allow_tags is None:
                allowed_by_tags = directives.allow_tools is None
            else:
                tag_match = tags & directives.allow_tags.values
                allowed_by_tags = bool(tag_match)
                if allowed_by_tags:
                    allow_scope = directives.allow_tags.scope

            allow_result = allowed_by_tools or allowed_by_tags
            reason = "allowed" if allow_result else "not_in_allowlist"
            deny_scope = None
            matched_tags = (
                tags & directives.allow_tags.values if directives.allow_tags else frozenset()
            )

            if directives.deny_tools is not None and tool_id in directives.deny_tools.values:
                allow_result = False
                reason = "denied:tool"
                deny_scope = directives.deny_tools.scope
            elif directives.deny_tags is not None:
                denied_tags = tags & directives.deny_tags.values
                if denied_tags:
                    allow_result = False
                    reason = "denied:tag"
                    deny_scope = directives.deny_tags.scope
                    matched_tags = denied_tags

            decision = PolicyDecision(
                tool=tool_id,
                allowed=allow_result,
                reason=reason,
                allow_scope=allow_scope,
                deny_scope=deny_scope,
                matched_tags=frozenset(matched_tags),
            )
            decisions[tool_id] = decision

            if allow_result:
                allowed.add(tool_id)
            else:
                denied.add(tool_id)
                if reason.startswith("denied") or reason == "not_in_allowlist":
                    denials.append(
                        PolicyDenial(tool=tool_id, reason=reason, scope=deny_scope or allow_scope)
                    )

        snapshot = PolicySnapshot(
            allowed=frozenset(allowed),
            denied=frozenset(denied),
            candidates=candidate_set,
            decisions=decisions,
            denials=tuple(denials),
        )

        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_resolved",
            scope=self._stack[-1].scope if self._stack else "<root>",
            payload={
                "allowed": sorted(snapshot.allowed),
                "denied": sorted(snapshot.denied),
                "candidates": sorted(snapshot.candidates),
                "stack_depth": self.stack_depth,
            },
        )
        return snapshot

    def enforce(
        self,
        tool: str,
        *,
        raise_on_violation: bool = True,
    ) -> PolicySnapshot:
        if tool not in self._tool_descriptors:
            raise PolicyError(f"unknown tool '{tool}'")

        snapshot = self.effective_allowlist(candidates=[tool])
        if tool in snapshot.allowed:
            return snapshot

        denial = next((denial for denial in snapshot.denials if denial.tool == tool), None)
        if denial is None:
            denial = PolicyDenial(tool=tool, reason="not_in_allowlist", scope=None)

        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_violation",
            scope=denial.scope or (self._stack[-1].scope if self._stack else "<root>"),
            payload={
                "tool": tool,
                "reason": denial.reason,
                "stack_depth": self.stack_depth,
            },
        )

        if raise_on_violation:
            raise PolicyViolationError(denial)

        return snapshot

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _validate_tool_sets(self) -> None:
        for set_name in self._tool_sets:
            self._expand_tool_set(set_name, deque(), set())

    def _expand_tool_set(
        self,
        set_name: str,
        stack: deque[str],
        seen: set[str],
    ) -> frozenset[str]:
        if set_name in seen or set_name in stack:
            cycle = " -> ".join(list(stack) + [set_name])
            raise PolicyError(f"cyclic tool_set reference detected: {cycle}")

        if set_name not in self._tool_sets:
            raise PolicyError(f"unknown tool_set '{set_name}'")

        stack.append(set_name)
        seen.add(set_name)
        expanded: set[str] = set()
        for entry in self._tool_sets[set_name]:
            if entry in self._tool_descriptors:
                expanded.add(entry)
            elif entry in self._tool_sets:
                expanded.update(self._expand_tool_set(entry, stack, seen))
            else:
                raise PolicyError(f"unknown tool reference '{entry}' in tool_set '{set_name}'")
        stack.pop()
        return frozenset(expanded)

    def _normalize_policy(
        self, policy: Mapping[str, Sequence[str]]
    ) -> dict[str, frozenset[str] | None]:
        normalized: dict[str, frozenset[str] | None] = {
            "allow_tools": None,
            "allow_tags": None,
            "deny_tools": None,
            "deny_tags": None,
        }

        if "allow_tools" in policy:
            normalized["allow_tools"] = self._expand_entries(policy["allow_tools"])
        if "deny_tools" in policy:
            normalized["deny_tools"] = self._expand_entries(policy["deny_tools"])
        if "allow_tags" in policy:
            normalized["allow_tags"] = frozenset(policy["allow_tags"])
        if "deny_tags" in policy:
            normalized["deny_tags"] = frozenset(policy["deny_tags"])

        return normalized

    def _expand_entries(self, entries: Sequence[str]) -> frozenset[str]:
        expanded: set[str] = set()
        for entry in entries:
            if entry in self._tool_descriptors:
                expanded.add(entry)
            elif entry in self._tool_sets:
                expanded.update(self._expand_tool_set(entry, deque(), set()))
            else:
                raise PolicyError(f"unknown tool reference '{entry}'")
        return frozenset(expanded)

    def _resolve_directives(self) -> _DirectiveResolution:
        allow_tools = _DirectiveValue.empty()
        allow_tags = _DirectiveValue.empty()
        deny_tools = _DirectiveValue.empty()
        deny_tags = _DirectiveValue.empty()

        for frame in reversed(self._stack):
            if allow_tools.is_empty and frame.allow_tools is not None:
                allow_tools = _DirectiveValue(frame.allow_tools, frame.scope)
            if allow_tags.is_empty and frame.allow_tags is not None:
                allow_tags = _DirectiveValue(frame.allow_tags, frame.scope)
            if deny_tools.is_empty and frame.deny_tools is not None:
                deny_tools = _DirectiveValue(frame.deny_tools, frame.scope)
            if deny_tags.is_empty and frame.deny_tags is not None:
                deny_tags = _DirectiveValue(frame.deny_tags, frame.scope)

            if not any(
                value.is_empty for value in (allow_tools, allow_tags, deny_tools, deny_tags)
            ):
                break

        return _DirectiveResolution(
            allow_tools=None if allow_tools.is_empty else allow_tools,
            allow_tags=None if allow_tags.is_empty else allow_tags,
            deny_tools=None if deny_tools.is_empty else deny_tools,
            deny_tags=None if deny_tags.is_empty else deny_tags,
        )


@dataclass(frozen=True)
class _DirectiveValue:
    values: frozenset[str]
    scope: str | None

    @property
    def is_empty(self) -> bool:
        return self.scope is None

    @staticmethod
    def empty() -> _DirectiveValue:
        return _DirectiveValue(values=frozenset(), scope=None)


@dataclass(frozen=True)
class _DirectiveResolution:
    allow_tools: _DirectiveValue | None
    allow_tags: _DirectiveValue | None
    deny_tools: _DirectiveValue | None
    deny_tags: _DirectiveValue | None


def _normalize_tags(meta: Mapping[str, Any]) -> frozenset[str]:
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        return frozenset([tags])
    if not isinstance(tags, Iterable):
        return frozenset()
    return frozenset(str(tag) for tag in tags)


def _serialize_policy(policy: Mapping[str, frozenset[str] | None]) -> Mapping[str, Sequence[str]]:
    serialized: dict[str, Sequence[str]] = {}
    for key, value in policy.items():
        if value is not None:
            serialized[key] = tuple(sorted(value))
    return MappingProxyType(serialized)


def _resolve_candidates(
    candidates: Iterable[str] | None,
    registry: Mapping[str, ToolDescriptor],
) -> frozenset[str]:
    if candidates is None:
        return frozenset(registry.keys())

    resolved: set[str] = set()
    for candidate in candidates:
        if candidate not in registry:
            raise PolicyError(f"unknown tool '{candidate}' in candidates")
        resolved.add(candidate)
    return frozenset(resolved)

