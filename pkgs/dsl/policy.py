"""Hierarchical policy stack enforcing DSL allow/deny semantics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .models import (
    PolicyDecision,
    PolicyDenial,
    PolicyResolution,
    PolicySnapshot,
    ToolDescriptor,
    mapping_proxy,
)

__all__ = [
    "PolicyError",
    "PolicyViolationError",
    "PolicyTraceEvent",
    "PolicyTraceRecorder",
    "PolicyStack",
]


class PolicyError(RuntimeError):
    """Raised when policy configuration or evaluation fails."""


class PolicyViolationError(PolicyError):
    """Raised when `PolicyStack.enforce()` detects a violation."""

    def __init__(self, denial: PolicyDenial) -> None:
        super().__init__(
            f"Tool '{denial.tool}' blocked by policy: {', '.join(denial.reasons)}"
        )
        self.denial = denial


@dataclass(frozen=True, slots=True)
class PolicyTraceEvent:
    """Structured trace record emitted by :class:`PolicyStack`."""

    event: str
    scope: str
    data: Mapping[str, object]


class PolicyTraceRecorder:
    """Collector used in tests to capture policy trace events."""

    def __init__(self) -> None:
        self._events: list[PolicyTraceEvent] = []

    def record(self, event: PolicyTraceEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> Sequence[PolicyTraceEvent]:
        return tuple(self._events)


def _emit_policy_event(
    recorder: PolicyTraceRecorder | None,
    sink: Callable[[PolicyTraceEvent], None] | None,
    *,
    event: str,
    scope: str,
    payload: Mapping[str, object],
) -> None:
    record = PolicyTraceEvent(event=event, scope=scope, data=mapping_proxy(payload))
    if recorder is not None:
        recorder.record(record)
    if sink is not None:
        sink(record)


@dataclass(slots=True)
class _PolicyFrame:
    scope: str
    source: str | None
    allow_tools: frozenset[str] | None
    deny_tools: frozenset[str] | None
    allow_tags: frozenset[str] | None
    deny_tags: frozenset[str] | None
    raw: Mapping[str, object]


@dataclass(slots=True)
class _ResolvedDirectives:
    allow_tools: frozenset[str] | None
    allow_tags: frozenset[str] | None
    deny_tools: frozenset[str] | None
    deny_tags: frozenset[str] | None
    allow_tools_scope: str | None
    allow_tags_scope: str | None
    deny_tools_scope: str | None
    deny_tags_scope: str | None


class PolicyStack:
    """Resolve hierarchical allow/deny policies for DSL execution."""

    def __init__(
        self,
        *,
        tools: Mapping[str, Mapping[str, object]],
        tool_sets: Mapping[str, Sequence[str]] | None = None,
        trace: PolicyTraceRecorder | None = None,
        event_sink: Callable[[PolicyTraceEvent], None] | None = None,
    ) -> None:
        self._trace = trace
        self._event_sink = event_sink
        self._registry: dict[str, ToolDescriptor] = {}
        for tool_name, definition in tools.items():
            tags = definition.get("tags", [])
            if not isinstance(tags, Iterable):
                raise PolicyError(f"Tool '{tool_name}' tags must be iterable")
            metadata = dict(definition)
            metadata.setdefault("tags", list(tags))
            descriptor = ToolDescriptor(
                name=tool_name,
                tags=frozenset(str(tag) for tag in tags),
                metadata=metadata,
            )
            self._registry[tool_name] = descriptor

        self._tool_sets = {
            name: tuple(entries)
            for name, entries in (tool_sets or {}).items()
        }
        self._validate_tool_sets()
        self._frames: list[_PolicyFrame] = []

    # ------------------------------------------------------------------
    # Stack operations
    # ------------------------------------------------------------------
    def push(
        self,
        policy: Mapping[str, Iterable[str]] | None,
        *,
        scope: str,
        source: str | None = None,
    ) -> None:
        if policy is not None and not isinstance(policy, Mapping):
            raise PolicyError("policy must be a mapping or None")
        normalized = dict(policy or {})

        frame = _PolicyFrame(
            scope=scope,
            source=source,
            allow_tools=self._expand_entries(normalized.get("allow_tools")),
            deny_tools=self._expand_entries(normalized.get("deny_tools")),
            allow_tags=self._normalize_tags(normalized.get("allow_tags")),
            deny_tags=self._normalize_tags(normalized.get("deny_tags")),
            raw=mapping_proxy(normalized),
        )
        self._frames.append(frame)
        _emit_policy_event(
            self._trace,
            self._event_sink,
            event="policy_push",
            scope=scope,
            payload={
                "source": source,
                "policy": frame.raw,
                "stack_depth": len(self._frames),
            },
        )

    def pop(self, expected_scope: str | None = None) -> None:
        if not self._frames:
            raise PolicyError("cannot pop from empty policy stack")
        frame = self._frames.pop()
        if expected_scope is not None and frame.scope != expected_scope:
            self._frames.append(frame)
            raise PolicyError(
                f"scope mismatch on pop: expected {expected_scope!r}, got {frame.scope!r}"
            )
        _emit_policy_event(
            self._trace,
            self._event_sink,
            event="policy_pop",
            scope=frame.scope,
            payload={
                "source": frame.source,
                "policy": frame.raw,
                "stack_depth": len(self._frames),
            },
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def effective_allowlist(
        self, candidates: Iterable[str] | None = None
    ) -> PolicyResolution:
        return self._resolve(candidates, emit_trace=True)

    def snapshot(self, candidates: Iterable[str] | None = None) -> PolicySnapshot:
        resolution = self._resolve(candidates, emit_trace=False)
        return self._make_snapshot(resolution)

    def enforce(
        self,
        tool: str,
        *,
        raise_on_violation: bool = True,
    ) -> PolicySnapshot:
        resolution = self._resolve([tool], emit_trace=False)
        decision = resolution.decisions[tool]
        if decision.allowed:
            return self._make_snapshot(resolution)

        denial = PolicyDenial(tool=tool, reasons=decision.reasons, decision=decision)
        _emit_policy_event(
            self._trace,
            self._event_sink,
            event="policy_violation",
            scope=decision.denied_by or decision.granted_by or "stack",
            payload={
                "tool": tool,
                "reasons": decision.reasons,
                "stack_depth": len(self._frames),
            },
        )
        if raise_on_violation:
            raise PolicyViolationError(denial)
        return self._make_snapshot(resolution)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve(
        self,
        candidates: Iterable[str] | None,
        *,
        emit_trace: bool,
    ) -> PolicyResolution:
        tool_names = list(candidates) if candidates is not None else list(self._registry)
        directives = self._resolve_directives()

        allowed: list[str] = []
        denied: dict[str, tuple[str, ...]] = {}
        decisions: dict[str, PolicyDecision] = {}

        for tool_name in tool_names:
            descriptor = self._registry.get(tool_name)
            if descriptor is None:
                raise PolicyError(f"unknown tool '{tool_name}'")

            decision = self._evaluate_tool(tool_name, descriptor, directives)
            decisions[tool_name] = decision
            if decision.allowed:
                allowed.append(tool_name)
            else:
                denied[tool_name] = decision.reasons

        directives_payload = {
            "allow_tools": directives.allow_tools,
            "allow_tags": directives.allow_tags,
            "deny_tools": directives.deny_tools,
            "deny_tags": directives.deny_tags,
            "allow_tools_scope": directives.allow_tools_scope,
            "allow_tags_scope": directives.allow_tags_scope,
            "deny_tools_scope": directives.deny_tools_scope,
            "deny_tags_scope": directives.deny_tags_scope,
        }

        resolution = PolicyResolution(
            allowed=frozenset(allowed),
            denied=mapping_proxy(denied),
            decisions=mapping_proxy(decisions),
            stack_depth=len(self._frames),
            candidates=tuple(tool_names),
            directives=mapping_proxy(directives_payload),
        )

        if emit_trace:
            _emit_policy_event(
                self._trace,
                self._event_sink,
                event="policy_resolved",
                scope="stack",
                payload={
                    "allowed": sorted(resolution.allowed),
                    "denied": {
                        name: list(reasons) for name, reasons in resolution.denied.items()
                    },
                    "candidates": list(tool_names),
                    "stack_depth": resolution.stack_depth,
                },
            )
        return resolution

    def _make_snapshot(self, resolution: PolicyResolution) -> PolicySnapshot:
        return PolicySnapshot(
            allowed=resolution.allowed,
            denied=resolution.denied,
            decisions=resolution.decisions,
            stack_depth=resolution.stack_depth,
            candidates=resolution.candidates,
            directives=resolution.directives,
        )

    def _evaluate_tool(
        self,
        tool_name: str,
        descriptor: ToolDescriptor,
        directives: _ResolvedDirectives,
    ) -> PolicyDecision:
        allow_tools = directives.allow_tools
        allow_tags = directives.allow_tags
        deny_tools = directives.deny_tools
        deny_tags = directives.deny_tags
        allow_tool_scope = directives.allow_tools_scope
        allow_tag_scope = directives.allow_tags_scope
        deny_tool_scope = directives.deny_tools_scope
        deny_tag_scope = directives.deny_tags_scope

        allowed = True
        granted_by: str | None = None
        denied_by: str | None = None
        matched_tags: list[str] = []
        reasons: list[str] = []

        if allow_tools is not None or allow_tags is not None:
            allowed = True
            allow_reasons: list[tuple[str, str | None]] = []

            if allow_tools is not None:
                if tool_name in allow_tools:
                    allow_reasons.append((f"allow:tool:{tool_name}", allow_tool_scope))
                else:
                    allowed = False

            if allowed and allow_tags is not None:
                tag_hit = sorted(descriptor.tags & allow_tags)
                if tag_hit:
                    matched_tags.extend(tag_hit)
                    allow_reasons.append((f"allow:tag:{tag_hit[0]}", allow_tag_scope))
                else:
                    allowed = False

            if allowed:
                for reason_entry, scope in allow_reasons:
                    reasons.append(reason_entry)
                    if scope is not None:
                        granted_by = scope
                if granted_by is None:
                    granted_by = allow_tag_scope or allow_tool_scope
            else:
                reasons.append("not_in_allowlist")

        if allowed:
            if deny_tools is not None and tool_name in deny_tools:
                allowed = False
                denied_by = deny_tool_scope
                reasons.append(f"deny:tool:{tool_name}")
            else:
                deny_tag_matches = sorted(descriptor.tags & (deny_tags or frozenset()))
                if deny_tag_matches:
                    allowed = False
                    denied_by = deny_tag_scope
                    matched_tags.extend(deny_tag_matches)
                    reasons.append(f"deny:tag:{deny_tag_matches[0]}")

        return PolicyDecision(
            descriptor=descriptor,
            allowed=allowed,
            reasons=tuple(reasons),
            granted_by=granted_by,
            denied_by=denied_by,
            matched_tags=tuple(matched_tags),
        )

    def _resolve_directives(self) -> _ResolvedDirectives:
        allow_tools = None
        allow_tags = None
        deny_tools = None
        deny_tags = None
        allow_tools_scope = None
        allow_tags_scope = None
        deny_tools_scope = None
        deny_tags_scope = None

        for frame in reversed(self._frames):
            if allow_tools is None and frame.allow_tools is not None:
                allow_tools = frame.allow_tools
                allow_tools_scope = frame.scope
            if allow_tags is None and frame.allow_tags is not None:
                allow_tags = frame.allow_tags
                allow_tags_scope = frame.scope
            if deny_tools is None and frame.deny_tools is not None:
                deny_tools = frame.deny_tools
                deny_tools_scope = frame.scope
            if deny_tags is None and frame.deny_tags is not None:
                deny_tags = frame.deny_tags
                deny_tags_scope = frame.scope

        return _ResolvedDirectives(
            allow_tools=allow_tools,
            allow_tags=allow_tags,
            deny_tools=deny_tools,
            deny_tags=deny_tags,
            allow_tools_scope=allow_tools_scope,
            allow_tags_scope=allow_tags_scope,
            deny_tools_scope=deny_tools_scope,
            deny_tags_scope=deny_tags_scope,
        )

    def _expand_entries(
        self, entries: Iterable[str] | None
    ) -> frozenset[str] | None:
        if entries is None:
            return None
        expanded: list[str] = []
        for entry in entries:
            expanded.extend(self._expand_tool_ref(str(entry), set()))
        return frozenset(expanded)

    def _normalize_tags(
        self, tags: Iterable[str] | None
    ) -> frozenset[str] | None:
        if tags is None:
            return None
        normalized = [str(tag) for tag in tags]
        return frozenset(normalized)

    def _expand_tool_ref(self, ref: str, seen: set[str]) -> list[str]:
        if ref in self._registry:
            return [ref]
        if ref in seen:
            raise PolicyError(f"cycle detected in tool_sets: {' -> '.join(seen | {ref})}")
        if ref not in self._tool_sets:
            raise PolicyError(f"unknown tool or tool_set '{ref}'")
        seen.add(ref)
        expanded: list[str] = []
        for nested in self._tool_sets[ref]:
            expanded.extend(self._expand_tool_ref(nested, seen))
        seen.remove(ref)
        return expanded

    def _validate_tool_sets(self) -> None:
        for name in self._tool_sets:
            self._expand_tool_ref(name, set())
