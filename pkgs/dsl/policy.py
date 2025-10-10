"""Policy stack implementation for the DSL runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .models import PolicyDecision, PolicyDenial, PolicySnapshot, ToolDescriptor


class PolicyError(RuntimeError):
    """Raised when policy configuration or usage is invalid."""


@dataclass(frozen=True, slots=True)
class PolicyTraceEvent:
    """Structured trace emitted by ``PolicyStack`` operations."""

    event: str
    scope: str
    data: Mapping[str, Any]


class PolicyTraceRecorder:
    """In-memory recorder capturing policy trace events."""

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
    payload: Mapping[str, Any],
) -> None:
    record = PolicyTraceEvent(
        event=event,
        scope=scope,
        data=MappingProxyType(dict(payload)),
    )
    recorder.record(record)
    if sink is not None:
        sink(record)


@dataclass(frozen=True, slots=True)
class _PolicyDefinition:
    allow_tools: frozenset[str]
    deny_tools: frozenset[str]
    allow_tags: frozenset[str]
    deny_tags: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PolicyFrame:
    scope: str
    policy: _PolicyDefinition
    source: str | None


class PolicyViolationError(RuntimeError):
    """Raised when ``enforce`` detects that a tool is not permitted."""

    def __init__(self, denial: PolicyDenial, snapshot: PolicySnapshot) -> None:
        self.denial = denial
        self.snapshot = snapshot
        message = f"Tool '{denial.tool}' blocked by policy: {', '.join(denial.reasons)}"
        super().__init__(message)


class PolicyStack:
    """Maintains a stack of policy frames and resolves effective allowlists."""

    def __init__(
        self,
        tool_registry: Mapping[str, Mapping[str, Any]],
        tool_sets: Mapping[str, Sequence[str]],
        *,
        recorder: PolicyTraceRecorder | None = None,
        event_sink: Callable[[PolicyTraceEvent], None] | None = None,
    ) -> None:
        if not tool_registry:
            raise PolicyError("tool_registry must not be empty")
        self._tool_descriptors = self._normalize_registry(tool_registry)
        self._tool_sets_raw = {name: tuple(entries) for name, entries in tool_sets.items()}
        self._tool_sets_resolved = self._resolve_all_tool_sets()
        self._frames: list[_PolicyFrame] = []
        self._recorder = recorder or PolicyTraceRecorder()
        self._event_sink = event_sink

    @staticmethod
    def _normalize_registry(
        registry: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, ToolDescriptor]:
        descriptors: dict[str, ToolDescriptor] = {}
        for name, meta in registry.items():
            if not isinstance(name, str) or not name:
                raise PolicyError("tool names must be non-empty strings")
            if not isinstance(meta, Mapping):
                raise PolicyError(f"tool '{name}' metadata must be a mapping")
            tags = meta.get("tags", [])
            if not isinstance(tags, Iterable) or isinstance(tags, str | bytes):
                raise PolicyError(f"tool '{name}' tags must be an iterable of strings")
            tag_list = tuple(sorted({str(tag) for tag in tags}))
            descriptors[name] = ToolDescriptor(
                name=name,
                tags=tag_list,
                metadata=MappingProxyType(dict(meta)),
            )
        return MappingProxyType(descriptors)

    def _resolve_all_tool_sets(self) -> Mapping[str, frozenset[str]]:
        resolved: dict[str, frozenset[str]] = {}

        def resolve(name: str, stack: tuple[str, ...]) -> frozenset[str]:
            if name in resolved:
                return resolved[name]
            if name in stack:
                raise PolicyError(f"Cycle detected while resolving tool set '{name}'")
            if name not in self._tool_sets_raw:
                raise PolicyError(f"Unknown tool set '{name}' referenced during initialization")
            entries = self._tool_sets_raw[name]
            expanded: list[str] = []
            for entry in entries:
                expanded.extend(self._expand_tool_entry(entry, stack + (name,)))
            resolved[name] = frozenset(expanded)
            return resolved[name]

        for set_name in self._tool_sets_raw:
            resolve(set_name, tuple())
        return MappingProxyType(resolved)

    def _expand_tool_entry(self, entry: str, stack: tuple[str, ...]) -> list[str]:
        if entry in self._tool_descriptors:
            return [entry]
        resolved_sets = getattr(self, "_tool_sets_resolved", {})
        if entry in resolved_sets:
            return list(resolved_sets[entry])
        if entry in self._tool_sets_raw:
            return list(self._resolve_tool_set(entry, stack))
        raise PolicyError(f"Unknown tool or tool set '{entry}' referenced in policies")

    def _resolve_tool_set(self, name: str, stack: tuple[str, ...]) -> frozenset[str]:
        if name in self._tool_descriptors:
            # Allow direct references even if registry key accidentally collides with tool id.
            return frozenset({name})
        if name in stack:
            raise PolicyError(f"Cycle detected while resolving tool set '{name}'")
        if name not in self._tool_sets_raw:
            raise PolicyError(f"Unknown tool set '{name}'")
        expanded: list[str] = []
        for entry in self._tool_sets_raw[name]:
            expanded.extend(self._expand_tool_entry(entry, stack + (name,)))
        return frozenset(expanded)

    @property
    def tool_descriptors(self) -> Mapping[str, ToolDescriptor]:
        return self._tool_descriptors

    @property
    def recorder(self) -> PolicyTraceRecorder:
        return self._recorder

    def clone(self) -> PolicyStack:
        clone_stack = PolicyStack(
            tool_registry={
                name: dict(desc.metadata) for name, desc in self._tool_descriptors.items()
            },
            tool_sets={name: tuple(entries) for name, entries in self._tool_sets_raw.items()},
            recorder=self._recorder,
            event_sink=self._event_sink,
        )
        clone_stack._frames = list(self._frames)
        return clone_stack

    def push(
        self,
        scope: str,
        policy: Mapping[str, Any] | None,
        *,
        source: str | None = None,
    ) -> None:
        if not isinstance(scope, str) or not scope:
            raise PolicyError("scope must be a non-empty string")
        definition = self._normalize_policy(policy)
        frame = _PolicyFrame(scope=scope, policy=definition, source=source)
        self._frames.append(frame)
        payload = {
            "policy": {
                "allow_tools": sorted(definition.allow_tools),
                "deny_tools": sorted(definition.deny_tools),
                "allow_tags": sorted(definition.allow_tags),
                "deny_tags": sorted(definition.deny_tags),
            },
            "source": source,
            "stack_depth": len(self._frames),
        }
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_push",
            scope=scope,
            payload=payload,
        )

    def pop(self, scope: str) -> None:
        if not self._frames:
            raise PolicyError("PolicyStack.pop() on empty stack")
        frame = self._frames[-1]
        if frame.scope != scope:
            raise PolicyError(f"Scope mismatch on pop: expected '{frame.scope}' got '{scope}'")
        self._frames.pop()
        payload = {
            "stack_depth": len(self._frames),
            "source": frame.source,
        }
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_pop",
            scope=scope,
            payload=payload,
        )

    def effective_allowlist(self, *, candidates: Iterable[str] | None = None) -> PolicySnapshot:
        candidate_set = self._normalize_candidates(candidates)
        decisions: dict[str, PolicyDecision] = {}
        denials: dict[str, PolicyDenial] = {}

        for tool in candidate_set:
            descriptor = self._tool_descriptors[tool]
            decision = self._evaluate_tool(tool, descriptor)
            decisions[tool] = decision
            if not decision.allowed:
                denials[tool] = PolicyDenial(
                    tool=tool,
                    deciding_scope=decision.deciding_scope,
                    reasons=decision.reasons,
                    matched_tags=decision.matched_tags,
                )

        snapshot = PolicySnapshot.from_dicts(
            decisions=decisions,
            denied=denials,
            candidates=tuple(sorted(candidate_set)),
        )

        payload = {
            "candidates": list(snapshot.candidates),
            "allowed": list(snapshot.allowed),
            "denied": {
                tool: {
                    "reasons": list(denial.reasons),
                    "scope": denial.deciding_scope,
                    "matched_tags": list(denial.matched_tags),
                }
                for tool, denial in snapshot.denied.items()
            },
            "stack_depth": len(self._frames),
        }
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_resolved",
            scope="stack",
            payload=payload,
        )
        return snapshot

    def enforce(
        self,
        tool: str,
        *,
        raise_on_violation: bool = True,
    ) -> PolicySnapshot:
        snapshot = self.effective_allowlist(candidates=[tool])
        denial = snapshot.denied.get(tool)
        if denial is None:
            return snapshot
        payload = {
            "tool": tool,
            "reasons": list(denial.reasons),
            "scope": denial.deciding_scope,
            "stack_depth": len(self._frames),
        }
        emit_policy_event(
            self._recorder,
            self._event_sink,
            event="policy_violation",
            scope=denial.deciding_scope or "stack",
            payload=payload,
        )
        if raise_on_violation:
            raise PolicyViolationError(denial, snapshot)
        return snapshot

    def _normalize_candidates(self, candidates: Iterable[str] | None) -> list[str]:
        if candidates is None:
            return sorted(self._tool_descriptors.keys())
        normalized: list[str] = []
        for tool in candidates:
            if tool not in self._tool_descriptors:
                raise PolicyError(f"Unknown tool '{tool}' in candidate set")
            normalized.append(tool)
        return sorted(dict.fromkeys(normalized))

    def _evaluate_tool(
        self,
        tool: str,
        descriptor: ToolDescriptor,
    ) -> PolicyDecision:
        allow_tools_set: frozenset[str] | None = None
        allow_tools_scope: str | None = None
        allow_tags_set: frozenset[str] | None = None
        allow_tags_scope: str | None = None

        for frame in reversed(self._frames):
            policy = frame.policy
            if tool in policy.deny_tools:
                return PolicyDecision(tool, False, frame.scope, ("deny_tools",), tuple())
            deny_tag_matches = sorted(tag for tag in descriptor.tags if tag in policy.deny_tags)
            if deny_tag_matches:
                return PolicyDecision(
                    tool,
                    False,
                    frame.scope,
                    ("deny_tags",),
                    tuple(deny_tag_matches),
                )

            allow_tools_here = bool(policy.allow_tools and tool in policy.allow_tools)
            allow_tag_matches_here = sorted(
                tag for tag in descriptor.tags if tag in policy.allow_tags
            )
            if allow_tools_here or allow_tag_matches_here:
                reasons: list[str] = []
                if allow_tools_here:
                    reasons.append("allow_tools")
                if allow_tag_matches_here:
                    reasons.append("allow_tags")
                return PolicyDecision(
                    tool,
                    True,
                    frame.scope,
                    tuple(reasons),
                    tuple(allow_tag_matches_here),
                )

            if allow_tools_scope is None and policy.allow_tools:
                allow_tools_set = policy.allow_tools
                allow_tools_scope = frame.scope
            if allow_tags_scope is None and policy.allow_tags:
                allow_tags_set = policy.allow_tags
                allow_tags_scope = frame.scope

        if allow_tools_set is None and allow_tags_set is None:
            return PolicyDecision(tool, True, None, ("default_allow",), tuple())

        allowed_by_tools = bool(allow_tools_set and tool in allow_tools_set)
        tag_matches = sorted(tag for tag in descriptor.tags if allow_tags_set and tag in allow_tags_set)
        allowed_by_tags = bool(tag_matches)

        reasons: list[str] = []
        deciding_scope: str | None = None
        if allowed_by_tools:
            reasons.append("allow_tools")
            deciding_scope = allow_tools_scope
        if allowed_by_tags:
            reasons.append("allow_tags")
            if deciding_scope is None:
                deciding_scope = allow_tags_scope

        if reasons:
            return PolicyDecision(
                tool,
                True,
                deciding_scope,
                tuple(reasons),
                tuple(tag_matches),
            )

        denial_reasons: list[str] = []
        if allow_tools_set is not None:
            denial_reasons.append("not_in_allowlist")
            deciding_scope = deciding_scope or allow_tools_scope
        if allow_tags_set is not None:
            denial_reasons.append("missing_allow_tag")
            deciding_scope = deciding_scope or allow_tags_scope
        if not denial_reasons:
            denial_reasons.append("not_in_allowlist")
        return PolicyDecision(tool, False, deciding_scope, tuple(denial_reasons), tuple())

    def _normalize_policy(self, policy: Mapping[str, Any] | None) -> _PolicyDefinition:
        if policy is None:
            return _PolicyDefinition(frozenset(), frozenset(), frozenset(), frozenset())
        if not isinstance(policy, Mapping):
            raise PolicyError("policy must be a mapping")

        allowed_keys = {"allow_tools", "deny_tools", "allow_tags", "deny_tags"}
        for key in policy.keys():
            if key not in allowed_keys:
                raise PolicyError(f"Unknown policy directive '{key}'")

        allow_tools = self._expand_entries(policy.get("allow_tools"))
        deny_tools = self._expand_entries(policy.get("deny_tools"))
        allow_tags = self._normalize_strings(policy.get("allow_tags"))
        deny_tags = self._normalize_strings(policy.get("deny_tags"))

        return _PolicyDefinition(allow_tools, deny_tools, allow_tags, deny_tags)

    def _expand_entries(self, entries: Any) -> frozenset[str]:
        if entries is None:
            return frozenset()
        if isinstance(entries, str | bytes):
            raise PolicyError("Policy entries must be iterables of strings")
        if not isinstance(entries, Iterable):
            raise PolicyError("Policy entries must be iterables of strings")
        expanded: list[str] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise PolicyError("Policy tool references must be non-empty strings")
            expanded.extend(self._expand_tool_entry(entry, tuple()))
        return frozenset(expanded)

    def _normalize_strings(self, entries: Any) -> frozenset[str]:
        if entries is None:
            return frozenset()
        if isinstance(entries, str | bytes):
            raise PolicyError("Policy entries must be iterables of strings")
        if not isinstance(entries, Iterable):
            raise PolicyError("Policy entries must be iterables of strings")
        normalized: list[str] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise PolicyError("Policy tag references must be non-empty strings")
            normalized.append(entry)
        return frozenset(sorted(normalized))


__all__ = [
    "PolicyError",
    "PolicyStack",
    "PolicyTraceEvent",
    "PolicyTraceRecorder",
    "PolicyViolationError",
    "emit_policy_event",
]

