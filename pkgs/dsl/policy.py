"""Policy resolution utilities for the RAGX DSL."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


class PolicyError(RuntimeError):
    """Raised when policy definitions are invalid or cannot be resolved."""


@dataclass(slots=True)
class PolicyTraceEvent:
    """A lightweight trace record emitted when the policy stack mutates."""

    event: str
    scope: str
    data: Mapping[str, object]


class PolicyTraceRecorder:
    """Collects policy trace events for later inspection (e.g. tests or logs)."""

    def __init__(self) -> None:
        self.events: list[PolicyTraceEvent] = []

    def record(self, event: PolicyTraceEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class _ToolMetadata:
    """Normalized tool metadata required for policy resolution."""

    name: str
    tags: frozenset[str]


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    """Normalized representation of a policy layer."""

    allow_tools: frozenset[str] | None = None
    deny_tools: frozenset[str] | None = None
    allow_tags: frozenset[str] | None = None
    deny_tags: frozenset[str] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Iterable[str]] | None) -> PolicyDefinition:
        if data is None:
            return cls()

        def _normalize(key: str) -> frozenset[str] | None:
            if key not in data:
                return None
            value = data[key]
            if value is None:
                return frozenset()
            if not isinstance(value, Iterable) or isinstance(value, str | bytes):
                raise PolicyError(f"Policy field '{key}' must be an iterable of strings")
            items: set[str] = set()
            for entry in value:
                if not isinstance(entry, str):
                    raise PolicyError(f"Policy field '{key}' must contain only strings")
                items.add(entry)
            return frozenset(items)

        return cls(
            allow_tools=_normalize("allow_tools"),
            deny_tools=_normalize("deny_tools"),
            allow_tags=_normalize("allow_tags"),
            deny_tags=_normalize("deny_tags"),
        )

    def to_dict(self) -> Mapping[str, Sequence[str] | None]:
        return MappingProxyType(
            {
                "allow_tools": sorted(self.allow_tools) if self.allow_tools is not None else None,
                "deny_tools": sorted(self.deny_tools) if self.deny_tools is not None else None,
                "allow_tags": sorted(self.allow_tags) if self.allow_tags is not None else None,
                "deny_tags": sorted(self.deny_tags) if self.deny_tags is not None else None,
            }
        )


@dataclass(slots=True)
class PolicyResolution:
    """Effective allowlist and diagnostics for the current policy stack."""

    allowed: frozenset[str]
    blocked: frozenset[str]
    reasons: Mapping[str, str]

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed


@dataclass(slots=True)
class _PolicyLayer:
    """A single entry in the policy stack."""

    policy: PolicyDefinition
    scope: str
    source: str


class PolicyStack:
    """Maintains hierarchical policies and computes effective allowlists."""

    def __init__(
        self,
        *,
        tools: Mapping[str, Mapping[str, object]],
        tool_sets: Mapping[str, Sequence[str]] | None = None,
        trace: PolicyTraceRecorder | None = None,
    ) -> None:
        if not isinstance(tools, Mapping):
            raise PolicyError("tools must be a mapping of tool name -> metadata")

        self._tools: dict[str, _ToolMetadata] = {
            name: _ToolMetadata(name=name, tags=self._normalize_tags(meta))
            for name, meta in tools.items()
        }
        self._tool_sets: dict[str, tuple[str, ...]] = {
            name: tuple(self._normalize_tool_ref(ref) for ref in refs)
            for name, refs in (tool_sets or {}).items()
        }
        self.trace: PolicyTraceRecorder = trace or PolicyTraceRecorder()
        self.stack: list[_PolicyLayer] = []

    @staticmethod
    def _normalize_tags(meta: Mapping[str, object]) -> frozenset[str]:
        tags = meta.get("tags", []) if isinstance(meta, Mapping) else []
        if tags is None:
            return frozenset()
        if not isinstance(tags, Iterable) or isinstance(tags, str | bytes):
            raise PolicyError("Tool metadata 'tags' must be an iterable of strings")
        normalized: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str):
                raise PolicyError("Tool metadata 'tags' must contain only strings")
            normalized.add(tag)
        return frozenset(normalized)

    @staticmethod
    def _normalize_tool_ref(ref: str) -> str:
        if not isinstance(ref, str) or not ref:
            raise PolicyError("Tool set entries must be non-empty strings")
        return ref

    def _expand_tool_refs(
        self, refs: Iterable[str], *, _stack: tuple[str, ...] = ()
    ) -> frozenset[str]:
        expanded: set[str] = set()
        for ref in refs:
            if ref in self._tools:
                expanded.add(ref)
                continue
            if ref in self._tool_sets:
                if ref in _stack:
                    cycle = " -> ".join((*_stack, ref))
                    raise PolicyError(f"Detected cycle while expanding tool set '{cycle}'")
                expanded.update(self._expand_tool_refs(self._tool_sets[ref], _stack=(*_stack, ref)))
                continue
            raise PolicyError(f"Unknown policy reference '{ref}'")
        return frozenset(expanded)

    def push(
        self,
        policy_data: Mapping[str, Iterable[str]] | None,
        *,
        scope: str,
        source: str,
    ) -> None:
        policy = PolicyDefinition.from_mapping(policy_data)
        if policy.allow_tools is not None:
            self._expand_tool_refs(policy.allow_tools)
        if policy.deny_tools is not None:
            self._expand_tool_refs(policy.deny_tools)
        self.stack.append(_PolicyLayer(policy=policy, scope=scope, source=source))
        self.trace.record(
            PolicyTraceEvent(
                event="policy_push",
                scope=scope,
                data={"source": source, "policy": policy.to_dict()},
            )
        )

    def pop(self, *, scope: str) -> None:
        if not self.stack:
            raise PolicyError("Cannot pop from an empty policy stack")
        layer = self.stack.pop()
        if layer.scope != scope:
            raise PolicyError(
                f"Policy stack scope mismatch: expected to pop '{layer.scope}', got '{scope}'"
            )
        self.trace.record(
            PolicyTraceEvent(
                event="policy_pop",
                scope=scope,
                data={"source": layer.source},
            )
        )

    def _resolve_tools_dimension(self, attribute: str) -> frozenset[str] | None:
        for layer in reversed(self.stack):
            refs = getattr(layer.policy, attribute)
            if refs is not None:
                return self._expand_tool_refs(refs)
        return None

    def _resolve_tags_dimension(self, attribute: str) -> frozenset[str] | None:
        for layer in reversed(self.stack):
            tags = getattr(layer.policy, attribute)
            if tags is not None:
                return tags
        return None

    def _resolve_dimensions(self) -> tuple[
        frozenset[str] | None,
        frozenset[str] | None,
        frozenset[str] | None,
        frozenset[str] | None,
    ]:
        allow_tools = self._resolve_tools_dimension("allow_tools")
        deny_tools = self._resolve_tools_dimension("deny_tools")
        allow_tags = self._resolve_tags_dimension("allow_tags")
        deny_tags = self._resolve_tags_dimension("deny_tags")
        return allow_tools, deny_tools, allow_tags, deny_tags

    def effective_allowlist(self) -> PolicyResolution:
        allow_tools, deny_tools, allow_tags, deny_tags = self._resolve_dimensions()

        allowed: set[str] = set()
        blocked: set[str] = set()
        reasons: dict[str, str] = {}

        for tool_name, meta in self._tools.items():
            base_allowed = False
            if allow_tools is None and allow_tags is None:
                base_allowed = True
            else:
                if allow_tools is not None and tool_name in allow_tools:
                    base_allowed = True
                if allow_tags is not None and meta.tags & allow_tags:
                    base_allowed = True

            if deny_tools is not None and tool_name in deny_tools:
                blocked.add(tool_name)
                reasons[tool_name] = f"denied:tool:{tool_name}"
                continue

            if deny_tags is not None:
                overlap = meta.tags & deny_tags
                if overlap:
                    blocked.add(tool_name)
                    denied_tag = sorted(overlap)[0]
                    reasons[tool_name] = f"denied:tag:{denied_tag}"
                    continue

            if base_allowed:
                allowed.add(tool_name)
            else:
                blocked.add(tool_name)
                reasons[tool_name] = "not_in_allowlist"

        resolution = PolicyResolution(
            allowed=frozenset(allowed),
            blocked=frozenset(blocked),
            reasons=MappingProxyType(dict(reasons)),
        )
        self.trace.record(
            PolicyTraceEvent(
                event="policy_resolved",
                scope="stack",
                data={
                    "allowed": sorted(resolution.allowed),
                    "blocked": sorted(resolution.blocked),
                    "allow_tools": sorted(allow_tools) if allow_tools is not None else None,
                    "deny_tools": sorted(deny_tools) if deny_tools is not None else None,
                    "allow_tags": sorted(allow_tags) if allow_tags is not None else None,
                    "deny_tags": sorted(deny_tags) if deny_tags is not None else None,
                },
            )
        )
        return resolution

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._tools.keys())
