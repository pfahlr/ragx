from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class PolicyDefinitionError(ValueError):
    """Raised when a policy references unknown tools or tool sets."""


class PolicyViolationError(RuntimeError):
    """Raised when a tool invocation violates the active policy stack."""


@dataclass(frozen=True)
class PolicyDenial:
    """Structured reason for why a tool is not permitted."""

    reason: str
    scope: str
    policy_source: str | None = None


@dataclass(frozen=True)
class PolicyEvent:
    """Structured event emitted on policy mutations or violations."""

    kind: str
    scope: str
    policy: Mapping[str, Any] | None
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySnapshot:
    """Immutable snapshot of the current allowlist state."""

    allowed_tools: frozenset[str]
    denied_tools: Mapping[str, PolicyDenial]
    stack: Sequence[str]


@dataclass(frozen=True)
class _PolicyEntry:
    scope: str
    original: Mapping[str, Any]
    allow_tools: frozenset[str]
    allow_tags: frozenset[str]
    deny_tools: frozenset[str]
    deny_tags: frozenset[str]
    allow_from_tags: frozenset[str]
    deny_from_tags: Mapping[str, str]

    @property
    def allowed_subset(self) -> frozenset[str] | None:
        if self.allow_tools or self.allow_tags:
            return frozenset(self.allow_tools | self.allow_from_tags)
        return None

    @property
    def deny_reasons(self) -> Mapping[str, str]:
        reasons: dict[str, str] = {tool: "deny_tools" for tool in self.deny_tools}
        for tool, tag in self.deny_from_tags.items():
            reasons.setdefault(tool, f"deny_tags:{tag}")
        return reasons

    def reason_for_allow_exclusion(self, tool: str) -> str:
        if self.allow_tools and tool not in self.allow_tools:
            return "allow_tools_exclusion"
        if self.allow_tags and tool not in self.allow_from_tags:
            return "allow_tags_exclusion"
        return "allowlist_restriction"


class PolicyStack:
    """Stack of hierarchical DSL policies governing tool access."""

    def __init__(
        self,
        *,
        tool_registry: Mapping[str, Mapping[str, Any]],
        tool_sets: Mapping[str, Sequence[str]],
        event_sink: Callable[[PolicyEvent], None] | None = None,
    ) -> None:
        self._tool_registry = {
            name: meta if isinstance(meta, Mapping) else {}
            for name, meta in tool_registry.items()
        }
        self._tool_sets = {
            name: tuple(members)
            for name, members in tool_sets.items()
        }
        self._event_sink = event_sink
        self._stack: list[_PolicyEntry] = []

    # ------------------- Stack primitives -------------------
    def push(self, policy: Mapping[str, Any] | None, *, scope: str) -> None:
        entry = self._normalise_policy(policy or {}, scope=scope)
        self._stack.append(entry)
        self._emit("policy_push", scope=scope, policy=entry.original, detail={})

    def pop(self) -> None:
        if not self._stack:
            raise RuntimeError("Policy stack underflow")
        entry = self._stack.pop()
        self._emit("policy_pop", scope=entry.scope, policy=entry.original, detail={})

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._stack)

    # ------------------- Core operations -------------------
    def effective_allowlist(self) -> PolicySnapshot:
        allowed = set(self._tool_registry.keys())
        denied: MutableMapping[str, PolicyDenial] = {}
        scopes: list[str] = [entry.scope for entry in self._stack]

        for entry in self._stack:
            subset = entry.allowed_subset
            if subset is not None:
                removed = allowed - set(subset)
                for tool in removed:
                    if tool not in denied:
                        reason = entry.reason_for_allow_exclusion(tool)
                        denied[tool] = PolicyDenial(
                            reason=reason,
                            scope=entry.scope,
                            policy_source=self._describe_policy_source(entry, reason),
                        )
                allowed &= set(subset)

            for tool, reason in entry.deny_reasons.items():
                if tool in allowed:
                    allowed.remove(tool)
                if tool not in denied:
                    denied[tool] = PolicyDenial(
                        reason=reason,
                        scope=entry.scope,
                        policy_source=self._describe_policy_source(entry, reason),
                    )

        for tool in self._tool_registry:
            if tool not in allowed and tool not in denied:
                denied[tool] = PolicyDenial(
                    reason="allowlist_restriction",
                    scope=scopes[-1] if scopes else "root",
                    policy_source=None,
                )

        return PolicySnapshot(
            allowed_tools=frozenset(allowed),
            denied_tools=dict(denied),
            stack=tuple(scopes),
        )

    def enforce(
        self,
        tool_ref: str,
        *,
        scope: str | None = None,
        raise_on_violation: bool = True,
    ) -> bool:
        if tool_ref not in self._tool_registry:
            raise PolicyDefinitionError(f"Unknown tool '{tool_ref}'")

        snapshot = self.effective_allowlist()
        if tool_ref in snapshot.allowed_tools:
            return True

        denial = snapshot.denied_tools.get(tool_ref)
        reason = denial.reason if denial else "blocked_by_policy"
        detail = {
            "tool": tool_ref,
            "scope": scope,
            "reason": reason,
            "policy_scope": denial.scope if denial else None,
        }
        self._emit(
            "policy_violation",
            scope=scope or "<unspecified>",
            policy=None,
            detail=detail,
        )

        if raise_on_violation:
            raise PolicyViolationError(
                f"Tool '{tool_ref}' blocked by policy at scope "
                f"{denial.scope if denial else 'unknown'}: {reason}"
            )
        return False

    # ------------------- Helpers -------------------
    def _normalise_policy(self, policy: Mapping[str, Any], *, scope: str) -> _PolicyEntry:
        allow_tools = self._expand_tool_entries(policy.get("allow_tools"), scope)
        deny_tools = self._expand_tool_entries(policy.get("deny_tools"), scope)
        allow_tags = self._ensure_str_set(policy.get("allow_tags"))
        deny_tags = self._ensure_str_set(policy.get("deny_tags"))

        allow_from_tags = self._tools_with_tags(allow_tags)
        deny_from_tags = self._tools_with_tags_mapping(deny_tags)

        return _PolicyEntry(
            scope=scope,
            original=dict(policy),
            allow_tools=frozenset(allow_tools),
            allow_tags=frozenset(allow_tags),
            deny_tools=frozenset(deny_tools),
            deny_tags=frozenset(deny_tags),
            allow_from_tags=frozenset(allow_from_tags),
            deny_from_tags=deny_from_tags,
        )

    def _expand_tool_entries(
        self,
        tools: Iterable[str] | None,
        scope: str,
    ) -> set[str]:
        if tools is None:
            return set()
        expanded: set[str] = set()
        for entry in tools:
            if not isinstance(entry, str):
                raise PolicyDefinitionError(
                    f"Policy entry '{entry}' in scope '{scope}' must be a string"
                )
            if entry in self._tool_sets:
                expanded.update(self._tool_sets[entry])
            elif entry in self._tool_registry:
                expanded.add(entry)
            else:
                raise PolicyDefinitionError(
                    f"Policy scope '{scope}' references unknown tool or set '{entry}'"
                )
        return expanded

    @staticmethod
    def _ensure_str_set(values: Iterable[str] | None) -> set[str]:
        if values is None:
            return set()
        result: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise PolicyDefinitionError(
                    f"Policy tag entry '{value}' must be a string"
                )
            result.add(value)
        return result

    def _tools_with_tags(self, tags: Iterable[str]) -> set[str]:
        result: set[str] = set()
        if not tags:
            return result
        for tool, meta in self._tool_registry.items():
            meta_tags = meta.get("tags") if isinstance(meta, Mapping) else None
            if not isinstance(meta_tags, Iterable):
                continue
            tool_tags = {
                tag for tag in meta_tags if isinstance(tag, str)
            }
            if tool_tags & set(tags):
                result.add(tool)
        return result

    def _tools_with_tags_mapping(self, tags: Iterable[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for tag in tags:
            for tool, meta in self._tool_registry.items():
                meta_tags = meta.get("tags") if isinstance(meta, Mapping) else None
                if not isinstance(meta_tags, Iterable):
                    continue
                if tag in meta_tags and tool not in mapping:
                    mapping[tool] = tag
        return mapping

    @staticmethod
    def _describe_policy_source(entry: _PolicyEntry, reason: str) -> str | None:
        if reason.startswith("deny_tags") and entry.deny_tags:
            return ",".join(sorted(entry.deny_tags))
        if reason == "deny_tools" and entry.deny_tools:
            return ",".join(sorted(entry.deny_tools))
        if reason.startswith("allow_tags") and entry.allow_tags:
            return ",".join(sorted(entry.allow_tags))
        if reason.startswith("allow_tools") and entry.allow_tools:
            return ",".join(sorted(entry.allow_tools))
        return None

    def _emit(
        self,
        kind: str,
        *,
        scope: str,
        policy: Mapping[str, Any] | None,
        detail: Mapping[str, Any],
    ) -> None:
        if self._event_sink is None:
            return
        self._event_sink(PolicyEvent(kind=kind, scope=scope, policy=policy, detail=detail))
