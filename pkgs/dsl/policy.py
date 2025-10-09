from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .models import ToolDescriptor


@dataclass(frozen=True, slots=True)
class PolicyTraceEvent:
    """Structured record describing stack mutations or violations."""

    event: str
    scope: str
    payload: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result of evaluating a tool against the active policy stack."""

    allowed: bool
    source: str
    scope: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyResolution:
    """Summary of an allowlist evaluation for the current stack state."""

    allowed: frozenset[str]
    denied: frozenset[str]
    decisions: Mapping[str, PolicyDecision]


@dataclass(frozen=True, slots=True)
class _PolicyFrame:
    scope: str
    allow_tools: Mapping[str, str]
    deny_tools: Mapping[str, str]
    allow_tags: frozenset[str]
    deny_tags: frozenset[str]
    raw: Mapping[str, object]


class PolicyStack:
    """Hierarchical allow/deny evaluator for DSL policies.

    The stack processes policies in the order they are pushed.  The closest
    (most recent) policy wins when directives conflict, which matches the
    ``scoping_order`` contract in the specification (``globals < loop <
    decision-option < node``).
    """

    def __init__(
        self,
        *,
        tool_sets: Mapping[str, Sequence[str]] | None = None,
        trace: MutableSequence[PolicyTraceEvent] | None = None,
    ) -> None:
        self._tool_sets: Mapping[str, tuple[str, ...]] = (
            {
                name: tuple(values)
                for name, values in (tool_sets or {}).items()
            }
        )
        self.stack: list[_PolicyFrame] = []
        self.trace: MutableSequence[PolicyTraceEvent] = trace or []

    # ------------------------------------------------------------------
    # Stack mutations
    # ------------------------------------------------------------------
    def push(self, policy: Mapping[str, object] | None, *, scope: str) -> None:
        """Push a policy frame onto the stack.

        ``None`` policies are ignored so callers can forward optional
        directives without additional checks.
        """

        if not policy:
            return

        frame = self._build_frame(policy, scope=scope)
        self.stack.append(frame)
        self.trace.append(PolicyTraceEvent("policy_push", scope, frame.raw))

    def pop(self) -> None:
        if not self.stack:
            raise IndexError("PolicyStack.pop() called on empty stack")
        frame = self.stack.pop()
        self.trace.append(PolicyTraceEvent("policy_pop", frame.scope, frame.raw))

    def clone(self) -> PolicyStack:
        clone = PolicyStack(tool_sets=self._tool_sets, trace=list(self.trace))
        clone.stack = list(self.stack)
        return clone

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def effective_allowlist(
        self, tools: Mapping[str, ToolDescriptor | Mapping[str, object]]
    ) -> PolicyResolution:
        """Return the allow/deny decision for every tool in ``tools``."""

        descriptors: dict[str, ToolDescriptor] = {}
        for name, descriptor in tools.items():
            if isinstance(descriptor, ToolDescriptor):
                descriptors[name] = descriptor
            elif isinstance(descriptor, Mapping):
                descriptors[name] = ToolDescriptor.from_mapping(name, descriptor)
            else:  # pragma: no cover - defensive guard for unexpected input
                raise TypeError(
                    f"Unsupported tool descriptor for '{name}': {type(descriptor)!r}"
                )

        any_allow_directive = any(
            frame.allow_tools or frame.allow_tags for frame in self.stack
        )

        decisions: dict[str, PolicyDecision] = {}
        allowed: set[str] = set()
        denied: set[str] = set()
        for name, descriptor in descriptors.items():
            decision = self._resolve_tool(
                descriptor, any_allow_directive=any_allow_directive
            )
            decisions[name] = decision
            (allowed if decision.allowed else denied).add(name)

        return PolicyResolution(
            allowed=frozenset(allowed),
            denied=frozenset(denied),
            decisions=MappingProxyType(decisions),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_tool(
        self, descriptor: ToolDescriptor, *, any_allow_directive: bool
    ) -> PolicyDecision:
        tags = set(descriptor.tags)
        for frame in reversed(self.stack):
            if descriptor.name in frame.deny_tools:
                detail = frame.deny_tools[descriptor.name]
                return PolicyDecision(False, "deny_tools", frame.scope, detail)
            if descriptor.name in frame.allow_tools:
                detail = frame.allow_tools[descriptor.name]
                return PolicyDecision(True, "allow_tools", frame.scope, detail)

            deny_match = self._match_tags(tags, frame.deny_tags)
            if deny_match:
                return PolicyDecision(False, "deny_tags", frame.scope, deny_match)

            allow_match = self._match_tags(tags, frame.allow_tags)
            if allow_match:
                return PolicyDecision(True, "allow_tags", frame.scope, allow_match)

        if any_allow_directive:
            return PolicyDecision(False, "implicit_deny", "effective", None)
        return PolicyDecision(True, "implicit_allow", "effective", None)

    def _build_frame(
        self, policy: Mapping[str, object], *, scope: str
    ) -> _PolicyFrame:
        allow_tools = self._expand_tool_entries(policy.get("allow_tools"))
        deny_tools = self._expand_tool_entries(policy.get("deny_tools"))
        allow_tags = frozenset(self._string_iter(policy.get("allow_tags")))
        deny_tags = frozenset(self._string_iter(policy.get("deny_tags")))
        raw = MappingProxyType(dict(policy))
        return _PolicyFrame(
            scope=scope,
            allow_tools=MappingProxyType(allow_tools),
            deny_tools=MappingProxyType(deny_tools),
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            raw=raw,
        )

    def _expand_tool_entries(
        self, entries: object
    ) -> dict[str, str]:  # mapping tool -> origin
        tools: dict[str, str] = {}
        for entry in self._string_iter(entries):
            expanded = self._tool_sets.get(entry, (entry,))
            for tool_name in expanded:
                tools[tool_name] = entry
        return tools

    @staticmethod
    def _string_iter(entries: object | None) -> Iterable[str]:
        if entries is None:
            return ()
        if isinstance(entries, str):
            return (entries,)
        if isinstance(entries, Iterable):
            return (str(item) for item in entries)
        raise TypeError(f"Policy entries must be strings or iterables, got {type(entries)!r}")

    @staticmethod
    def _match_tags(tags: set[str], policy_tags: frozenset[str]) -> str | None:
        if not tags or not policy_tags:
            return None
        intersection = sorted(tags.intersection(policy_tags))
        if not intersection:
            return None
        return ",".join(intersection)
