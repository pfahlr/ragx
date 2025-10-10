"""Core data models for the DSL policy engine.

These models are intentionally lightweight so that the policy stack can share
immutable diagnostics across trace events, enforcement checks, and linter
analysis.  All containers exposed here favour ``frozenset`` or
``MappingProxyType`` to avoid accidental mutation after construction.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Normalized description of a tool registered with the policy engine."""

    name: str
    tags: frozenset[str]

    @classmethod
    def from_spec(cls, name: str, *, tags: Sequence[str] | None = None) -> ToolDescriptor:
        return cls(name=name, tags=frozenset(tags or ()))


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Decision diagnostic for a single tool resolution."""

    tool: str
    allowed: bool
    scope: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class PolicyResolution:
    """Aggregate view of policy decisions for all tools under evaluation."""

    decisions: Mapping[str, PolicyDecision]

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        object.__setattr__(self, "decisions", MappingProxyType(dict(self.decisions)))

    @property
    def allowed_tools(self) -> frozenset[str]:
        return frozenset(tool for tool, decision in self.decisions.items() if decision.allowed)

    @property
    def denied_tools(self) -> Mapping[str, PolicyDecision]:
        denied: MutableMapping[str, PolicyDecision] = {}
        for tool, decision in self.decisions.items():
            if not decision.allowed:
                denied[tool] = decision
        return MappingProxyType(denied)


@dataclass(frozen=True, slots=True)
class PolicyDenial:
    """Structured metadata attached to a policy violation."""

    tool: str
    decision: PolicyDecision
    resolution: PolicyResolution


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Immutable snapshot captured during enforcement."""

    stack_depth: int
    resolution: PolicyResolution

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        object.__setattr__(self, "resolution", self.resolution)


@dataclass(frozen=True, slots=True)
class PolicyTraceEvent:
    """Trace event emitted by the policy engine."""

    event: str
    scope: str
    data: Mapping[str, object]


class PolicyTraceRecorder:
    """Simple recorder that stores emitted trace events for later inspection."""

    def __init__(self) -> None:
        self._events: list[PolicyTraceEvent] = []

    def record(self, event: PolicyTraceEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> Sequence[PolicyTraceEvent]:
        return tuple(self._events)

