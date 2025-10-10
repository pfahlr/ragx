from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ToolDescriptor:
    """Normalized metadata describing a tool in the registry."""

    tool_id: str
    tags: frozenset[str]


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of evaluating a tool against the active policy stack."""

    tool: str
    allowed: bool
    reason: str
    allow_scope: str | None
    deny_scope: str | None
    matched_tags: frozenset[str]


@dataclass(frozen=True)
class PolicyResolution:
    """Aggregate view of an allowlist evaluation."""

    allowed: frozenset[str]
    denied: frozenset[str]
    candidates: frozenset[str]
    decisions: Mapping[str, PolicyDecision]

    def __post_init__(self) -> None:  # pragma: no cover - defensive copy guard.
        object.__setattr__(self, "decisions", MappingProxyType(dict(self.decisions)))


@dataclass(frozen=True)
class PolicyDenial:
    """Structured payload describing why a tool was rejected."""

    tool: str
    reason: str
    scope: str | None


@dataclass(frozen=True)
class PolicySnapshot(PolicyResolution):
    """Immutable snapshot reused for enforcement and diagnostics."""

    denials: tuple[PolicyDenial, ...]


class PolicyViolationError(RuntimeError):
    """Raised when enforcement encounters a disallowed tool."""

    def __init__(self, denial: PolicyDenial) -> None:
        super().__init__(f"Policy violation for tool '{denial.tool}': {denial.reason}")
        self.denial = denial


@dataclass(frozen=True)
class PolicyTraceEvent:
    """Structured trace emitted by policy stack operations."""

    event: str
    scope: str
    data: Mapping[str, object]

