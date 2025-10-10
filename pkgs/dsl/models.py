"""Shared data models for the DSL policy engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Canonical description for a tool entry in the registry."""

    name: str
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Per-tool decision metadata captured during policy resolution."""

    descriptor: ToolDescriptor
    allowed: bool
    reasons: tuple[str, ...]
    granted_by: str | None
    denied_by: str | None
    matched_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyResolution:
    """Aggregated allow/deny information for a set of candidate tools."""

    allowed: frozenset[str]
    denied: Mapping[str, tuple[str, ...]]
    decisions: Mapping[str, PolicyDecision]
    stack_depth: int
    candidates: tuple[str, ...]
    directives: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PolicySnapshot(PolicyResolution):
    """Immutable snapshot returned by :class:`PolicyStack` evaluations."""


@dataclass(frozen=True, slots=True)
class PolicyDenial:
    """Structured representation for a blocked tool enforcement request."""

    tool: str
    reasons: tuple[str, ...]
    decision: PolicyDecision


T = TypeVar("T")


def mapping_proxy(data: Mapping[str, T] | None = None) -> Mapping[str, T]:
    """Return an immutable mapping used by trace payloads and snapshots."""

    return MappingProxyType(dict(data or {}))
