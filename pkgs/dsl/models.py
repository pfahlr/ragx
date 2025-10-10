"""Data models used by the DSL policy engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Normalized view of a tool entry from the registry."""

    name: str
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Outcome of evaluating a single tool against the policy stack."""

    tool: str
    allowed: bool
    deciding_scope: str | None
    reasons: tuple[str, ...]
    matched_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyDenial:
    """Structured details describing why a tool was denied."""

    tool: str
    deciding_scope: str | None
    reasons: tuple[str, ...]
    matched_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Immutable snapshot of policy evaluation for a candidate set."""

    allowed: tuple[str, ...]
    denied: Mapping[str, PolicyDenial]
    decisions: Mapping[str, PolicyDecision]
    candidates: tuple[str, ...]

    @classmethod
    def from_dicts(
        cls,
        *,
        decisions: Mapping[str, PolicyDecision],
        denied: Mapping[str, PolicyDenial],
        candidates: tuple[str, ...],
    ) -> PolicySnapshot:
        ordered_allowed = tuple(
            sorted(tool for tool, decision in decisions.items() if decision.allowed)
        )
        proxy_denied = MappingProxyType(dict(denied))
        proxy_decisions = MappingProxyType(dict(decisions))
        return cls(ordered_allowed, proxy_denied, proxy_decisions, candidates)

