"""Lightweight PolicyStack implementation for the sandbox runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .trace import TraceEventEmitter


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


class PolicyStack:
    """Minimal stack that emits trace events around policy evaluation."""

    def __init__(
        self,
        trace_emitter: TraceEventEmitter,
        allowlist: Sequence[str] | None = None,
        denylist: Sequence[str] | None = None,
    ) -> None:
        self._trace = trace_emitter
        self._allow = set(allowlist or [])
        self._deny = set(denylist or [])
        self._stack: List[str] = []

    def evaluate(self, node_id: str, context: dict) -> PolicyDecision:
        self._trace.emit_policy_push(node_id)
        if self._deny and node_id in self._deny:
            decision = PolicyDecision(allowed=False, reason="denied_by_policy")
            self._trace.emit_policy_violation(node_id, decision.reason)
        elif self._allow and node_id not in self._allow:
            decision = PolicyDecision(allowed=False, reason="not_allowlisted")
            self._trace.emit_policy_violation(node_id, decision.reason)
        else:
            decision = PolicyDecision(allowed=True, reason=None)
            self._trace.emit_policy_resolved(node_id)
        self._stack.append(node_id)
        return decision

    def complete(self, node_id: str) -> None:
        if not self._stack or self._stack[-1] != node_id:
            raise RuntimeError(f"Policy stack out of order for node {node_id}")
        self._stack.pop()
        self._trace.emit_policy_pop(node_id)

    @property
    def stack(self) -> Iterable[str]:  # pragma: no cover - inspection helper
        return tuple(self._stack)
