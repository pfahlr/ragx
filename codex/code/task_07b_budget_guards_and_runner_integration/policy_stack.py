"""Simplified PolicyStack implementation for FlowRunner tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

from .trace_emitter import TraceEventEmitter

if False:  # pragma: no cover
    from .flow_runner import FlowNode


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    metadata: Optional[Mapping[str, Any]] = None


class PolicyStack:
    """Evaluates policy rules against nodes and records trace events."""

    def __init__(self, *, rules: Optional[Iterable[Callable[["FlowNode"], Optional[PolicyDecision]]]] = None, emitter: Optional[TraceEventEmitter] = None) -> None:
        self._rules = list(rules or [])
        self.emitter = emitter or TraceEventEmitter()

    def evaluate(self, node: "FlowNode") -> PolicyDecision:
        for rule in self._rules:
            decision = rule(node)
            if decision is not None:
                self.record(node, decision, self.emitter)
                return decision
        decision = PolicyDecision(allowed=True, reason="allow")
        self.record(node, decision, self.emitter)
        return decision

    def record(self, node: "FlowNode", decision: PolicyDecision, emitter: TraceEventEmitter) -> None:
        payload = {
            "node_id": node.node_id,
            "decision": "allow" if decision.allowed else "deny",
            "reason": decision.reason,
        }
        if decision.metadata:
            payload["metadata"] = dict(decision.metadata)
        emitter.emit("policy_decision", scope_type="policy", scope_id=node.node_id, payload=payload)


__all__ = ["PolicyDecision", "PolicyStack"]
