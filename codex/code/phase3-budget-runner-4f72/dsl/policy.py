"""Minimal PolicyStack used by the sandbox runner.

The behaviour mirrors the allowlist enforcement path introduced during task 07a
and emits explicit `policy_resolved` events to satisfy POSTEXECUTION
requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from .trace import TraceEventEmitter


class PolicyViolation(RuntimeError):
    """Raised when a node violates policy constraints."""

    def __init__(self, node_id: str, adapter: str, reason: str) -> None:
        super().__init__(f"Policy violation at {node_id} for adapter {adapter}: {reason}")
        self.node_id = node_id
        self.adapter = adapter
        self.reason = reason


@dataclass
class PolicyStack:
    emitter: TraceEventEmitter
    allowlist: Set[str] = field(default_factory=set)

    def check(self, run_id: str, node_id: str, adapter_name: str) -> None:
        """Validate the adapter against the allowlist and emit `policy_push`."""
        self.emitter.policy_push(node_id, adapter_name, run_id)
        if self.allowlist and adapter_name not in self.allowlist:
            reason = "adapter not allowed"
            self.emitter.policy_violation(node_id, adapter_name, run_id, reason)
            raise PolicyViolation(node_id, adapter_name, reason)

    def resolved(self, run_id: str, node_id: str, adapter_name: str) -> None:
        self.emitter.policy_resolved(node_id, adapter_name, run_id)

    def extend_allowlist(self, adapters: Iterable[str]) -> None:
        self.allowlist.update(adapters)
