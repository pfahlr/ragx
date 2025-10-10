"""Minimal policy stack enforcement used by the FlowRunner tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set


class PolicyViolationError(RuntimeError):
    """Raised when a node attempts to execute a disallowed tool."""

    def __init__(self, node_id: str, tool_ref: str, allowlist: Iterable[str]):
        allow_display = ", ".join(sorted(set(allowlist))) or "<empty>"
        super().__init__(f"Tool '{tool_ref}' not permitted for node '{node_id}' (allow: {allow_display})")
        self.node_id = node_id
        self.tool_ref = tool_ref
        self.allowlist = tuple(sorted(set(allowlist)))


@dataclass
class PolicyFrame:
    allow: Set[str]
    deny: Set[str]


class PolicyStack:
    """Simple allow/deny based policy evaluation stack."""

    def __init__(self, global_allow: Optional[Iterable[str]] = None, global_deny: Optional[Iterable[str]] = None) -> None:
        self._frames = [
            PolicyFrame(allow=set(global_allow or []), deny=set(global_deny or []))
        ]

    def resolve(self, node_id: str, tool_ref: str, node_policy: Optional[dict] = None) -> list[str]:
        frame = self._frames[-1]
        effective = set(frame.allow) if frame.allow else set()
        if not effective:
            effective = {tool_ref}
        effective -= frame.deny

        if node_policy:
            node_allow = set(node_policy.get("allow", []))
            node_deny = set(node_policy.get("deny", []))
            if node_allow:
                effective &= node_allow
            effective -= node_deny

        if tool_ref not in effective:
            raise PolicyViolationError(node_id=node_id, tool_ref=tool_ref, allowlist=effective)
        return sorted(effective)
