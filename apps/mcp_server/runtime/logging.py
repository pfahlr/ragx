"""Helper utilities for MCP server structured logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ToolInvocationLogPaths", "resolve_tool_invocation_log_paths"]


@dataclass(frozen=True, slots=True)
class ToolInvocationLogPaths:
    """Filesystem layout for tool invocation logs.

    The contract is specified in ``codex/specs/ragx_master_spec.yaml`` under
    ``structured_logging_contract``. All paths are expressed relative to the
    caller-provided ``root`` directory.
    """

    root: Path

    @property
    def storage_prefix(self) -> Path:
        """Return the directory prefix for rotated JSONL log files."""

        return self.root / "logs" / "mcp_server" / "tool_invocations"

    @property
    def latest_symlink(self) -> Path:
        """Return the canonical symlink pointing at the latest log file."""

        return self.root / "logs" / "mcp_server" / "tool_invocations.latest.jsonl"

    def ensure_directories(self) -> None:
        """Create the storage directories if they do not yet exist."""

        self.storage_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.latest_symlink.parent.mkdir(parents=True, exist_ok=True)


def resolve_tool_invocation_log_paths(root: Path) -> ToolInvocationLogPaths:
    """Return the canonical log layout for a given ``root`` directory."""

    return ToolInvocationLogPaths(root=Path(root))
