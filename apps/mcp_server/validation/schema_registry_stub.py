"""Schema registry stub for MCP envelope and tool validation.

This module is introduced as part of task 06cV2A to provide an importable
surface for the executable specs. The real implementation will arrive in the
follow-up task (06cV2B). The current methods intentionally raise
:class:`NotImplementedError` so that the tests remain ``xfail`` until the logic is
provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Validator(Protocol):
    """Minimal protocol for JSON schema validators used in tests."""

    def validate(self, instance: object) -> None: ...


@dataclass(frozen=True)
class ToolIOValidators:
    """Container for per-tool input and output validators."""

    input: Validator
    output: Validator


class SchemaRegistry:
    """Stub registry that will load and cache JSON schema validators."""

    def __init__(self, *, schema_root: Path) -> None:
        self._schema_root = Path(schema_root)

    def load_envelope(self) -> Validator:
        """Return a compiled validator for the envelope schema."""

        msg = (
            "Envelope validator not implemented yet (06cV2A scaffold). "
            "Implement in 06cV2B."
        )
        raise NotImplementedError(msg)

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:
        """Return per-tool input/output validators.

        Args:
            tool_id: Fully qualified tool identifier.
        """

        msg = (
            "Tool IO validators not implemented yet (06cV2A scaffold). "
            "Implement in 06cV2B."
        )
        raise NotImplementedError(msg)


__all__ = ["SchemaRegistry", "ToolIOValidators", "Validator"]
