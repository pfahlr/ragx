"""Stub SchemaRegistry used by executable specs in Part A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolIOValidators:
    """Container for compiled tool input/output validators."""

    input_validator: Any
    output_validator: Any


class SchemaRegistry:
    """Placeholder registry that will load JSON schemas in Part B."""

    def __init__(self, *, schema_dir: Path) -> None:
        self._schema_dir = Path(schema_dir)

    def load_envelope(self) -> Any:  # pragma: no cover - stub
        """Return the compiled validator for the MCP envelope schema."""

        raise NotImplementedError("SchemaRegistry.load_envelope will be implemented in Part B")

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:  # pragma: no cover - stub
        """Return validators for the given tool's input/output schemas."""

        raise NotImplementedError("SchemaRegistry.load_tool_io will be implemented in Part B")
