from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["SchemaRegistry", "ToolIOValidators", "ValidatorProtocol"]


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol describing the subset of jsonschema validators used in tests."""

    def validate(self, instance: object) -> None:  # pragma: no cover - protocol definition
        ...


@dataclass(frozen=True)
class ToolIOValidators:
    """Bundle of compiled validators for tool input and output schemas."""

    input_validator: ValidatorProtocol
    output_validator: ValidatorProtocol


class SchemaRegistry:
    """Stubbed schema registry pending full implementation in Part B."""

    def __init__(self, *, schema_root: Path) -> None:
        self._schema_root = Path(schema_root)

    def load_envelope(self) -> ValidatorProtocol:
        """Return the compiled envelope validator (stub)."""

        raise NotImplementedError("SchemaRegistry.load_envelope will be implemented in Part B")

    def load_tool_io(self, tool_id: str) -> ToolIOValidators:
        """Return the compiled validators for a given tool (stub)."""

        raise NotImplementedError("SchemaRegistry.load_tool_io will be implemented in Part B")
