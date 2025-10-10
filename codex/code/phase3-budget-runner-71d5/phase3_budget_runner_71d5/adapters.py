from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .budgeting import CostSnapshot

__all__ = ["ToolAdapter", "ToolExecutionResult"]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    result: dict[str, object]
    cost: CostSnapshot
    stop_loop: bool = False


class ToolAdapter(Protocol):
    """Protocol implemented by concrete tool adapters."""

    name: str

    def estimate(self, node: object, context: dict[str, object] | None = None) -> CostSnapshot:
        ...

    def execute(self, node: object, context: dict[str, object] | None = None) -> ToolExecutionResult:
        ...
