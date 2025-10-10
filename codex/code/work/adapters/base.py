"""Protocol definitions for tool adapters used by FlowRunner."""

from __future__ import annotations

from typing import Mapping, Protocol, Tuple

from ..budget.models import CostSnapshot

__all__ = ["ToolAdapter"]


class ToolAdapter(Protocol):
    """Adapter contract used by FlowRunner for deterministic tooling."""

    def estimate_cost(self, inputs: Mapping[str, object]) -> CostSnapshot:
        """Return a cheap preview of the minimal expected cost for the call."""

    def execute(self, inputs: Mapping[str, object]) -> Tuple[Mapping[str, object], CostSnapshot]:
        """Execute the tool and return structured outputs plus actual spend."""
