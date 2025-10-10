from __future__ import annotations

import importlib.util
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .budget import CostSnapshot
else:
    _MODULE_DIR = pathlib.Path(__file__).resolve().parent

    def _load_budget():
        path = _MODULE_DIR / "budget.py"
        spec = importlib.util.spec_from_file_location("task07b_budget", path)
        assert spec.loader is not None  # pragma: no cover - defensive
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _budget = _load_budget()
    CostSnapshot = _budget.CostSnapshot

__all__ = ["AdapterContext", "AdapterResult", "ToolAdapter"]


@dataclass(frozen=True, slots=True)
class AdapterContext:
    run_id: str
    node_id: str
    loop_id: str
    iteration: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterResult:
    output: object
    cost: CostSnapshot


class ToolAdapter(Protocol):
    """Protocol describing tool adapters consumed by FlowRunner."""

    name: str

    def estimate(self, context: AdapterContext) -> AdapterResult:
        ...

    def execute(self, context: AdapterContext) -> AdapterResult:
        ...
