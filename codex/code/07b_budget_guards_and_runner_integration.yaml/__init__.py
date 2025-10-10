"""Dynamic exports for the budget guard integration package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MODULE_ROOT = Path(__file__).resolve().parent

if "budget_integration" not in sys.modules:
    package = ModuleType("budget_integration")
    package.__path__ = [str(MODULE_ROOT)]  # type: ignore[attr-defined]
    sys.modules["budget_integration"] = package


def _load(name: str) -> ModuleType:
    module_name = f"budget_integration.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, MODULE_ROOT / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load module '{name}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

_budget_models = _load("budget_models")
_budget_manager = _load("budget_manager")
_trace_emitter = _load("trace_emitter")
_flow_runner = _load("flow_runner")

BudgetMode = _budget_models.BudgetMode
BudgetSpec = _budget_models.BudgetSpec
CostSnapshot = _budget_models.CostSnapshot
BudgetManager = _budget_manager.BudgetManager
BudgetBreachError = _budget_manager.BudgetBreachError
TraceEventEmitter = _trace_emitter.TraceEventEmitter
FlowRunner = _flow_runner.FlowRunner

__all__ = [
    "BudgetMode",
    "BudgetSpec",
    "CostSnapshot",
    "BudgetManager",
    "BudgetBreachError",
    "TraceEventEmitter",
    "FlowRunner",
]

