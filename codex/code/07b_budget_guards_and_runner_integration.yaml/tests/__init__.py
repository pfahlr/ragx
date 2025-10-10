"""Test utilities for the budget guards integration suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def load_module(name: str):
    """Dynamically import a module from the feature package regardless of dots in the path."""

    module_name = f"budget_integration.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_ROOT / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Unable to load module '{name}' from {MODULE_ROOT}")
    if "budget_integration" not in sys.modules:
        package = ModuleType("budget_integration")
        package.__path__ = [str(MODULE_ROOT)]  # type: ignore[attr-defined]
        sys.modules["budget_integration"] = package

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

