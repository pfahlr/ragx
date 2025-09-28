"""Helpers to compile the `_ragcore_cpp` pybind11 extension."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Distribution


def build_extension(*, force: bool = False, verbose: bool = False) -> Path:
    """Build the native extension in-place and return the produced artifact path."""

    project_root = Path(__file__).resolve().parents[3]
    source_dir = project_root / "cpp"
    source_file = source_dir / "ragcore_cpp_module.cpp"
    if not source_file.exists():  # pragma: no cover - safety net
        raise FileNotFoundError(f"missing C++ source file: {source_file}")

    ext_modules: Iterable[Pybind11Extension] = [
        Pybind11Extension(
            "ragcore.backends._ragcore_cpp",
            [str(source_file)],
            include_dirs=[numpy.get_include()],
            cxx_std=17,
        )
    ]

    distribution = Distribution({"name": "ragcore-backends-cpp", "ext_modules": ext_modules})
    distribution.package_dir = {"": str(project_root)}

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.inplace = True
    cmd.force = force
    cmd.verbose = verbose
    cmd.run()

    outputs = [Path(path) for path in cmd.get_outputs()]
    for artifact in outputs:
        if artifact.name.startswith("_ragcore_cpp") and artifact.parent.name == "backends":
            return artifact

    if outputs:  # pragma: no cover - fallback selection
        return outputs[0]
    raise RuntimeError("no build outputs produced for ragcore.backends._ragcore_cpp")


__all__ = ["build_extension"]
