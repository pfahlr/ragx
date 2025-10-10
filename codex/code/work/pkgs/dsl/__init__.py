"""Workspace convenience exports mirroring :mod:`pkgs.dsl`."""

from pkgs import dsl as _dsl  # type: ignore[attr-defined]
from pkgs.dsl import *  # noqa: F401,F403

__all__ = _dsl.__all__  # type: ignore[attr-defined]
