"""Compatibility layer exposing canonical vector DB interfaces."""

from __future__ import annotations

from ragcore.interfaces import (
    Backend,
    Handle,
    IndexSpec,
    SerializedIndex,
    VectorIndexHandle,
)

__all__ = [
    "Backend",
    "Handle",
    "IndexSpec",
    "SerializedIndex",
    "VectorIndexHandle",
]

