"""Ingestion helpers for vector database corpus preparation."""

from .scanner import DocumentRecord, scan_corpus

__all__ = [
    "DocumentRecord",
    "scan_corpus",
]
