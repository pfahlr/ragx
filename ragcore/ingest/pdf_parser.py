"""PDF ingestion utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PdfReader = None


def parse_pdf(
    path: Path,
    base_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Parse a PDF document, extracting text and merging metadata."""

    if PdfReader is None:
        raise RuntimeError(
            "pypdf is required for PDF ingestion; install ragx[ingest] "
            "or add pypdf to your environment."
        )

    reader = PdfReader(str(path))

    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        cleaned = extracted.strip()
        if cleaned:
            text_parts.append(cleaned)

    text = "\n".join(text_parts)

    metadata = dict(base_metadata or {})
    pdf_meta = getattr(reader, "metadata", None) or {}
    for key, value in pdf_meta.items():
        if not key:
            continue
        key_str = key.lstrip("/") if isinstance(key, str) else str(key)
        metadata.setdefault(key_str, value)

    return text, metadata
