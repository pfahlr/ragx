from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract text and metadata from a PDF document.

    The implementation prefers :mod:`pypdf` but degrades gracefully with a
    helpful error message if the dependency is missing. This keeps the module
    importable in environments where PDF ingestion is not yet required.
    """

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "PDF parsing requires the 'pypdf' package. Install it to enable PDF ingestion."
        ) from exc

    reader = PdfReader(str(path))
    text_parts: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_parts.append(extracted.rstrip())

    metadata: dict[str, Any] = {
        "title": path.stem,
        "derived_title": path.stem,
    }

    doc_info: Any = reader.metadata or {}
    title = getattr(doc_info, "title", None)
    if title:
        metadata["title"] = title
    for key, value in getattr(doc_info, "items", lambda: [])():
        if not value:
            continue
        normalized_key = str(key).strip("/")
        metadata.setdefault(normalized_key, value)

    text = "\n".join(part for part in text_parts if part).strip()
    return text, metadata
