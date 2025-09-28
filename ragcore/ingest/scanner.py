"""Directory scanner for corpus ingestion."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .md_parser import parse_markdown
from .pdf_parser import parse_pdf

SUPPORTED_FORMATS = {"md", "pdf"}


@dataclass(frozen=True)
class IngestedDocument:
    path: Path
    text: str
    metadata: dict[str, Any]
    format: str


def scan_corpus(
    corpus_dir: Path,
    accept_formats: Sequence[str] | None = None,
) -> list[IngestedDocument]:
    """Scan a corpus directory and parse supported documents."""

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    requested = set(accept_formats) if accept_formats else set(SUPPORTED_FORMATS)
    unknown = requested - SUPPORTED_FORMATS
    if unknown:
        raise ValueError(f"Unsupported formats requested: {sorted(unknown)}")

    documents: list[IngestedDocument] = []
    for file_path in _iter_files(corpus_dir):
        suffix = file_path.suffix.lower().lstrip(".")
        if suffix not in requested:
            continue

        parser = _get_parser(suffix)
        text, metadata = parser(file_path, base_metadata={"format": suffix})
        merged = dict(metadata)
        merged.setdefault("format", suffix)

        documents.append(
            IngestedDocument(
                path=file_path,
                text=text,
                metadata=merged,
                format=suffix,
            )
        )

    documents.sort(key=lambda doc: doc.path.relative_to(corpus_dir).as_posix())
    return documents


def _iter_files(corpus_dir: Path) -> Iterator[Path]:
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file():
            yield path


def _get_parser(fmt: str):
    if fmt == "md":
        return parse_markdown
    if fmt == "pdf":
        return parse_pdf
    raise ValueError(f"Unsupported format: {fmt}")
