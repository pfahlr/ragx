from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from . import md_parser, pdf_parser

SUPPORTED_FORMATS = {"md", "pdf"}


@dataclass(frozen=True)
class DocumentRecord:
    """Normalized representation of an ingested document."""

    source_path: Path
    source_relpath: str
    text: str
    metadata: dict[str, object]


def scan_corpus(
    corpus_dir: Path, accept_formats: Iterable[str] | None = None
) -> list[DocumentRecord]:
    """Scan ``corpus_dir`` collecting documents that match ``accept_formats``."""

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    if accept_formats is None:
        normalized_formats = SUPPORTED_FORMATS
    else:
        normalized_formats = {fmt.lower() for fmt in accept_formats}
        unsupported = normalized_formats - SUPPORTED_FORMATS
        if unsupported:
            raise ValueError(f"Unsupported formats requested: {sorted(unsupported)}")

    parser_map: dict[str, Callable[[Path], tuple[str, dict[str, object]]]] = {
        "md": md_parser.parse_markdown,
        "pdf": pdf_parser.parse_pdf,
    }

    records: list[DocumentRecord] = []
    for fmt in sorted(normalized_formats):
        suffix = f"*.{fmt}"
        for path in sorted(corpus_dir.rglob(suffix)):
            text, metadata = parser_map[fmt](path)
            relpath = path.relative_to(corpus_dir)
            relpath_str = relpath.as_posix()
            doc_metadata: dict[str, object] = {
                "source_path": str(path),
                "source_relpath": relpath_str,
                "source_format": fmt,
            }
            doc_metadata.update(metadata)
            records.append(
                DocumentRecord(
                    source_path=path,
                    source_relpath=relpath_str,
                    text=text,
                    metadata=doc_metadata,
                )
            )

    records.sort(key=lambda record: record.source_relpath)
    return records
