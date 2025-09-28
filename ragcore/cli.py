from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .ingest.scanner import DocumentRecord, scan_corpus


def build_command(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus_dir)
    accept_formats: Sequence[str] | None = args.accept_format
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_corpus(corpus_dir, accept_formats=accept_formats)
    docmap_entries = _build_docmap(records)
    accepted_formats = sorted(
        {
            str(record.metadata["source_format"])
            for record in records
            if "source_format" in record.metadata
        }
    )

    docmap_path = output_dir / "docmap.json"
    docmap_path.write_text(
        json.dumps({"documents": docmap_entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "documents": len(docmap_entries),
        "docmap_path": str(docmap_path),
        "formats": accepted_formats,
    }
    print(json.dumps(summary))
    return 0


def _build_docmap(records: Sequence[DocumentRecord]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for index, record in enumerate(records):
        metadata = dict(record.metadata)
        doc_id = metadata.get("id") or metadata.get("slug") or metadata.get("source_relpath")
        if not doc_id:
            doc_id = f"doc-{index:04d}"
        entries.append(
            {
                "id": str(doc_id),
                "text": record.text,
                "metadata": metadata,
            }
        )
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vectordb-builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build vector index artifacts")
    build_parser.add_argument(
        "--corpus-dir",
        required=True,
        help="Directory containing source documents",
    )
    build_parser.add_argument(
        "--accept-format",
        action="append",
        dest="accept_format",
        choices=["pdf", "md"],
        help="Accepted source formats when scanning the corpus",
    )
    build_parser.add_argument("--out", required=True, help="Output directory for index artifacts")
    build_parser.set_defaults(func=build_command)

    args = parser.parse_args(argv)
    command = args.func  # type: ignore[attr-defined]
    return command(args)


if __name__ == "__main__":
    sys.exit(main())
