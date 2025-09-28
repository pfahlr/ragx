"""CLI entry point for vectordb-builder."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .ingest.scanner import IngestedDocument, scan_corpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vectordb-builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build vector index artifacts from a corpus.")
    build.add_argument("--corpus-dir", type=Path, required=True, help="Path to corpus directory.")
    build.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for build artifacts.",
    )
    build.add_argument(
        "--accept-format",
        dest="accept_format",
        action="append",
        choices=["pdf", "md"],
        help="Accepted source formats when scanning --corpus-dir.",
    )
    build.set_defaults(func=_cmd_build)

    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    formats: Sequence[str] | None = args.accept_format
    documents = scan_corpus(args.corpus_dir, accept_formats=formats)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    docmap = _build_docmap(args.corpus_dir, documents)
    docmap_path = out_dir / "docmap.json"
    docmap_path.write_text(json.dumps(docmap, indent=2, sort_keys=True), encoding="utf-8")

    summary = {"documents": len(documents), "docmap_path": str(docmap_path)}
    print(json.dumps(summary))
    return 0


def _build_docmap(corpus_dir: Path, documents: Sequence[IngestedDocument]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for record in documents:
        metadata = dict(record.metadata)
        doc_id = str(metadata.get("id") or record.path.stem)
        counter = seen.get(doc_id)
        if counter is None:
            seen[doc_id] = 0
        else:
            seen[doc_id] = counter + 1
            doc_id = f"{doc_id}-{seen[doc_id]}"

        try:
            rel_path = record.path.relative_to(corpus_dir).as_posix()
        except ValueError:
            rel_path = record.path.resolve().as_posix()

        entries.append(
            {
                "id": doc_id,
                "path": rel_path,
                "format": record.format,
                "metadata": metadata,
                "text": record.text,
            }
        )

    return {"documents": entries}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
