from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from ragcore.cli import _build_docmap
from ragcore.ingest.scanner import IngestedDocument


def test_docmap_prefers_slug_when_id_missing(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    file_path = corpus_dir / "doc-one.md"
    file_path.write_text("Body text", encoding="utf-8")

    record = IngestedDocument(
        path=file_path,
        text="Body text",
        metadata={"slug": "doc-one"},
        format="md",
    )

    docmap = _build_docmap(corpus_dir, [record])
    doc_entry = docmap["documents"][0]

    assert doc_entry["id"] == "doc-one"
    assert doc_entry["metadata"]["slug"] == "doc-one"


def test_docmap_prefers_metadata_id_over_slug(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    file_path = corpus_dir / "doc-two.md"
    file_path.write_text("Text", encoding="utf-8")

    record = IngestedDocument(
        path=file_path,
        text="Text",
        metadata={"slug": "doc-two", "id": "doc-two-preferred"},
        format="md",
    )

    docmap = _build_docmap(corpus_dir, [record])
    doc_entry = docmap["documents"][0]

    assert doc_entry["id"] == "doc-two-preferred"
    assert doc_entry["metadata"]["id"] == "doc-two-preferred"
