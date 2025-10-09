from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from ragcore.cli import _build_docmap
from ragcore.ingest.scanner import IngestedDocument


def test_build_docmap_prefers_slug_over_path(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc_path = corpus_dir / "doc_one.md"
    doc_path.write_text("Example", encoding="utf-8")

    document = IngestedDocument(
        path=doc_path,
        text="Example body",
        metadata={"slug": "doc-one", "title": "Example"},
        format="md",
    )

    result = _build_docmap(corpus_dir, [document])
    assert result["documents"][0]["id"] == "doc-one"
    assert result["documents"][0]["metadata"]["slug"] == "doc-one"
