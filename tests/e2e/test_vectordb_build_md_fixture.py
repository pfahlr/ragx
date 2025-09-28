from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_vectordb_build_md_fixture(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "doc1.md").write_text(
        "---\n"
        "id: doc-one\n"
        "title: Document One\n"
        "category: reference\n"
        "---\n"
        "# Document One\n"
        "Body text here.\n",
        encoding="utf-8",
    )

    (corpus / "doc2.md").write_text(
        "title: Document Two\n"
        "tags: t1, t2\n"
        "---\n"
        "Document two content.\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "-m",
        "ragcore.cli",
        "build",
        "--corpus-dir",
        str(corpus),
        "--out",
        str(out_dir),
        "--accept-format",
        "md",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    docmap_path = out_dir / "docmap.json"
    assert docmap_path.exists(), "docmap.json should be written"

    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))
    documents = {entry["id"]: entry for entry in docmap["documents"]}

    doc_one = documents["doc-one"]
    assert doc_one["metadata"]["title"] == "Document One"
    assert doc_one["metadata"]["category"] == "reference"

    doc_two = documents["doc2"]
    assert doc_two["metadata"]["title"] == "Document Two"
    assert doc_two["metadata"]["tags"] == "t1, t2"

    assert doc_one["path"].endswith("doc1.md")
    assert doc_two["path"].endswith("doc2.md")
