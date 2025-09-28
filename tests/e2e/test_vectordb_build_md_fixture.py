from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _write(tmp_dir: Path, relative: str, content: str) -> Path:
    path = tmp_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_vectordb_builder_ingests_markdown(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    _write(
        corpus_dir,
        "doc_one.md",
        """title: Doc One
slug: doc-one
summary: Preferred summary from front matter
---
# Ignored Heading

Body content for doc one.
""",
    )

    _write(
        corpus_dir,
        "notes/doc_two.md",
        """# Doc Two Heading

Doc two body paragraph.
""",
    )

    output_dir = tmp_path / "index"

    cmd = [
        "python",
        "-m",
        "ragcore.cli",
        "build",
        "--corpus-dir",
        str(corpus_dir),
        "--accept-format",
        "md",
        "--out",
        str(output_dir),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr

    summary = json.loads(proc.stdout)
    assert summary["documents"] == 2

    docmap_path = Path(summary["docmap_path"])
    assert docmap_path.exists()

    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))
    docs_by_id = {entry["id"]: entry for entry in docmap["documents"]}

    assert "doc-one" in docs_by_id
    doc_one = docs_by_id["doc-one"]
    assert doc_one["metadata"]["title"] == "Doc One"
    assert doc_one["metadata"]["summary"] == "Preferred summary from front matter"
    assert doc_one["metadata"]["source_relpath"] == "doc_one.md"

    fallback_id = "notes/doc_two.md"
    assert fallback_id in docs_by_id
    doc_two = docs_by_id[fallback_id]
    assert doc_two["metadata"]["title"] == "Doc Two Heading"
    assert doc_two["metadata"]["source_relpath"] == fallback_id

    # Text content should be present so downstream embedding steps can operate.
    assert doc_one["text"].startswith("# Ignored Heading")
    assert "Doc two body" in doc_two["text"]
