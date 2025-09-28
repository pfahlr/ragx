from __future__ import annotations

import json
from pathlib import Path

from ragcore import registry
from ragcore.cli import main


def setup_module() -> None:
    # Ensure registry is in a clean state before CLI tests run.
    registry._reset_registry()


def test_cli_list_outputs_registered_backends(capsys) -> None:
    exit_code = main(["list"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert any(entry["name"] == "dummy" for entry in payload)


def test_cli_build_with_dummy_backend(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text("# Title\n\nSome content.", encoding="utf-8")

    out_dir = tmp_path / "out"

    exit_code = main(
        [
            "build",
            "--backend",
            "dummy",
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(out_dir),
            "--index-kind",
            "flat",
            "--metric",
            "ip",
            "--dim",
            "2",
        ]
    )

    assert exit_code == 0
    spec_path = out_dir / "index_spec.json"
    docmap_path = out_dir / "docmap.json"
    assert spec_path.exists()
    assert docmap_path.exists()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["backend"] == "dummy"
    assert spec["kind"] == "flat"

    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))
    assert docmap["documents"][0]["path"].endswith("doc1.md")
