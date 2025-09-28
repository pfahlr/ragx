from __future__ import annotations
from pathlib import Path
import pytest
from ragcore.ingest.md_parser import parse_markdown

def _write(tmp_path: Path, name: str, contents: str) -> Path:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def test_yaml_front_matter_overrides_base_metadata(tmp_path: Path) -> None:
    doc = _write(
        tmp_path,
        "yaml_doc.md",
        "---\n"
        "title: Doc Title\n"
        "aliases:\n"
        "  - dt\n"
        "---\n"
        "# Heading\n"
        "Body text.\n",
    )

    text, metadata = parse_markdown(doc, base_metadata={"title": "Old", "category": "notes"})

    assert "Heading" in text
    assert metadata["title"] == "Doc Title"
    assert metadata["category"] == "notes"
    assert metadata["aliases"] == ["dt"]


def test_key_value_front_matter_parses_and_overrides(tmp_path: Path) -> None:
    doc = _write(
        tmp_path,
        "kv_doc.md",
        "title: Front Matter Title\n"
        "tags: a, b\n"
        "---\n"
        "Actual content\n",
    )

    text, metadata = parse_markdown(doc, base_metadata={"title": "Fallback", "tags": "c"})

    assert text.strip() == "Actual content"
    assert metadata["title"] == "Front Matter Title"
    assert metadata["tags"] == "a, b"


def test_missing_front_matter_returns_plain_text(tmp_path: Path) -> None:
    doc = _write(tmp_path, "plain.md", "No front matter here\nSecond line\n")

    text, metadata = parse_markdown(doc, base_metadata={"category": "misc"})

    assert text.startswith("No front matter")
    assert metadata == {"category": "misc"}


@pytest.mark.parametrize(
    "header_lines",
    [
        "invalid header\n---\nBody\n",
        "key only\n---\nBody\n",
    ],
)

def test_invalid_header_does_not_raise(tmp_path: Path, header_lines: str) -> None:
    doc = _write(tmp_path, "invalid.md", header_lines)

    text, metadata = parse_markdown(doc)

    assert text.endswith("Body\n")
    # Invalid headers are ignored; metadata remains unchanged (empty dict)
    assert metadata == {}
