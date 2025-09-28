from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _write_markdown(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


def test_yaml_front_matter_overrides_heading(tmp_path: Path) -> None:
    path = _write_markdown(
        tmp_path,
        "front_matter.md",
        """
        title: Override Title
        summary: Example summary
        tags:
          - ingestion
          - markdown
        ---
        # Heading Title

        Body content goes here.
        """,
    )

    from ragcore.ingest.md_parser import parse_markdown  # import inside for TDD ordering

    text, metadata = parse_markdown(path)

    assert text.startswith("# Heading Title")
    assert metadata["title"] == "Override Title"
    assert metadata["summary"] == "Example summary"
    assert metadata["tags"] == ["ingestion", "markdown"]
    assert metadata["derived_title"] == "Heading Title"


def test_header_block_without_nested_yaml(tmp_path: Path) -> None:
    path = _write_markdown(
        tmp_path,
        "header_block.md",
        """
        title: Plain Header Title
        category: research
        owner: ops
        ---
        Content without headings.
        """,
    )

    from ragcore.ingest.md_parser import parse_markdown

    text, metadata = parse_markdown(path)

    assert text.startswith("Content without headings.")
    assert metadata["title"] == "Plain Header Title"
    assert metadata["category"] == "research"
    assert metadata["owner"] == "ops"


def test_title_falls_back_to_first_heading_or_stem(tmp_path: Path) -> None:
    path_with_heading = _write_markdown(
        tmp_path,
        "has_heading.md",
        """
        # Heading Driven Title

        Some content.
        """,
    )

    path_without_heading = _write_markdown(
        tmp_path,
        "no_heading.md",
        """
        Paragraph only without any headings.
        """,
    )

    from ragcore.ingest.md_parser import parse_markdown

    text_heading, metadata_heading = parse_markdown(path_with_heading)
    text_none, metadata_none = parse_markdown(path_without_heading)

    assert text_heading.startswith("# Heading Driven Title")
    assert metadata_heading["title"] == "Heading Driven Title"
    assert metadata_heading["derived_title"] == "Heading Driven Title"

    assert text_none.startswith("Paragraph only")
    assert metadata_none["title"] == "no_heading"
    assert metadata_none["derived_title"] == "no_heading"


if __name__ == "__main__":
    pytest.main([__file__])
