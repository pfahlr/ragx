from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


class MarkdownFrontMatterError(RuntimeError):
    """Raised when front-matter parsing fails."""


def parse_markdown(path: Path) -> tuple[str, dict[str, Any]]:
    """Parse a Markdown document returning clean text and metadata.

    The parser looks for a front-matter block preceding the first line that only
    contains ``---``. When present, the block is treated as YAML. Front-matter
    keys override metadata derived from the document itself, respecting the
    product decision ``front_matter_overrides`` documented in the spec.
    """

    raw = path.read_text(encoding="utf-8")
    front_matter_text, body_lines = _split_front_matter(raw)
    body_text = "\n".join(body_lines).lstrip("\n")

    derived_title = _derive_title(body_lines, path)
    metadata: dict[str, Any] = {
        "title": derived_title,
        "derived_title": derived_title,
        "source_name": path.name,
        "source_stem": path.stem,
    }

    if front_matter_text is not None:
        front_meta = _parse_front_matter(front_matter_text, path)
        metadata.update(front_meta)
        metadata["title"] = front_meta.get("title", metadata.get("title"))
        metadata["derived_title"] = derived_title

    return body_text, metadata


def _split_front_matter(raw: str) -> tuple[str | None, list[str]]:
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "---":
            front_lines = lines[:index]
            body_lines = lines[index + 1 :]
            front_text = "\n".join(front_lines).strip()
            if not front_text:
                return None, lines
            return front_text, body_lines
    return None, lines


def _parse_front_matter(front_text: str, path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(front_text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise MarkdownFrontMatterError(
            f"Failed to parse front matter in {path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):  # pragma: no cover - spec guard
        raise MarkdownFrontMatterError(
            f"Front matter in {path} must yield a mapping, got {type(loaded).__name__}"
        )
    return loaded


def _derive_title(lines: Iterable[str], path: Path) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate:
                return candidate
    return path.stem
