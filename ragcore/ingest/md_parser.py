"""Markdown ingestion utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

FrontMatter = dict[str, Any]


def parse_markdown(
    path: Path,
    base_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Parse a Markdown document extracting optional front-matter.

    Parameters
    ----------
    path:
        The Markdown file to parse.
    base_metadata:
        Existing metadata extracted from other sources. Values defined in
        front-matter take precedence per the ``markdown_front_matter_precedence``
        open decision.
    """

    raw_text = path.read_text(encoding="utf-8")
    text, front_matter = _split_front_matter(raw_text)

    metadata = dict(base_metadata or {})
    if front_matter:
        metadata.update(front_matter)

    return text, metadata


def _split_front_matter(raw_text: str) -> tuple[str, FrontMatter]:
    text = raw_text.lstrip("\ufeff")

    if text.startswith("---"):
        extracted = _extract_yaml_front_matter(text)
        if extracted is not None:
            return extracted

    extracted = _extract_key_value_front_matter(text)
    if extracted is not None:
        return extracted

    return text, {}


def _extract_yaml_front_matter(text: str) -> tuple[str, FrontMatter] | None:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None

    closing = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing = idx
            break

    if closing is None:
        return None

    block = "".join(lines[1:closing])
    try:
        data = yaml.safe_load(block) or {}
    except yaml.YAMLError:
        data = {}

    if not isinstance(data, dict):
        data = {}

    remainder = "".join(lines[closing + 1 :])
    return remainder, data


def _extract_key_value_front_matter(text: str) -> tuple[str, FrontMatter] | None:
    lines = text.splitlines(keepends=True)
    try:
        delimiter_index = next(idx for idx, line in enumerate(lines) if line.strip() == "---")
    except StopIteration:
        return None

    if delimiter_index == 0:
        return None

    header_lines = lines[:delimiter_index]
    metadata: dict[str, Any] = {}
    for raw_line in header_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            return None
        key, value = stripped.split(":", 1)
        metadata[key.strip()] = value.strip()

    remainder = "".join(lines[delimiter_index + 1 :])
    return remainder, metadata
