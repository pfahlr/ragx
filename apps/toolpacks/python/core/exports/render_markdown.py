"""Deterministic markdown rendering toolpack implementation."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

_ENV = Environment(undefined=StrictUndefined)


def _front_matter_block(front_matter: Mapping[str, Any] | None) -> str:
    if not front_matter:
        return ""
    dumped = yaml.safe_dump(dict(front_matter), sort_keys=True).strip()
    return f"---\n{dumped}\n---\n\n"


def render_markdown(payload: Mapping[str, Any]) -> dict[str, Any]:
    title = str(payload["title"])
    template = str(payload["template"])
    body = str(payload.get("body", ""))
    front_matter = payload.get("front_matter")

    rendered = _ENV.from_string(template).render(
        title=title,
        body=body,
        front_matter=front_matter or {},
    )
    markdown = f"{_front_matter_block(front_matter)}{rendered.strip()}\n"
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    metadata = {
        "title": title,
        "has_front_matter": bool(front_matter),
        "template_path": payload.get("template_path"),
    }

    return {
        "markdown": markdown,
        "content_hash": content_hash,
        "metadata": metadata,
    }
