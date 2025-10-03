"""Deterministic markdown rendering toolpack."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

_ENV = Environment(autoescape=False, undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def _render(template: str, context: Mapping[str, Any]) -> str:
    compiled = _ENV.from_string(template)
    return compiled.render(**context)


def _front_matter_block(front_matter: Mapping[str, Any]) -> str:
    if not front_matter:
        return ""
    serialised = yaml.safe_dump(dict(front_matter), sort_keys=True, allow_unicode=True).strip()
    return f"---\n{serialised}\n---\n\n"


def render_markdown(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Render markdown content with optional front matter."""

    title = str(payload["title"])
    template = str(payload["template"])
    body = payload.get("body", "")
    front_matter_raw = payload.get("front_matter") or {}
    if not isinstance(front_matter_raw, Mapping):
        raise ValueError("front_matter must be a mapping")

    context = {"title": title, "body": body}
    extra_context = payload.get("context") or {}
    if isinstance(extra_context, Mapping):
        context.update(extra_context)
    else:
        raise ValueError("context must be a mapping")

    rendered = _render(template, context).rstrip() + "\n"
    front_matter = _front_matter_block(front_matter_raw)
    markdown = f"{front_matter}{rendered}"
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    metadata = {
        "title": title,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "front_matter": dict(front_matter_raw),
    }

    return {
        "markdown": markdown,
        "content_hash": content_hash,
        "metadata": metadata,
    }


__all__ = ["render_markdown"]

