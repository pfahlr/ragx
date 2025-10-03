from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import yaml
from jinja2 import Template


def render_markdown(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Render markdown content using a Jinja2 template with YAML front matter."""

    title = str(payload.get("title", ""))
    template = Template(str(payload.get("template", "")))
    body = str(payload.get("body", ""))
    front_matter = payload.get("front_matter") or {}

    rendered_body = template.render(title=title, body=body, front_matter=front_matter)
    front_matter_block = yaml.safe_dump(front_matter, sort_keys=True, allow_unicode=True).strip()
    if front_matter_block:
        markdown = f"---\n{front_matter_block}\n---\n{rendered_body}\n"
    else:
        markdown = rendered_body

    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    metadata = {
        "title": title,
        "content_type": "text/markdown",
        "hash_algorithm": "sha256",
    }

    return {
        "markdown": markdown,
        "content_hash": content_hash,
        "metadata": metadata,
    }
