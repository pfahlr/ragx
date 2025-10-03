from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        value = context.get(key)
        if isinstance(value, (str, int, float)):
            return str(value)
        if value is None:
            return ""
        return yaml.safe_dump(value, sort_keys=True, default_flow_style=False).strip()

    return _PLACEHOLDER.sub(replacer, template)


def _front_matter(payload: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(payload.get("frontMatter") or {})
    title = payload.get("title", "")
    if title:
        base.setdefault("title", title)
    return base


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    title = str(payload["title"]).strip()
    template = str(payload["template"]).strip()
    body = str(payload.get("body", ""))

    front_matter = _front_matter(payload)
    context: dict[str, Any] = {"title": title, "body": body}
    for key, value in front_matter.items():
        if isinstance(value, (str, int, float)):
            context.setdefault(key, value)
    rendered = _render_template(template, context)

    fm_yaml = ""
    if front_matter:
        fm_yaml = yaml.safe_dump(front_matter, sort_keys=True, default_flow_style=False).strip()
    front_section = f"---\n{fm_yaml}\n---\n" if fm_yaml else "---\n---\n"

    markdown = f"{front_section}{rendered}\n"
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    return {
        "markdown": markdown,
        "contentHash": content_hash,
        "metadata": {
            "title": title,
            "frontMatter": front_matter,
        },
    }
