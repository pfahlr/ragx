from __future__ import annotations

import hashlib
from typing import Any, Mapping

import yaml
from jinja2 import Environment, StrictUndefined


def render_markdown(payload: Mapping[str, Any]) -> dict[str, Any]:
    title = str(payload["title"]).strip()
    template = str(payload["template"])
    body = str(payload.get("body", ""))
    front_matter = dict(payload.get("front_matter") or {})

    environment = Environment(autoescape=False, undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    template_obj = environment.from_string(template)
    rendered_body = template_obj.render(title=title, body=body, front_matter=front_matter)

    if "title" not in front_matter:
        front_matter["title"] = title

    front_matter_yaml = yaml.safe_dump(front_matter, sort_keys=True).strip()
    markdown = f"---\n{front_matter_yaml}\n---\n\n{rendered_body.strip()}\n"

    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    words = [word for word in rendered_body.split() if word.isalpha()]
    metadata = {
        "title": title,
        "word_count": len(words),
        "front_matter_keys": sorted(front_matter.keys()),
        "summary": rendered_body.strip().splitlines()[0][:160] if rendered_body.strip() else "",
    }

    return {
        "markdown": markdown,
        "content_hash": f"sha256:{content_hash}",
        "metadata": metadata,
        "front_matter": front_matter,
    }
