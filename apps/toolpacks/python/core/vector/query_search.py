from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DOC_FIXTURES = [
    ("sample_article", Path("tests/fixtures/mcp/docs/sample_article.md")),
    ("example_fixture", Path("tests/fixtures/mcp/core_tools/docs/example.md")),
]

_WORKSPACE_ROOT = Path.cwd()


def _canonical_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(_WORKSPACE_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass(frozen=True)
class _Document:
    doc_id: str
    title: str
    content: str
    source_path: str

    @classmethod
    def from_fixture(cls, doc_id: str, path: Path) -> _Document:
        text = path.read_text(encoding="utf-8")
        title_line = text.splitlines()[0].lstrip("# ") if text else doc_id
        return cls(
            doc_id=doc_id,
            title=title_line.strip(),
            content=text,
            source_path=_canonical_path(path),
        )


@dataclass(frozen=True)
class _ScoredHit:
    document: _Document
    score: float
    tie_breaker: float

    def sort_key(self) -> tuple[float, float]:
        return (-self.score, self.tie_breaker)

    def to_hit(self, rank: int) -> dict[str, Any]:
        snippet = self.document.content.splitlines()[1:] or [self.document.content]
        snippet_text = " ".join(snippet)[:240].strip()
        return {
            "id": self.document.doc_id,
            "score": self.score,
            "rank": rank,
            "document": {
                "title": self.document.title,
                "snippet": snippet_text,
                "sourcePath": self.document.source_path,
            },
        }


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_documents() -> list[_Document]:
    docs: list[_Document] = []
    for doc_id, path in _DOC_FIXTURES:
        docs.append(_Document.from_fixture(doc_id, path))
    return docs


_DOCUMENTS = _load_documents()


def _score(query: str, document: _Document) -> float:
    tokens = _tokenise(query)
    doc_tokens = _tokenise(document.content)
    if not tokens:
        return 0.0
    score = 0
    doc_counts: dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1
    for token in tokens:
        score += doc_counts.get(token, 0)
    return float(score)


def _tie_breaker(doc_id: str) -> float:
    seed = os.getenv("RAGX_SEED", "0")
    digest = hashlib.sha256(f"{doc_id}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def run(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload["query"]).strip()
    top_k = int(payload.get("topK", 5))
    top_k = max(1, min(top_k, len(_DOCUMENTS)))

    scored_hits = [
        _ScoredHit(
            document=document,
            score=_score(query, document),
            tie_breaker=_tie_breaker(document.doc_id),
        )
        for document in _DOCUMENTS
    ]
    scored_hits.sort(key=lambda hit: hit.sort_key())

    hits = [
        scored_hit.to_hit(rank)
        for rank, scored_hit in enumerate(scored_hits[:top_k], start=1)
    ]

    return {
        "query": query,
        "hits": hits,
        "total": len(_DOCUMENTS),
    }
