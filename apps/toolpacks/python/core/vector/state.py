"""Shared deterministic state for the vector.query.search stub."""
from __future__ import annotations

SAMPLE_INDEX = [
    {
        "document_id": "doc-alpha",
        "score": 1.0,
        "metadata": {
            "summary": "Overview of retrieval augmented generation",
            "source": "docs/alpha.md",
        },
    },
    {
        "document_id": "doc-beta",
        "score": 0.85,
        "metadata": {
            "summary": "Vector search configuration cheatsheet",
            "source": "docs/beta.md",
        },
    },
    {
        "document_id": "doc-gamma",
        "score": 0.75,
        "metadata": {
            "summary": "How retries improve resilience",
            "source": "docs/gamma.md",
        },
    },
]
