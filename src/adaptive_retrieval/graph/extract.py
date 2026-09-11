"""Entity and relation extraction with Claude.

This is the most expensive part of the build - in Microsoft's GraphRAG, graph
extraction is roughly 75% of total index cost - so three things are structural
rather than optional:

* **Content-hash caching.** Re-ingesting an unchanged chunk costs nothing. The
  cache key is the chunk's text, not its ID, so identical text anywhere in the
  corpus is extracted once.
* **The Batches API.** 50% cost, and extraction is not latency-sensitive.
  Results come back in **any order**, so they are keyed by ``custom_id`` and
  never by position.
* **Prompt caching.** The instructions are long and identical every time, the
  chunk is short and varies - so the stable part goes in ``system`` and the
  chunk goes last, because caching is a prefix match.

If extraction cost runs away, the documented fallback is LinearRAG's approach:
a competitive entity graph built with classical NLP and zero LLM tokens.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from adaptive_retrieval.chunking import Chunk
from adaptive_retrieval.graph.store import Triple

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedTriples",
    "ExtractionCache",
    "Extractor",
    "batch_requests",
    "triples_from_payload",
]

EXTRACTION_SYSTEM_PROMPT = """Extract entities and the relations between them from a passage.

Rules:
- Extract only relations stated in the passage. Do not infer, and do not use
  prior knowledge about the entities.
- Entities are proper nouns: people, organisations, places, products.
- Use the entity name exactly as written in the passage. Do not resolve
  pronouns and do not merge variant spellings - a later normalisation step
  handles that.
- The predicate is a short snake_case verb phrase, e.g. signed_by, worked_at,
  acquired, vendor_of.
- If the passage states no relation between two named entities, return an
  empty list. An empty extraction is a valid and common result."""


class ExtractedRelation(BaseModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class ExtractedTriples(BaseModel):
    """The structured output contract for one chunk."""

    relations: list[ExtractedRelation] = Field(default_factory=list)


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class ExtractionCache:
    """On-disk cache keyed by chunk *content*.

    Keyed by text rather than chunk ID so the same passage appearing in two
    documents is extracted once, and so a re-chunk that produces identical text
    under a new ID is free.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, list[dict[str, str]]] = {}
        if self.path.exists():
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, text: str) -> ExtractedTriples | None:
        raw = self._entries.get(_cache_key(text))
        if raw is None:
            return None
        return ExtractedTriples(relations=[ExtractedRelation(**r) for r in raw])

    def put(self, text: str, payload: ExtractedTriples) -> None:
        self._entries[_cache_key(text)] = [r.model_dump() for r in payload.relations]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._entries)


def triples_from_payload(payload: ExtractedTriples, chunk_id: str) -> list[Triple]:
    """Attach the source chunk ID to each extracted relation.

    This is the step that makes the graph a set of pointers: every edge knows
    which chunk it came from, and the chunk text stays in Elasticsearch.
    """
    return [
        Triple(
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            chunk_id=chunk_id,
        )
        for relation in payload.relations
    ]


def batch_requests(
    chunks: Iterable[Chunk], *, model: str, max_tokens: int = 2048
) -> list[dict[str, Any]]:
    """Build Batches API requests, one per chunk.

    ``custom_id`` is the chunk ID: batch results return in arbitrary order and
    must be joined by key, never by position.
    """
    requests: list[dict[str, Any]] = []
    for chunk in chunks:
        requests.append(
            {
                "custom_id": chunk.chunk_id,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    # Stable prefix first, volatile chunk last: prompt caching
                    # is a prefix match.
                    "system": EXTRACTION_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": f"Passage:\n\n{chunk.text}"}],
                },
            }
        )
    return requests


class _MessagesParse(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    @property
    def messages(self) -> _MessagesParse: ...


class Extractor:
    """Synchronous per-chunk extraction, for small corpora and for testing.

    For a full corpus use :func:`batch_requests` with the Batches API - it is
    half the price and extraction has no latency requirement.
    """

    def __init__(
        self,
        client: _Client,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 2048,
        cache: ExtractionCache | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.cache = cache

    def extract(self, chunk: Chunk) -> list[Triple]:
        if self.cache is not None:
            cached = self.cache.get(chunk.text)
            if cached is not None:
                return triples_from_payload(cached, chunk.chunk_id)

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=EXTRACTION_SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": f"Passage:\n\n{chunk.text}"}],
            output_format=ExtractedTriples,
        )
        payload = getattr(response, "parsed_output", None)
        if not isinstance(payload, ExtractedTriples):
            # An unparseable extraction yields no edges for this chunk. That is
            # a recall loss, not a correctness problem: the graph is an
            # additive candidate source, so a missing edge costs a candidate
            # rather than an answer.
            payload = ExtractedTriples()

        if self.cache is not None:
            self.cache.put(chunk.text, payload)
        return triples_from_payload(payload, chunk.chunk_id)
