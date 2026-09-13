"""Corpus loading and the ingest pipeline.

The corpus format is deliberately minimal - an id and a body - so the harness
is not welded to MultiHop-RAG. Anyone can point it at their own corpus, which
is the claim the benchmark half of this project is making.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from adaptive_retrieval.chunking import Chunk, ChunkingConfig, chunk_document

__all__ = ["CorpusError", "Document", "chunk_corpus", "load_corpus"]


class CorpusError(ValueError):
    """Raised when a corpus file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    title: str = ""


#: Field names accepted for the document body, in priority order. MultiHop-RAG
#: uses "body"; other corpora use "text" or "content".
_TEXT_FIELDS = ("body", "text", "content", "passage")
_ID_FIELDS = ("id", "_id", "doc_id", "url")


def _extract(record: dict[str, object], candidates: tuple[str, ...]) -> str | None:
    for field_name in candidates:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_corpus(path: str | Path) -> list[Document]:
    """Load a JSONL or JSON corpus.

    Each record needs an id-like field and a body-like field; the accepted
    names cover MultiHop-RAG's shape and the common alternatives. A record
    missing either is an error rather than a skip - silently dropping documents
    would change the corpus without changing the config that describes it.
    """
    corpus_path = Path(path)
    try:
        raw = corpus_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"cannot read corpus {corpus_path}: {exc}") from exc

    records: list[dict[str, object]] = []
    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{corpus_path} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise CorpusError(f"{corpus_path} must contain a JSON array or JSONL")
        records = [r for r in parsed if isinstance(r, dict)]
    else:
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{corpus_path}:{line_number} is not valid JSON: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)

    documents: list[Document] = []
    for index, record in enumerate(records):
        text = _extract(record, _TEXT_FIELDS)
        if text is None:
            raise CorpusError(
                f"{corpus_path} record {index} has no body; expected one of {_TEXT_FIELDS}"
            )
        doc_id = _extract(record, _ID_FIELDS)
        if doc_id is None:
            raise CorpusError(
                f"{corpus_path} record {index} has no id; expected one of {_ID_FIELDS}"
            )
        title = _extract(record, ("title", "headline")) or ""
        documents.append(Document(id=doc_id, text=text, title=title))

    if not documents:
        raise CorpusError(f"{corpus_path} contains no documents")

    # Counter, not list.count in a comprehension: the latter is O(n^2) and
    # this runs over the whole corpus.
    counts = Counter(d.id for d in documents)
    duplicates = sorted(i for i, n in counts.items() if n > 1)
    if duplicates:
        raise CorpusError(f"{corpus_path} has duplicate document ids: {duplicates[:10]}")

    return documents


def chunk_corpus(
    documents: list[Document],
    config: ChunkingConfig | None = None,
) -> Iterator[Chunk]:
    """Chunk every document, detecting cross-document ID collisions.

    Within a document, identical text collapses to one chunk by design. Across
    documents the source_id differs, so two chunks sharing an ID is a genuine
    64-bit hash collision - improbable (~2.7e-8 at a million chunks) but silent
    if unchecked, because the second would overwrite the first in the index.
    """
    # Store a second, independent digest rather than the full text: the
    # collision check must not hold a copy of the corpus in memory.
    seen: dict[str, str] = {}
    for document in documents:
        for chunk in chunk_document(document.id, document.text, config):
            digest = hashlib.blake2b(chunk.text.encode("utf-8"), digest_size=16).hexdigest()
            previous = seen.get(chunk.chunk_id)
            if previous is not None and previous != digest:
                raise CorpusError(
                    f"chunk ID collision on {chunk.chunk_id}: two different passages "
                    f"hash to the same ID. Increase CHUNK_ID_HEX_LEN."
                )
            seen[chunk.chunk_id] = digest
            yield chunk
