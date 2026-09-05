"""Deterministic chunking with content-addressed chunk IDs.

Two properties matter here and nothing else does:

1. **Determinism.** The same document must produce byte-identical chunks on
   every run, on every machine. Chunking is a *controlled variable* in the
   benchmark: it is held fixed across all arms, so imperfect sentence
   segmentation costs every arm equally and cannot bias the comparison
   between them. What would bias the comparison is chunking that varies
   between runs.

2. **Stable identity.** Chunk IDs are content hashes, never Elasticsearch
   ``_id`` values. The Neptune graph stores chunk IDs as pointers back into
   Elasticsearch; if those IDs were assigned by Elasticsearch, one reindex
   would silently break every edge in the graph.

Default strategy mirrors Elasticsearch's own ``semantic_text`` default -
``sentence`` strategy, ``max_chunk_size: 250`` words, ``sentence_overlap: 1``
- so that our chunking stays comparable with the built-in behaviour.
Verified against the Inference API reference on 2026-09-05.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "chunk_document",
    "chunk_id_for",
    "split_sentences",
]

# 16 hex characters = 64 bits. For a corpus of n chunks the birthday collision
# probability is about n^2 / 2^65: at 1,000,000 chunks that is ~2.7e-8. The
# full digest is available if a corpus ever justifies it.
CHUNK_ID_HEX_LEN = 16

# ASCII unit separator. Used to delimit the hashed fields so that
# ("doc1", 12) and ("doc11", 2) cannot produce the same pre-image. It cannot
# appear in the source text we hash (guarded in chunk_id_for).
_ID_FIELD_SEP = "\x1f"

# Elasticsearch's documented bounds for the sentence strategy, mirrored so a
# config that is invalid for Elastic is also invalid for us.
_MIN_MAX_WORDS = 20
_ALLOWED_SENTENCE_OVERLAP = (0, 1)

# Tokens that end in a period without ending a sentence. Lower-cased, with
# internal periods preserved ("u.s"), trailing period stripped before lookup.
_ABBREVIATIONS = frozenset(
    {
        # titles
        "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "hon", "sr", "jr",
        # organisations
        "inc", "ltd", "co", "corp", "plc", "llc", "dept", "univ", "assn",
        # latin / editorial
        "vs", "etc", "eg", "e.g", "ie", "i.e", "al", "cf", "viz", "approx",
        # places
        "st", "mt", "ave", "blvd", "u.s", "u.k", "u.n", "e.u",
        # months (dates like "Jan. 5" are common in news text)
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
        # units / misc. "no" (as in "No. 5") is deliberately absent: the English
        # word "no" ends sentences far more often than the abbreviation appears,
        # and treating it as an abbreviation silently merges sentences.
        "vol", "fig", "pp", "est",
    }
)

# Closing quotation marks. A period followed by one of these closes a quoted
# sentence ("He said \"no.\"") and is a boundary regardless of the preceding
# token. Brackets are deliberately excluded: "(approx.)" is mid-sentence.
_CLOSING_QUOTES = frozenset("\"'\u201d\u2019")

# A run of sentence-ending punctuation, optionally followed by closing quotes
# or brackets, that is followed by whitespace or end-of-string.
_BOUNDARY_RE = re.compile("[.!?]+[\"'\u201d\u2019)\\]]*(?=\\s|$)")

# The word immediately preceding a candidate boundary, including any internal
# periods, e.g. "Dr" in "Dr." or "U.S" in "U.S.".
_TRAILING_WORD_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunking parameters. Pinned in ``benchmark.yaml`` and published with results."""

    max_words: int = 250
    sentence_overlap: int = 1

    def __post_init__(self) -> None:
        if self.max_words < _MIN_MAX_WORDS:
            raise ValueError(
                f"max_words must be >= {_MIN_MAX_WORDS} "
                f"(Elasticsearch's floor for the sentence strategy), got {self.max_words}"
            )
        if self.sentence_overlap not in _ALLOWED_SENTENCE_OVERLAP:
            raise ValueError(
                f"sentence_overlap must be one of {_ALLOWED_SENTENCE_OVERLAP} "
                f"(Elasticsearch allows only these), got {self.sentence_overlap}"
            )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexed unit. One Elasticsearch document is created per Chunk."""

    chunk_id: str
    source_id: str
    position: int
    text: str
    char_start: int
    char_end: int
    word_count: int
    #: True when a single sentence exceeded ``max_words`` and was emitted alone
    #: rather than being split mid-sentence.
    oversized: bool


def chunk_id_for(source_id: str, position: int, text: str) -> str:
    """Return the content-addressed ID for a chunk.

    The ID is a function of (source_id, position, text) and nothing else - no
    timestamps, no run IDs, no Elasticsearch state. Re-ingesting unchanged
    content therefore reproduces the same IDs, which is what keeps the graph's
    pointers valid across a reindex.

    ``position`` is included so that a document containing the same text twice
    (boilerplate, repeated disclaimers) yields two distinct chunks rather than
    a collision that would silently drop one of them.
    """
    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")
    if _ID_FIELD_SEP in source_id or _ID_FIELD_SEP in text:
        raise ValueError(
            "source_id and text must not contain the ASCII unit separator (0x1f); "
            "it delimits fields in the chunk ID pre-image"
        )
    payload = f"{source_id}{_ID_FIELD_SEP}{position}{_ID_FIELD_SEP}{text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:CHUNK_ID_HEX_LEN]


def _is_real_boundary(text: str, punct_start: int, matched: str) -> bool:
    """Decide whether the punctuation run ``matched`` at ``punct_start`` ends a sentence."""
    # A closing quote after the punctuation closes a quoted sentence.
    if any(ch in _CLOSING_QUOTES for ch in matched):
        return True

    preceding = _TRAILING_WORD_RE.search(text, 0, punct_start)
    if preceding is None:
        return True
    token = preceding.group(1).rstrip(".").lower()
    if not token:
        return True
    # A single letter before a period is an initial ("J. Smith"), not an end.
    if len(token) == 1:
        return False
    return token not in _ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence spans as ``(start, end)`` character offsets.

    Spans exclude surrounding whitespace and never overlap. Empty spans are
    dropped. This is a deterministic heuristic splitter, not a linguistic
    model: it handles common abbreviations and initials, and it will
    occasionally be wrong on unusual text. That is acceptable because the
    splitter is held constant across every arm of the benchmark.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text):
        if not _is_real_boundary(text, match.start(), match.group(0)):
            continue
        end = match.end()
        segment = text[cursor:end]
        stripped_start = cursor + (len(segment) - len(segment.lstrip()))
        stripped_end = cursor + len(segment.rstrip())
        if stripped_end > stripped_start:
            spans.append((stripped_start, stripped_end))
        cursor = end

    # Trailing text with no terminal punctuation.
    if cursor < len(text):
        tail = text[cursor:]
        stripped_start = cursor + (len(tail) - len(tail.lstrip()))
        stripped_end = cursor + len(tail.rstrip())
        if stripped_end > stripped_start:
            spans.append((stripped_start, stripped_end))

    return spans


def chunk_document(
    source_id: str,
    text: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk one source document into indexable units.

    Sentences are accumulated greedily until the next one would exceed
    ``max_words``; each chunk then carries the last ``sentence_overlap``
    sentences of the previous chunk. A sentence longer than ``max_words`` is
    emitted as its own chunk rather than being split mid-sentence, and is
    flagged ``oversized``.
    """
    cfg = config or ChunkingConfig()
    spans = split_sentences(text)
    if not spans:
        return []

    word_counts = [len(text[start:end].split()) for start, end in spans]
    total = len(spans)

    chunks: list[Chunk] = []
    index = 0
    position = 0

    while index < total:
        end_index = index
        words = 0
        # Always take at least one sentence, so an oversized sentence still
        # makes progress instead of producing an empty chunk forever.
        while end_index < total and (
            end_index == index or words + word_counts[end_index] <= cfg.max_words
        ):
            words += word_counts[end_index]
            end_index += 1

        char_start = spans[index][0]
        char_end = spans[end_index - 1][1]
        chunk_text = text[char_start:char_end]
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(source_id, position, chunk_text),
                source_id=source_id,
                position=position,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                word_count=words,
                oversized=(end_index - index == 1 and words > cfg.max_words),
            )
        )
        position += 1

        if end_index >= total:
            break

        # Step back by the overlap, but never far enough to revisit the
        # sentence we started this chunk on - that would not terminate.
        index = max(end_index - cfg.sentence_overlap, index + 1)

    return chunks
