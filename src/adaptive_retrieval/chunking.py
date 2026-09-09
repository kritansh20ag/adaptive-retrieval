"""Deterministic chunking with content-addressed chunk IDs.

Two properties matter here and nothing else does:

1. **Determinism.** The same document must produce byte-identical chunks on
   every run, on every machine. Chunking is a *controlled variable*: it is held
   fixed across all arms, so imperfect sentence segmentation costs every arm
   equally. What would bias the comparison is chunking that varies between
   runs, or that represents a chunk differently to different retrievers.

2. **Stable identity.** Chunk IDs are content hashes, never Elasticsearch
   ``_id`` values. The Neptune graph stores chunk IDs as pointers back into
   Elasticsearch; if those IDs were assigned by Elasticsearch, one reindex
   would break every edge in the graph.

Three decisions worth reading before changing anything here
-----------------------------------------------------------
**The ID hashes ``(source_id, text)`` and NOT ``position``.** Including
position made the ID a function of the document's *prefix*: deleting an early
sentence reissued the IDs of every later chunk even when their text was
byte-identical, dangling every graph edge pointing at them. That is the exact
failure this design exists to prevent, arriving through a different door. The
cost of dropping position is that identical text repeated within one document
collapses to a single chunk - which is correct: it is one passage, indexed
once, and both occurrences legitimately point at it.

**Overlap defaults to 0.** With overlap on, every sentence lives in two chunks
with two different IDs. That makes ``gold_chunks`` ambiguous (a retriever
returning the *other* chunk containing the answer scores as a miss, and the
different retrievers break that tie differently, so the penalty is
arm-dependent). Worse, it duplicates every entity in the overlapped sentence
into the knowledge graph, inflating node ``degree`` - which is the routing
signal A6's entire claim rests on. Overlap is available, but it is off, and
turning it on requires resolving gold labels to a *set* of acceptable chunk IDs
first.

**Text is normalised before it is hashed.** CRLF vs LF, or a stray NBSP from a
scraped page, would otherwise reissue every ID in the corpus.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "chunk_document",
    "chunk_id_for",
    "normalise_text",
    "split_sentences",
]

# 16 hex characters = 64 bits. For n chunks the probability of at least one
# collision is about n^2 / 2^65: at 1,000,000 chunks that is ~2.7e-8. Small,
# but not zero - so chunk_document also detects collisions rather than trusting
# the arithmetic.
CHUNK_ID_HEX_LEN = 16

# ASCII unit separator, delimiting the hashed fields so ("doc1", "2x") and
# ("doc12", "x") cannot share a pre-image.
_ID_FIELD_SEP = "\x1f"

# Elasticsearch's documented bounds for the sentence strategy, mirrored so a
# config invalid for Elastic is also invalid for us.
_MIN_MAX_WORDS = 20
_ALLOWED_SENTENCE_OVERLAP = (0, 1)

# A word is never this long. Bounding the lookback turns the per-boundary
# abbreviation check from O(document) into O(1); without it the splitter is
# quadratic and a 10 MB document takes hours.
_MAX_LOOKBACK = 64

# Tokens that end in a period without ending a sentence. Lower-cased, internal
# periods preserved ("u.s"), trailing period stripped before lookup.
_ABBREVIATIONS = frozenset(
    {
        # titles
        "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "hon", "sr", "jr",
        "gov", "sen", "rep", "gen", "adm", "lt", "col", "sgt", "capt",
        "messrs", "bros",
        # organisations
        "inc", "ltd", "co", "corp", "plc", "llc", "dept", "univ", "assn",
        # latin / editorial
        "vs", "etc", "eg", "e.g", "ie", "i.e", "al", "cf", "viz", "approx",
        # qualifications and times - very high frequency in news copy
        "a.m", "p.m", "ph.d", "m.d", "b.a", "m.a", "j.d", "d.c",
        # places
        "st", "mt", "ave", "blvd", "u.s", "u.k", "u.n", "e.u",
        "calif", "mass", "fla", "tenn", "conn", "penn", "wash",
        # months (dates like "Jan. 5" are common in news text).
        # "may" is deliberately absent: it is a common word.
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
        # units / misc. "no" (as in "No. 5") is deliberately absent: the English
        # word ends sentences far more often than the abbreviation appears.
        "vol", "fig", "pp", "est",
    }
)

_CLOSING_QUOTES = frozenset("\"'\u201d\u2019")
_CLOSING_BRACKETS = frozenset(")]")
_TRAILING_CHARS = "".join(sorted(_CLOSING_QUOTES | _CLOSING_BRACKETS))

# Sentence-ending punctuation. Two alternatives, because the scripts differ:
# a Latin terminator must be followed by whitespace or end-of-string (so "3.14"
# and "example.com" are not boundaries), whereas CJK and Arabic/Urdu text does
# not put a space after the terminator, so requiring one would segment nothing.
_LATIN_TERMINATORS = ".!?"
# U+3002 ideographic full stop, U+FF01/FF1F fullwidth ! and ?,
# U+06D4 Arabic full stop, U+061F Arabic question mark.
_WIDE_TERMINATORS = "\u3002\uff01\uff1f\u06d4\u061f"
_BOUNDARY_RE = re.compile(
    f"[{re.escape(_LATIN_TERMINATORS)}]+[{re.escape(_TRAILING_CHARS)}]*(?=\\s|$)"
    f"|[{re.escape(_WIDE_TERMINATORS)}]+[{re.escape(_TRAILING_CHARS)}]*"
)

# The word immediately preceding a candidate boundary. Unicode-aware: an
# ASCII-only class made "Kraków." merge (it saw the trailing "w" as an initial)
# while "Łódź." split, which is the same construction behaving two ways.
_TRAILING_WORD_RE = re.compile(r"([^\W\d_][^\W\d_.]*)$")

# Three or more periods is an ellipsis, not a terminator.
_ELLIPSIS_RE = re.compile(r"\.{3,}$")


def normalise_text(text: str) -> str:
    """Canonicalise text before it is chunked or hashed.

    Without this, a re-download of the corpus, a git checkout with
    ``core.autocrlf``, or an editor's line-ending default silently reissues
    every chunk ID and dangles every graph pointer.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunking parameters. Pinned in ``benchmark.yaml`` and published with results."""

    max_words: int = 250
    #: Off by default - see the module docstring. Turning it on requires
    #: resolving gold labels to a set of acceptable chunk IDs first.
    sentence_overlap: int = 0
    #: Hard character ceiling, independent of the word budget. Two reasons:
    #: scripts without spaces have a word count of 1 no matter how long they
    #: are, and an over-long chunk is truncated by the embedding models at
    #: ~512 tokens while BM25 still scores the whole thing - which would make
    #: A1 and A2 see different amounts of the same chunk.
    max_chars: int = 2000

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
        if self.max_chars < 100:
            raise ValueError(f"max_chars must be >= 100, got {self.max_chars}")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexed unit. One Elasticsearch document is created per Chunk."""

    chunk_id: str
    source_id: str
    #: Ordinal within the document. Metadata only - deliberately NOT hashed.
    position: int
    text: str
    char_start: int
    char_end: int
    word_count: int
    #: True when the chunk had to be emitted despite exceeding a budget: a
    #: single sentence longer than ``max_words``, or a span hard-split at
    #: ``max_chars``.
    oversized: bool


def chunk_id_for(source_id: str, text: str) -> str:
    """Return the content-addressed ID for a chunk.

    A function of ``(source_id, text)`` and nothing else - no position, no
    timestamps, no run IDs, no Elasticsearch state. Re-ingesting unchanged
    content reproduces the same IDs even if the surrounding document changed,
    which is what keeps the graph's pointers valid.
    """
    if _ID_FIELD_SEP in source_id or _ID_FIELD_SEP in text:
        raise ValueError(
            "source_id and text must not contain the ASCII unit separator (0x1f); "
            "it delimits fields in the chunk ID pre-image"
        )
    payload = f"{source_id}{_ID_FIELD_SEP}{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CHUNK_ID_HEX_LEN]


def _is_real_boundary(text: str, punct_start: int, matched: str) -> bool:
    """Decide whether the punctuation run ``matched`` at ``punct_start`` ends a sentence."""
    # An ellipsis is not a terminator.
    if _ELLIPSIS_RE.match(matched.rstrip(_TRAILING_CHARS)):
        return False

    # The abbreviation check runs FIRST. Checking the closing quote first made
    # quoted abbreviations - '"the U.S." policy', "'Dr.' Smith" - shatter into
    # fragments, and those are everywhere in news copy.
    preceding = _TRAILING_WORD_RE.search(text, max(0, punct_start - _MAX_LOOKBACK), punct_start)
    if preceding is not None:
        token = preceding.group(1).rstrip(".").lower()
        # A single letter before a period is an initial ("J. Smith").
        if len(token) == 1 or token in _ABBREVIATIONS:
            return False
    return True


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence spans as ``(start, end)`` character offsets.

    Spans exclude surrounding whitespace, never overlap, and together cover
    every non-whitespace character of the input. This is a deterministic
    heuristic, not a linguistic model; it is held constant across every arm.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text):
        if not _is_real_boundary(text, match.start(), match.group(0)):
            continue
        end = match.end()
        segment = text[cursor:end]
        start = cursor + (len(segment) - len(segment.lstrip()))
        stop = cursor + len(segment.rstrip())
        if stop > start:
            spans.append((start, stop))
        cursor = end

    if cursor < len(text):
        tail = text[cursor:]
        start = cursor + (len(tail) - len(tail.lstrip()))
        stop = cursor + len(tail.rstrip())
        if stop > start:
            spans.append((start, stop))

    return spans


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split an over-long span at whitespace, never exceeding ``max_chars``.

    Without this, a single enormous sentence (or a script with no spaces) is
    indexed whole: BM25 scores the entire text while the embedding models
    silently truncate at their context window, so A1 sees the whole chunk and
    A2/A3 see a prefix of it.
    """
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_chars:
        window_end = cursor + max_chars
        split_at = text.rfind(" ", cursor, window_end)
        # No whitespace to split on (CJK, a URL, a hash) - cut mid-token.
        if split_at <= cursor:
            split_at = window_end
        pieces.append((cursor, split_at))
        cursor = split_at
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if end > cursor:
        pieces.append((cursor, end))
    return pieces


def chunk_document(
    source_id: str,
    text: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk one source document into indexable units.

    Sentences accumulate greedily until the next would exceed ``max_words``;
    any span still exceeding ``max_chars`` is hard-split. Chunks whose text is
    identical within a document collapse to one - the ID is content-addressed,
    and two identical passages are one passage.
    """
    cfg = config or ChunkingConfig()
    text = normalise_text(text)
    spans = split_sentences(text)
    if not spans:
        return []

    word_counts = [len(text[start:end].split()) for start, end in spans]
    total = len(spans)

    #: (char_start, char_end, oversized) before de-duplication.
    windows: list[tuple[int, int, bool]] = []
    index = 0
    previous_end_index = -1

    while index < total:
        end_index = index
        words = 0
        while end_index < total and (
            end_index == index or words + word_counts[end_index] <= cfg.max_words
        ):
            words += word_counts[end_index]
            end_index += 1

        # With overlap on, a window that consumes no new sentence would be a
        # strict subset of its predecessor - an extra ES document, an extra
        # embedding, and duplicated entities in the graph, for no new content.
        if end_index <= previous_end_index:
            index = max(end_index, index + 1)
            continue
        previous_end_index = end_index

        char_start = spans[index][0]
        char_end = spans[end_index - 1][1]
        single_sentence_over_budget = end_index - index == 1 and words > cfg.max_words

        for piece_start, piece_end in _hard_split(text, char_start, char_end, cfg.max_chars):
            was_split = (piece_start, piece_end) != (char_start, char_end)
            windows.append((piece_start, piece_end, single_sentence_over_budget or was_split))

        if end_index >= total:
            break
        index = max(end_index - cfg.sentence_overlap, index + 1)

    chunks: list[Chunk] = []
    seen: dict[str, str] = {}
    position = 0
    for char_start, char_end, oversized in windows:
        chunk_text = text[char_start:char_end]
        chunk_id = chunk_id_for(source_id, chunk_text)
        if chunk_id in seen:
            # Identical text within one document is one passage. A differing
            # text under the same ID is a genuine hash collision and must be
            # loud, not silently overwritten.
            if seen[chunk_id] != chunk_text:
                raise ValueError(
                    f"chunk ID collision in {source_id!r}: {chunk_id} maps to two "
                    f"different passages. Increase CHUNK_ID_HEX_LEN."
                )
            continue
        seen[chunk_id] = chunk_text
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_id=source_id,
                position=position,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                word_count=len(chunk_text.split()),
                oversized=oversized,
            )
        )
        position += 1

    return chunks
