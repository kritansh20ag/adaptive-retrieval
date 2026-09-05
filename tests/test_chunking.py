from __future__ import annotations

from itertools import pairwise

import pytest

from adaptive_retrieval.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_document,
    chunk_id_for,
    split_sentences,
)

# --------------------------------------------------------------------------
# chunk identity - the property the whole graph design rests on
# --------------------------------------------------------------------------


def test_chunk_id_is_deterministic() -> None:
    a = chunk_id_for("doc1", 0, "Hello world.")
    b = chunk_id_for("doc1", 0, "Hello world.")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_chunk_id_changes_with_each_field() -> None:
    base = chunk_id_for("doc1", 0, "Hello world.")
    assert chunk_id_for("doc2", 0, "Hello world.") != base
    assert chunk_id_for("doc1", 1, "Hello world.") != base
    assert chunk_id_for("doc1", 0, "Hello world!") != base


def test_chunk_id_field_boundaries_are_unambiguous() -> None:
    """("doc1", 12) and ("doc11", 2) must not share a pre-image."""
    assert chunk_id_for("doc1", 12, "x") != chunk_id_for("doc11", 2, "x")


def test_chunk_id_rejects_separator_injection() -> None:
    with pytest.raises(ValueError, match="unit separator"):
        chunk_id_for("doc\x1f1", 0, "text")
    with pytest.raises(ValueError, match="unit separator"):
        chunk_id_for("doc1", 0, "te\x1fxt")


def test_chunk_id_rejects_negative_position() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        chunk_id_for("doc1", -1, "text")


def test_identical_text_at_different_positions_gets_distinct_ids() -> None:
    """Repeated boilerplate must not collide into one chunk."""
    assert chunk_id_for("doc1", 0, "Legal notice.") != chunk_id_for("doc1", 5, "Legal notice.")


# --------------------------------------------------------------------------
# sentence splitting
# --------------------------------------------------------------------------


def _texts(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]


def test_splits_on_terminal_punctuation() -> None:
    assert _texts("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_does_not_split_on_abbreviations() -> None:
    assert _texts("Dr. Mehta signed it. He left.") == ["Dr. Mehta signed it.", "He left."]
    assert _texts("Acme Inc. filed today.") == ["Acme Inc. filed today."]
    assert _texts("Reported on Jan. 5 by staff.") == ["Reported on Jan. 5 by staff."]


def test_does_not_split_on_initials() -> None:
    assert _texts("R. Mehta signed it.") == ["R. Mehta signed it."]


def test_handles_closing_quotes_and_brackets() -> None:
    assert _texts('He said "no." She left.') == ['He said "no."', "She left."]


def test_bracketed_abbreviation_does_not_split() -> None:
    """Closing brackets are not treated like closing quotes - "(approx.)" is mid-sentence."""
    assert _texts("See the report (approx. 40 pages) for detail.") == [
        "See the report (approx. 40 pages) for detail."
    ]


def test_keeps_trailing_text_without_punctuation() -> None:
    assert _texts("First one. Dangling tail") == ["First one.", "Dangling tail"]


def test_spans_index_back_into_the_original_text() -> None:
    text = "  First one.   Second one.  "
    for start, end in split_sentences(text):
        segment = text[start:end]
        assert segment == segment.strip()
        assert segment


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_input_yields_no_sentences(text: str) -> None:
    assert split_sentences(text) == []


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def _doc(sentence_count: int, words_each: int = 10) -> str:
    return " ".join(
        " ".join(f"w{i}s{j}" for j in range(words_each - 1)) + f" end{i}."
        for i in range(sentence_count)
    )


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("doc1", "") == []
    assert chunk_document("doc1", "    ") == []


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_document("doc1", "One sentence only.")
    assert len(chunks) == 1
    assert chunks[0].position == 0
    assert chunks[0].source_id == "doc1"
    assert chunks[0].text == "One sentence only."
    assert not chunks[0].oversized


def test_respects_max_words_budget() -> None:
    cfg = ChunkingConfig(max_words=25, sentence_overlap=0)
    chunks = chunk_document("doc1", _doc(10, words_each=10), cfg)
    assert len(chunks) > 1
    for chunk in chunks:
        # Only an oversized single sentence may exceed the budget.
        assert chunk.word_count <= cfg.max_words or chunk.oversized


def test_positions_are_sequential_from_zero() -> None:
    chunks = chunk_document("doc1", _doc(12), ChunkingConfig(max_words=25))
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_the_previous_sentence() -> None:
    text = _doc(6, words_each=10)
    with_overlap = chunk_document("doc1", text, ChunkingConfig(max_words=25, sentence_overlap=1))
    without = chunk_document("doc1", text, ChunkingConfig(max_words=25, sentence_overlap=0))

    assert len(with_overlap) >= len(without)
    # Each subsequent chunk must start at or before where the previous ended.
    for previous, current in pairwise(with_overlap):
        assert current.char_start < previous.char_end


def test_no_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_document("doc1", _doc(8), ChunkingConfig(max_words=25, sentence_overlap=0))
    for previous, current in pairwise(chunks):
        assert current.char_start >= previous.char_end


def test_oversized_sentence_is_emitted_alone_and_flagged() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(120)) + "."
    text = f"Short one. {long_sentence} Short two."
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=30, sentence_overlap=0))

    oversized = [c for c in chunks if c.oversized]
    assert len(oversized) == 1
    assert oversized[0].word_count > 30


def test_terminates_when_every_sentence_is_oversized() -> None:
    """Overlap must never step back onto the sentence a chunk started on."""
    long_sentence = " ".join(f"w{i}" for i in range(60)) + "."
    text = " ".join([long_sentence] * 5)
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=20, sentence_overlap=1))
    assert len(chunks) == 5
    assert all(c.oversized for c in chunks)


def test_chunking_is_deterministic_across_calls() -> None:
    text = _doc(20)
    first = chunk_document("doc1", text)
    second = chunk_document("doc1", text)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_text_matches_its_own_span() -> None:
    text = _doc(15)
    for chunk in chunk_document("doc1", text, ChunkingConfig(max_words=30)):
        assert chunk.text == text[chunk.char_start : chunk.char_end]


def test_chunk_ids_match_their_content() -> None:
    for chunk in chunk_document("doc1", _doc(15), ChunkingConfig(max_words=30)):
        assert chunk.chunk_id == chunk_id_for(chunk.source_id, chunk.position, chunk.text)


def test_word_count_matches_text() -> None:
    for chunk in chunk_document("doc1", _doc(15), ChunkingConfig(max_words=30)):
        assert chunk.word_count == len(chunk.text.split())


def test_chunks_are_immutable() -> None:
    chunk = chunk_document("doc1", "One sentence.")[0]
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]
    assert isinstance(chunk, Chunk)


# --------------------------------------------------------------------------
# config bounds, mirrored from Elasticsearch
# --------------------------------------------------------------------------


def test_rejects_max_words_below_elastic_floor() -> None:
    with pytest.raises(ValueError, match="max_words must be >= 20"):
        ChunkingConfig(max_words=19)


@pytest.mark.parametrize("overlap", [-1, 2, 5])
def test_rejects_overlap_outside_elastic_range(overlap: int) -> None:
    with pytest.raises(ValueError, match="sentence_overlap"):
        ChunkingConfig(sentence_overlap=overlap)
