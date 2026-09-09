from __future__ import annotations

import time
from itertools import pairwise

import pytest

from adaptive_retrieval.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_document,
    chunk_id_for,
    normalise_text,
    split_sentences,
)

# --------------------------------------------------------------------------
# chunk identity - the property the whole graph design rests on
# --------------------------------------------------------------------------


def test_chunk_id_is_deterministic() -> None:
    a = chunk_id_for("doc1", "Hello world.")
    b = chunk_id_for("doc1", "Hello world.")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_chunk_id_changes_with_source_and_text() -> None:
    base = chunk_id_for("doc1", "Hello world.")
    assert chunk_id_for("doc2", "Hello world.") != base
    assert chunk_id_for("doc1", "Hello world!") != base


def test_chunk_id_field_boundaries_are_unambiguous() -> None:
    assert chunk_id_for("doc1", "2x") != chunk_id_for("doc12", "x")


def test_ids_survive_an_edit_elsewhere_in_the_document() -> None:
    """THE test for the README's central claim: pointers survive a reindex.

    Delete the first sentences; every remaining chunk's text is unchanged, so
    every remaining chunk's ID must be unchanged. Hashing `position` broke
    this while looking correct.
    """
    # 7 words per sentence, 21-word budget => exactly 3 sentences per chunk.
    # Delete one whole chunk's worth, so the remaining chunks group identically
    # and their text is byte-for-byte unchanged.
    cfg = ChunkingConfig(max_words=21, sentence_overlap=0)
    sentences = [" ".join(f"x{i}_{j}" for j in range(6)) + f" end{i}." for i in range(9)]
    original = " ".join(sentences)
    trimmed = " ".join(sentences[3:])

    before = {c.text: c.chunk_id for c in chunk_document("doc1", original, cfg)}
    after = {c.text: c.chunk_id for c in chunk_document("doc1", trimmed, cfg)}

    shared = set(before) & set(after)
    assert shared, "the edit should leave some chunks byte-identical"
    for text in shared:
        assert before[text] == after[text]


def test_chunk_id_rejects_separator_injection() -> None:
    with pytest.raises(ValueError, match="unit separator"):
        chunk_id_for("doc\x1f1", "text")
    with pytest.raises(ValueError, match="unit separator"):
        chunk_id_for("doc1", "te\x1fxt")


def test_identical_text_within_a_document_collapses_to_one_chunk() -> None:
    """Two identical passages are one passage: indexed once, pointed at twice."""
    text = "Legal notice here. Something else entirely. Legal notice here."
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=20))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# normalisation - line endings must not reissue every ID in the corpus
# --------------------------------------------------------------------------


def test_crlf_and_lf_produce_identical_chunks() -> None:
    lf = "First sentence here.\nSecond sentence here."
    crlf = "First sentence here.\r\nSecond sentence here."
    assert [c.chunk_id for c in chunk_document("d", lf)] == [
        c.chunk_id for c in chunk_document("d", crlf)
    ]


def test_unicode_is_nfc_normalised() -> None:
    composed = "Kraków is a city."
    decomposed = "Kraków is a city."
    assert normalise_text(composed) == normalise_text(decomposed)
    assert [c.chunk_id for c in chunk_document("d", composed)] == [
        c.chunk_id for c in chunk_document("d", decomposed)
    ]


# --------------------------------------------------------------------------
# sentence splitting
# --------------------------------------------------------------------------


def _texts(text: str) -> list[str]:
    normalised = normalise_text(text)
    return [normalised[s:e] for s, e in split_sentences(normalised)]


def test_splits_on_terminal_punctuation() -> None:
    assert _texts("One. Two! Three?") == ["One.", "Two!", "Three?"]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Mehta signed it.",
        "Acme Inc. filed today.",
        "Reported on Jan. 5 by staff.",
        "The meeting starts at 9 a.m. Attendees should arrive early.",
        "She has a Ph.D. in physics.",
        "He flew to Washington, D.C. The president was waiting.",
        "Dr. Lee, M.D. treated him.",
        "Gov. Newsom signed it.",
        "R. Mehta signed it.",
    ],
)
def test_abbreviations_and_initials_do_not_split(text: str) -> None:
    assert _texts(text) == [text]


@pytest.mark.parametrize(
    "text",
    ['He called it "the U.S." policy.', "He said 'Dr.' Smith was late."],
)
def test_quoted_abbreviations_do_not_split(text: str) -> None:
    """The abbreviation check must run before the closing-quote rule; quoted
    abbreviations are everywhere in news copy."""
    assert _texts(text) == [text]


def test_closing_quote_still_ends_a_quoted_sentence() -> None:
    assert _texts('He said "no." She left.') == ['He said "no."', "She left."]


@pytest.mark.parametrize("text", ["He paused ... then left.", "Wait... what?"])
def test_ellipsis_is_not_a_terminator(text: str) -> None:
    assert _texts(text) == [text]


def test_ascii_and_unicode_ellipsis_agree() -> None:
    assert len(_texts("Wait... what?")) == len(_texts("Wait… what?"))


def test_bracketed_abbreviation_does_not_split() -> None:
    assert _texts("See the report (approx. 40 pages) for detail.") == [
        "See the report (approx. 40 pages) for detail."
    ]


def test_accented_words_behave_like_ascii_ones() -> None:
    """An ASCII-only lookback merged "Krakow." but split "Lodz." - same
    construction, opposite behaviour, decided by the final character."""
    assert len(_texts("He visited Krakow. Then he left.")) == len(
        _texts("He visited Kraków. Then he left.")
    )
    assert len(_texts("He visited Łódź. Then he left.")) == 2


def test_cjk_terminators_are_recognised() -> None:
    assert len(_texts("这是第一句。这是第二句。")) == 2


def test_keeps_trailing_text_without_punctuation() -> None:
    assert _texts("First one. Dangling tail") == ["First one.", "Dangling tail"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_input_yields_no_sentences(text: str) -> None:
    assert split_sentences(text) == []


def test_splitting_is_lossless() -> None:
    """Every non-whitespace character must survive into exactly one span."""
    text = (
        'Dr. Mehta said "no." He flew to Washington, D.C. at 9 a.m. '
        "Then he paused ... and left. The end?"
    )
    recovered = "".join(text[s:e] for s, e in split_sentences(text))
    assert "".join(recovered.split()) == "".join(text.split())


def test_spans_are_ordered_and_disjoint() -> None:
    spans = split_sentences("One here. Two here. Three here.")
    for (_, first_end), (second_start, _) in pairwise(spans):
        assert first_end <= second_start


# --------------------------------------------------------------------------
# performance - the splitter was quadratic
# --------------------------------------------------------------------------


def test_splitter_is_not_quadratic() -> None:
    """4,000 sentences took 18 seconds with an unbounded lookback."""
    text = " ".join(f"This is sentence number {i} and it is quite ordinary." for i in range(4000))
    started = time.perf_counter()
    spans = split_sentences(text)
    elapsed = time.perf_counter() - started
    assert len(spans) == 4000
    assert elapsed < 1.0, f"split_sentences took {elapsed:.2f}s; the lookback bound is gone"


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
    assert chunks[0].text == "One sentence only."
    assert not chunks[0].oversized


def test_normal_chunks_respect_the_word_budget() -> None:
    cfg = ChunkingConfig(max_words=25)
    chunks = chunk_document("doc1", _doc(10, words_each=10), cfg)
    normal = [c for c in chunks if not c.oversized]
    assert normal, "expected at least one normal chunk"
    for chunk in normal:
        assert chunk.word_count <= cfg.max_words


def test_positions_are_sequential_from_zero() -> None:
    chunks = chunk_document("doc1", _doc(12), ChunkingConfig(max_words=25))
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_default_has_no_overlap() -> None:
    """Overlap duplicates entities into the graph and inflates node degree -
    the signal A6's claim rests on."""
    assert ChunkingConfig().sentence_overlap == 0
    chunks = chunk_document("doc1", _doc(8), ChunkingConfig(max_words=25))
    for previous, current in pairwise(chunks):
        assert current.char_start >= previous.char_end


def test_overlap_repeats_a_whole_sentence_when_enabled() -> None:
    text = _doc(6, words_each=10)
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=25, sentence_overlap=1))
    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert current.char_start < previous.char_end, "overlap must repeat text"


def test_no_chunk_is_a_strict_subset_of_another() -> None:
    """The overlap loop could emit a window that consumed no new sentence."""
    text = " ".join(
        [
            " ".join(f"a{i}" for i in range(10)) + ".",
            " ".join(f"b{i}" for i in range(12)) + ".",
            " ".join(f"c{i}" for i in range(20)) + ".",
        ]
    )
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=25, sentence_overlap=1))
    for outer in chunks:
        for inner in chunks:
            if outer is inner:
                continue
            contained = inner.char_start >= outer.char_start and inner.char_end <= outer.char_end
            assert not contained, f"{inner.text!r} is contained in {outer.text!r}"


def test_every_sentence_appears_in_some_chunk() -> None:
    text = normalise_text(_doc(15))
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=30))
    covered = "".join("".join(c.text.split()) for c in chunks)
    for start, end in split_sentences(text):
        assert "".join(text[start:end].split()) in covered


def test_oversized_sentence_is_flagged() -> None:
    long_sentence = " ".join(f"w{i}" for i in range(120)) + "."
    text = f"Short one. {long_sentence} Short two."
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=30, max_chars=10_000))
    assert any(c.oversized for c in chunks)


def test_terminates_when_every_sentence_is_oversized() -> None:
    chunks = chunk_document(
        "doc1",
        " ".join(" ".join(f"w{i}_{j}" for j in range(60)) + "." for i in range(5)),
        ChunkingConfig(max_words=20, max_chars=10_000),
    )
    assert len(chunks) == 5


def test_char_ceiling_is_enforced_even_without_whitespace() -> None:
    """A script with no spaces has a word count of 1 no matter how long it is."""
    text = "这" * 5000 + "。"
    chunks = chunk_document("doc1", text, ChunkingConfig(max_chars=500))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 500
        assert chunk.oversized


def test_char_ceiling_prefers_whitespace_boundaries() -> None:
    text = " ".join(f"word{i}" for i in range(500)) + "."
    chunks = chunk_document("doc1", text, ChunkingConfig(max_words=10_000, max_chars=300))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 300
        assert not chunk.text.startswith(" ")


def test_chunking_is_deterministic_across_calls() -> None:
    text = _doc(20)
    assert [c.chunk_id for c in chunk_document("doc1", text)] == [
        c.chunk_id for c in chunk_document("doc1", text)
    ]


def test_chunk_text_matches_its_own_span() -> None:
    text = normalise_text(_doc(15))
    for chunk in chunk_document("doc1", text, ChunkingConfig(max_words=30)):
        assert chunk.text == text[chunk.char_start : chunk.char_end]


def test_word_count_matches_text() -> None:
    for chunk in chunk_document("doc1", _doc(15), ChunkingConfig(max_words=30)):
        assert chunk.word_count == len(chunk.text.split())


def test_chunks_are_immutable() -> None:
    chunk = chunk_document("doc1", "One sentence.")[0]
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]
    assert isinstance(chunk, Chunk)


# --------------------------------------------------------------------------
# config bounds
# --------------------------------------------------------------------------


def test_rejects_max_words_below_elastic_floor() -> None:
    with pytest.raises(ValueError, match="max_words must be >= 20"):
        ChunkingConfig(max_words=19)


@pytest.mark.parametrize("overlap", [-1, 2, 5])
def test_rejects_overlap_outside_elastic_range(overlap: int) -> None:
    with pytest.raises(ValueError, match="sentence_overlap"):
        ChunkingConfig(sentence_overlap=overlap)


def test_rejects_tiny_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        ChunkingConfig(max_chars=10)
