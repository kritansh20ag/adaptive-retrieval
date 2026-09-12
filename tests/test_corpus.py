from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_retrieval.chunking import ChunkingConfig
from adaptive_retrieval.ingest.corpus import CorpusError, chunk_corpus, load_corpus


def _write_jsonl(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_loads_multihop_rag_shape(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path,
        {"id": "d1", "title": "Acme layoffs", "body": "Acme cut staff."},
        {"id": "d2", "title": "Merger", "body": "Northwind merged."},
    )
    docs = load_corpus(path)
    assert [d.id for d in docs] == ["d1", "d2"]
    assert docs[0].title == "Acme layoffs"


@pytest.mark.parametrize("field", ["body", "text", "content", "passage"])
def test_accepts_alternative_body_fields(tmp_path: Path, field: str) -> None:
    docs = load_corpus(_write_jsonl(tmp_path, {"id": "d1", field: "Some text."}))
    assert docs[0].text == "Some text."


@pytest.mark.parametrize("field", ["id", "_id", "doc_id", "url"])
def test_accepts_alternative_id_fields(tmp_path: Path, field: str) -> None:
    docs = load_corpus(_write_jsonl(tmp_path, {field: "d1", "body": "Some text."}))
    assert docs[0].id == "d1"


def test_loads_json_array(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"id": "d1", "body": "Text."}]), encoding="utf-8")
    assert len(load_corpus(path)) == 1


def test_missing_body_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """Silently dropping documents changes the corpus without changing the
    config that claims to describe it."""
    with pytest.raises(CorpusError, match="has no body"):
        load_corpus(_write_jsonl(tmp_path, {"id": "d1", "title": "Only a title"}))


def test_missing_id_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="has no id"):
        load_corpus(_write_jsonl(tmp_path, {"body": "Text with no id."}))


def test_blank_body_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="has no body"):
        load_corpus(_write_jsonl(tmp_path, {"id": "d1", "body": "   "}))


def test_duplicate_document_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="duplicate document ids"):
        load_corpus(
            _write_jsonl(tmp_path, {"id": "d1", "body": "One."}, {"id": "d1", "body": "Two."})
        )


def test_empty_corpus_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="contains no documents"):
        load_corpus(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="cannot read corpus"):
        load_corpus(tmp_path / "nope.jsonl")


def test_malformed_line_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "d1", "body": "ok"}\n{broken\n', encoding="utf-8")
    with pytest.raises(CorpusError, match=r"corpus\.jsonl:2"):
        load_corpus(path)


# --------------------------------------------------------------------------
# chunking the corpus
# --------------------------------------------------------------------------


def test_chunks_every_document(tmp_path: Path) -> None:
    docs = load_corpus(
        _write_jsonl(
            tmp_path,
            {"id": "d1", "body": "First doc sentence one. First doc sentence two."},
            {"id": "d2", "body": "Second doc sentence one."},
        )
    )
    chunks = list(chunk_corpus(docs, ChunkingConfig(max_words=20)))
    assert {c.source_id for c in chunks} == {"d1", "d2"}


def test_identical_text_in_different_documents_gets_distinct_ids(tmp_path: Path) -> None:
    """source_id is in the hash, so shared boilerplate does not collide."""
    docs = load_corpus(
        _write_jsonl(
            tmp_path,
            {"id": "d1", "body": "Shared legal notice."},
            {"id": "d2", "body": "Shared legal notice."},
        )
    )
    chunks = list(chunk_corpus(docs))
    assert len({c.chunk_id for c in chunks}) == 2


def test_chunk_ids_are_unique_across_the_corpus(tmp_path: Path) -> None:
    docs = load_corpus(
        _write_jsonl(
            tmp_path,
            *[
                {"id": f"d{i}", "body": f"Document {i} sentence one. And sentence two."}
                for i in range(20)
            ],
        )
    )
    ids = [c.chunk_id for c in chunk_corpus(docs, ChunkingConfig(max_words=20))]
    assert len(ids) == len(set(ids))
