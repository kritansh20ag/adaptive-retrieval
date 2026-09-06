from __future__ import annotations

import pytest

from adaptive_retrieval.config import (
    Bm25Retriever,
    DenseRetriever,
    ElserRetriever,
    LinearFusion,
    RerankSpec,
    RetrievalArm,
    RrfFusion,
)
from adaptive_retrieval.retrieval.query_builder import (
    FieldNames,
    build_leaf_query,
    build_retriever,
    build_search_body,
)

FIELDS = FieldNames()


# --------------------------------------------------------------------------
# leaf queries
# --------------------------------------------------------------------------


def test_bm25_uses_match() -> None:
    query = build_leaf_query(Bm25Retriever(type="bm25", field="text"), "acme layoffs")
    assert query == {"match": {"text": {"query": "acme layoffs"}}}


def test_dense_uses_semantic_query() -> None:
    query = build_leaf_query(DenseRetriever(type="dense", field="text_semantic"), "acme layoffs")
    assert query == {"semantic": {"field": "text_semantic", "query": "acme layoffs"}}


def test_elser_uses_sparse_vector_without_inference_id() -> None:
    """On a semantic_text field Elasticsearch infers the inference ID from the
    mapping. Supplying one here would let query time and index time drift onto
    different models."""
    query = build_leaf_query(ElserRetriever(type="elser", field="text_elser"), "acme layoffs")
    assert query == {"sparse_vector": {"field": "text_elser", "query": "acme layoffs"}}
    assert "inference_id" not in query["sparse_vector"]


# --------------------------------------------------------------------------
# fusion - the two weight shapes differ, and that is the point
# --------------------------------------------------------------------------


def test_rrf_shape_and_parameters() -> None:
    fusion = RrfFusion(
        type="rrf", sources=["bm25", "dense", "elser"], rank_constant=60, rank_window_size=100
    )
    built = build_retriever(fusion, "q", FIELDS)

    assert set(built) == {"rrf"}
    body = built["rrf"]
    assert body["rank_constant"] == 60
    assert body["rank_window_size"] == 100
    assert len(body["retrievers"]) == 3
    # rrf children are bare retrievers - no nested "retriever" key.
    for child in body["retrievers"]:
        assert set(child) == {"standard"}


def test_rrf_children_follow_source_order() -> None:
    fusion = RrfFusion(type="rrf", sources=["elser", "bm25"], rank_window_size=50)
    body = build_retriever(fusion, "q", FIELDS)["rrf"]
    first, second = body["retrievers"]
    assert "sparse_vector" in first["standard"]["query"]
    assert "match" in second["standard"]["query"]


def test_linear_nests_children_under_a_retriever_key() -> None:
    """linear's child shape differs from rrf's. Getting this wrong yields
    unweighted results rather than an error."""
    fusion = LinearFusion(
        type="linear",
        sources=["bm25", "dense"],
        normalizer="minmax",
        weights={"bm25": 0.3, "dense": 0.7},
        rank_window_size=100,
    )
    body = build_retriever(fusion, "q", FIELDS)["linear"]

    assert body["normalizer"] == "minmax"
    assert body["rank_window_size"] == 100
    for entry in body["retrievers"]:
        assert "retriever" in entry
        assert set(entry["retriever"]) == {"standard"}
    assert [entry["weight"] for entry in body["retrievers"]] == [0.3, 0.7]


def test_linear_omits_weight_when_not_configured() -> None:
    """Elasticsearch defaults weight to 1.0; emitting nothing is not the same
    as emitting 0."""
    fusion = LinearFusion(type="linear", sources=["bm25", "dense"], rank_window_size=100)
    body = build_retriever(fusion, "q", FIELDS)["linear"]
    for entry in body["retrievers"]:
        assert "weight" not in entry


def test_rrf_and_linear_child_shapes_are_not_interchangeable() -> None:
    rrf = build_retriever(
        RrfFusion(type="rrf", sources=["bm25", "dense"], rank_window_size=100), "q", FIELDS
    )["rrf"]["retrievers"][0]
    linear = build_retriever(
        LinearFusion(type="linear", sources=["bm25", "dense"], rank_window_size=100), "q", FIELDS
    )["linear"]["retrievers"][0]
    assert set(rrf) != set(linear)


def test_custom_field_names_are_threaded_through() -> None:
    fields = FieldNames(text="body", semantic="body_semantic", elser="body_elser")
    fusion = RrfFusion(type="rrf", sources=["bm25", "dense", "elser"], rank_window_size=100)
    children = build_retriever(fusion, "q", fields)["rrf"]["retrievers"]
    assert children[0]["standard"]["query"]["match"] == {"body": {"query": "q"}}
    assert children[1]["standard"]["query"]["semantic"]["field"] == "body_semantic"
    assert children[2]["standard"]["query"]["sparse_vector"]["field"] == "body_elser"


# --------------------------------------------------------------------------
# reranking
# --------------------------------------------------------------------------


def _rerank_arm(**overrides: object) -> RetrievalArm:
    spec = {
        "inference_id": ".rerank-v1-elasticsearch",
        "field": "text",
        "depth": 100,
        **overrides,
    }
    return RetrievalArm(
        id="A4",
        kind="retrieval",
        retriever=RrfFusion(type="rrf", sources=["bm25", "dense"], rank_window_size=100),
        rerank=RerankSpec(**spec),  # type: ignore[arg-type]
    )


def test_reranker_wraps_the_inner_retriever() -> None:
    body = build_search_body(_rerank_arm(), "acme layoffs", k=10)
    reranker = body["retriever"]["text_similarity_reranker"]
    assert "rrf" in reranker["retriever"]


def test_reranker_inference_text_is_the_query() -> None:
    """inference_text is required by Elasticsearch and is the query, not the doc."""
    body = build_search_body(_rerank_arm(), "acme layoffs", k=10)
    reranker = body["retriever"]["text_similarity_reranker"]
    assert reranker["inference_text"] == "acme layoffs"
    assert reranker["inference_id"] == ".rerank-v1-elasticsearch"
    assert reranker["rank_window_size"] == 100


def test_reranker_omits_min_score_when_unset() -> None:
    body = build_search_body(_rerank_arm(), "q", k=10)
    assert "min_score" not in body["retriever"]["text_similarity_reranker"]


def test_reranker_includes_min_score_when_set() -> None:
    body = build_search_body(_rerank_arm(min_score=0.3), "q", k=10)
    assert body["retriever"]["text_similarity_reranker"]["min_score"] == 0.3


def test_arm_without_rerank_is_not_wrapped() -> None:
    arm = RetrievalArm(id="A3", kind="retrieval", retriever=Bm25Retriever(type="bm25"))
    body = build_search_body(arm, "q", k=10)
    assert "text_similarity_reranker" not in body["retriever"]


# --------------------------------------------------------------------------
# search body
# --------------------------------------------------------------------------


def test_search_body_sets_size_and_excludes_inference_fields() -> None:
    arm = RetrievalArm(id="A1", kind="retrieval", retriever=Bm25Retriever(type="bm25"))
    body = build_search_body(arm, "q", k=25)
    assert body["size"] == 25
    # Returning semantic_text inference blobs would dwarf the text itself.
    assert body["_source"] == ["chunk_id", "source_id", "position", "text"]
    assert "text_semantic" not in body["_source"]
    assert "text_elser" not in body["_source"]


def test_search_body_rejects_non_positive_k() -> None:
    arm = RetrievalArm(id="A1", kind="retrieval", retriever=Bm25Retriever(type="bm25"))
    with pytest.raises(ValueError, match="k must be >= 1"):
        build_search_body(arm, "q", k=0)


def test_no_default_rank_window_size_is_ever_emitted() -> None:
    """Elasticsearch's default is 10. It must always come from config, never
    from this module."""
    fusion = RrfFusion(type="rrf", sources=["bm25", "dense"], rank_window_size=37)
    assert build_retriever(fusion, "q", FIELDS)["rrf"]["rank_window_size"] == 37


def test_query_text_is_not_interpolated_into_field_names() -> None:
    """Query text is data. It must appear only in query positions."""
    fusion = RrfFusion(type="rrf", sources=["bm25", "dense", "elser"], rank_window_size=100)
    children = build_retriever(fusion, 'evil" }, "x": {"', FIELDS)["rrf"]["retrievers"]
    assert children[0]["standard"]["query"]["match"]["text"]["query"] == 'evil" }, "x": {"'
    assert list(children[1]["standard"]["query"]["semantic"]) == ["field", "query"]
