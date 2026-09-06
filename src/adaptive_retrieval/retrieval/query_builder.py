"""Build Elasticsearch ``_search`` bodies from arm config.

Pure functions: config in, request JSON out. No I/O, no client, no cluster.
This is deliberate - it means every API shape below is unit-testable without
a running Elasticsearch, and a mistake in a retriever's JSON surfaces as a
failing test rather than as a silently worse benchmark number.

Every shape here was verified against Elastic's retriever reference on
2026-09-05. Two of them are easy to get wrong and are called out inline:

* ``rrf`` and ``linear`` express per-retriever weights **differently**. In
  ``rrf`` the weight is a *sibling of the retriever type key*; in ``linear``
  each entry is an object with a nested ``retriever`` key. Using one shape in
  the other place is silently accepted as an unknown field by some clients and
  produces unweighted results.
* ``rrf``, ``linear`` and ``text_similarity_reranker`` all default
  ``rank_window_size`` to **10**. Config makes it mandatory (see config.py);
  this module never supplies a default of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_retrieval.config import (
    Bm25Retriever,
    DenseRetriever,
    ElserRetriever,
    LinearFusion,
    RerankSpec,
    RetrievalArm,
    RrfFusion,
)

__all__ = ["FieldNames", "build_leaf_query", "build_retriever", "build_search_body"]


@dataclass(frozen=True, slots=True)
class FieldNames:
    """Index field names. Defaults match ``ingest.mapping``."""

    text: str = "text"
    semantic: str = "text_semantic"
    elser: str = "text_elser"


def build_leaf_query(
    retriever: Bm25Retriever | DenseRetriever | ElserRetriever,
    query_text: str,
) -> dict[str, Any]:
    """Build the query DSL for a single-signal retriever.

    - BM25 uses a plain ``match``.
    - Dense uses the ``semantic`` query, which resolves the inference endpoint
      from the ``semantic_text`` field mapping.
    - ELSER uses the ``sparse_vector`` query. ``inference_id`` is deliberately
      omitted: when the queried field is ``semantic_text``, Elasticsearch infers
      the inference ID from the field mapping, which keeps query time and index
      time on the same model by construction.
    """
    if isinstance(retriever, Bm25Retriever):
        return {"match": {retriever.field: {"query": query_text}}}
    if isinstance(retriever, DenseRetriever):
        return {"semantic": {"field": retriever.field, "query": query_text}}
    if isinstance(retriever, ElserRetriever):
        return {"sparse_vector": {"field": retriever.field, "query": query_text}}
    raise TypeError(f"unsupported leaf retriever: {type(retriever).__name__}")


LeafRetriever = Bm25Retriever | DenseRetriever | ElserRetriever


def _leaf_for_source(source: str, fields: FieldNames) -> LeafRetriever:
    if source == "bm25":
        return Bm25Retriever(type="bm25", field=fields.text)
    if source == "dense":
        return DenseRetriever(type="dense", field=fields.semantic)
    if source == "elser":
        return ElserRetriever(type="elser", field=fields.elser)
    raise ValueError(f"unknown fusion source {source!r}")


def _standard(query: dict[str, Any]) -> dict[str, Any]:
    return {"standard": {"query": query}}


def build_retriever(
    retriever: Bm25Retriever | DenseRetriever | ElserRetriever | RrfFusion | LinearFusion,
    query_text: str,
    fields: FieldNames,
) -> dict[str, Any]:
    """Build the ``retriever`` clause for one arm."""
    if isinstance(retriever, Bm25Retriever | DenseRetriever | ElserRetriever):
        return _standard(build_leaf_query(retriever, query_text))

    if isinstance(retriever, RrfFusion):
        children = [
            _standard(build_leaf_query(_leaf_for_source(source, fields), query_text))
            for source in retriever.sources
        ]
        return {
            "rrf": {
                "retrievers": children,
                "rank_constant": retriever.rank_constant,
                "rank_window_size": retriever.rank_window_size,
            }
        }

    if isinstance(retriever, LinearFusion):
        entries: list[dict[str, Any]] = []
        for source in retriever.sources:
            entry: dict[str, Any] = {
                # NOTE: linear nests the child under a "retriever" key. rrf does
                # not. This asymmetry is Elasticsearch's, not ours.
                "retriever": _standard(
                    build_leaf_query(_leaf_for_source(source, fields), query_text)
                ),
            }
            if source in retriever.weights:
                entry["weight"] = retriever.weights[source]
            entries.append(entry)
        return {
            "linear": {
                "retrievers": entries,
                "normalizer": retriever.normalizer,
                "rank_window_size": retriever.rank_window_size,
            }
        }

    raise TypeError(f"unsupported retriever: {type(retriever).__name__}")


def _wrap_reranker(inner: dict[str, Any], rerank: RerankSpec, query_text: str) -> dict[str, Any]:
    """Wrap a retriever in a cross-encoder reranking stage.

    ``inference_text`` is required by Elasticsearch and is the query, not the
    document. Omitting it is a 400, not a silent no-op.
    """
    body: dict[str, Any] = {
        "retriever": inner,
        "field": rerank.field,
        "inference_id": rerank.inference_id,
        "inference_text": query_text,
        "rank_window_size": rerank.depth,
    }
    if rerank.min_score is not None:
        body["min_score"] = rerank.min_score
    return {"text_similarity_reranker": body}


def build_search_body(
    arm: RetrievalArm,
    query_text: str,
    k: int,
    fields: FieldNames | None = None,
    source_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete ``_search`` body for a retrieval arm.

    ``graph_expansion`` is not represented here. Graph expansion is a second,
    separate retrieval by chunk ID whose results are merged in the caller - it
    is not expressible as an Elasticsearch retriever, and pretending otherwise
    in this function would hide where the extra cost is incurred.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    resolved_fields = fields or FieldNames()
    retriever = build_retriever(arm.retriever, query_text, resolved_fields)
    if arm.rerank is not None:
        retriever = _wrap_reranker(retriever, arm.rerank, query_text)

    return {
        "retriever": retriever,
        "size": k,
        # Never fetch the embedding fields back: on a semantic_text field the
        # inference blob dwarfs the text and would inflate every response.
        "_source": source_fields or ["chunk_id", "source_id", "position", "text"],
    }
