"""Index mapping and inference endpoint definitions.

One Elasticsearch document per chunk, carrying all three retrieval signals:

    text           text          -> BM25
    text_semantic  semantic_text -> dense vectors
    text_elser     semantic_text -> ELSER learned-sparse

Verified against Elastic's inference and mapping references on 2026-09-05.

Platform note, and it is not a footnote
---------------------------------------
Elasticsearch ships preconfigured endpoints ``.elser-2-elasticsearch`` and
``.multilingual-e5-small-elasticsearch``, but those point at the
**linux-x86_64-optimised** model builds (``.elser_model_2_linux-x86_64``,
``.e5_model_2_linux-x86_64``). Those builds do not deploy on arm64, which is
every Apple Silicon Mac running Elasticsearch under Docker.

So we do not use the preconfigured endpoints. We create our own against the
platform-agnostic model IDs (``.elser_model_2``, ``.multilingual-e5-small``),
which run on both architectures. This costs a little startup time and buys a
stack that behaves the same on a laptop and in CI.
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = [
    "DENSE_ENDPOINT_ID",
    "ELSER_ENDPOINT_ID",
    "RERANK_ENDPOINT_ID",
    "dense_endpoint_body",
    "elser_endpoint_body",
    "index_mapping",
    "rerank_endpoint_body",
]

#: Our own endpoints, not Elastic's preconfigured ones. See the module docstring.
ELSER_ENDPOINT_ID = "ar-elser"
DENSE_ENDPOINT_ID = "ar-dense"
RERANK_ENDPOINT_ID = "ar-rerank"

#: Platform-agnostic model IDs. The ``_linux-x86_64`` variants are faster but
#: will not deploy on arm64.
ELSER_MODEL_ID = ".elser_model_2"
DENSE_MODEL_ID = ".multilingual-e5-small"
RERANK_MODEL_ID = ".rerank-v1"

Allocations = Literal["adaptive", "fixed"]


def _allocation_settings(
    mode: Allocations,
    num_allocations: int,
    max_allocations: int,
) -> dict[str, Any]:
    """Allocation settings for an ML-node-hosted model.

    Adaptive allocations scale with load and are the right default for a
    long-lived cluster. A fixed count is the right choice for a benchmark run,
    where we want the *same* serving capacity for every arm - autoscaling
    mid-run would let an arm that happened to run while scaled up look faster
    than one that ran while scaled down, and latency is a reported metric.
    """
    if mode == "fixed":
        return {"num_allocations": num_allocations, "num_threads": 1}
    return {
        "adaptive_allocations": {
            "enabled": True,
            "min_number_of_allocations": num_allocations,
            "max_number_of_allocations": max_allocations,
        },
        "num_threads": 1,
    }


def elser_endpoint_body(
    *,
    allocations: Allocations = "fixed",
    num_allocations: int = 1,
    max_allocations: int = 4,
) -> dict[str, Any]:
    """Body for ``PUT _inference/sparse_embedding/<ELSER_ENDPOINT_ID>``.

    ELSER indexes at roughly 26 documents/second per allocation, so allocation
    count is the single biggest lever on ingest wall-clock. Measure on a
    1,000-chunk sample before committing to a full corpus run.
    """
    return {
        "service": "elasticsearch",
        "service_settings": {
            "model_id": ELSER_MODEL_ID,
            **_allocation_settings(allocations, num_allocations, max_allocations),
        },
        # Our documents are already one chunk each; chunking them again would
        # create nested inference results and break the 1:1 chunk_id mapping
        # the graph depends on.
        "chunking_settings": {"strategy": "none"},
    }


def dense_endpoint_body(
    *,
    allocations: Allocations = "fixed",
    num_allocations: int = 1,
    max_allocations: int = 4,
) -> dict[str, Any]:
    """Body for ``PUT _inference/text_embedding/<DENSE_ENDPOINT_ID>``."""
    return {
        "service": "elasticsearch",
        "service_settings": {
            "model_id": DENSE_MODEL_ID,
            **_allocation_settings(allocations, num_allocations, max_allocations),
        },
        "chunking_settings": {"strategy": "none"},
    }


def rerank_endpoint_body(
    *,
    allocations: Allocations = "fixed",
    num_allocations: int = 1,
    max_allocations: int = 4,
) -> dict[str, Any]:
    """Body for ``PUT _inference/rerank/<RERANK_ENDPOINT_ID>``.

    Note there is no ``chunking_settings`` here: chunking does not apply to the
    ``rerank`` task type.
    """
    return {
        "service": "elasticsearch",
        "service_settings": {
            "model_id": RERANK_MODEL_ID,
            **_allocation_settings(allocations, num_allocations, max_allocations),
        },
    }


def index_mapping(
    *,
    elser_endpoint: str = ELSER_ENDPOINT_ID,
    dense_endpoint: str = DENSE_ENDPOINT_ID,
) -> dict[str, Any]:
    """Mapping for the chunk index.

    ``inference_id`` is set explicitly on both ``semantic_text`` fields. Elastic
    warns that omitting it lets a version upgrade give new indices a different
    embedding model than existing ones - which in a benchmark would mean two
    runs silently measuring two different systems.
    """
    return {
        "mappings": {
            # A chunk document has a fixed shape. An unexpected field is a bug
            # in the ingest pipeline, and should fail rather than be indexed.
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "position": {"type": "integer"},
                "char_start": {"type": "integer"},
                "char_end": {"type": "integer"},
                "word_count": {"type": "integer"},
                "oversized": {"type": "boolean"},
                # BM25 lexical signal.
                "text": {"type": "text"},
                # Dense semantic signal.
                "text_semantic": {
                    "type": "semantic_text",
                    "inference_id": dense_endpoint,
                    "chunking_settings": {"strategy": "none"},
                },
                # Learned-sparse signal.
                "text_elser": {
                    "type": "semantic_text",
                    "inference_id": elser_endpoint,
                    "chunking_settings": {"strategy": "none"},
                },
            },
        },
        "settings": {
            # Single-node dev clusters never reach green with replicas > 0, and
            # a benchmark index is disposable by definition.
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    }
