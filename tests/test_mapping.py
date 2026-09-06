from __future__ import annotations

from adaptive_retrieval.ingest.mapping import (
    DENSE_ENDPOINT_ID,
    ELSER_ENDPOINT_ID,
    dense_endpoint_body,
    elser_endpoint_body,
    index_mapping,
    rerank_endpoint_body,
)


def test_models_are_platform_agnostic() -> None:
    """The linux-x86_64 model builds do not deploy on arm64, which is every
    Apple Silicon dev machine. We must not reference them."""
    for body in (elser_endpoint_body(), dense_endpoint_body(), rerank_endpoint_body()):
        model_id = body["service_settings"]["model_id"]
        assert "linux-x86_64" not in model_id, model_id


def test_fixed_allocations_by_default() -> None:
    """Autoscaling mid-run would let one arm serve at a different capacity than
    another, and latency is a reported metric."""
    settings = elser_endpoint_body()["service_settings"]
    assert settings["num_allocations"] == 1
    assert "adaptive_allocations" not in settings


def test_adaptive_allocations_when_requested() -> None:
    settings = elser_endpoint_body(allocations="adaptive", num_allocations=2, max_allocations=8)[
        "service_settings"
    ]
    assert settings["adaptive_allocations"] == {
        "enabled": True,
        "min_number_of_allocations": 2,
        "max_number_of_allocations": 8,
    }
    assert "num_allocations" not in settings


def test_embedding_endpoints_disable_chunking() -> None:
    """Our documents are already one chunk each. Chunking again would create
    nested inference results and break the 1:1 chunk_id mapping the graph needs."""
    assert elser_endpoint_body()["chunking_settings"] == {"strategy": "none"}
    assert dense_endpoint_body()["chunking_settings"] == {"strategy": "none"}


def test_rerank_endpoint_has_no_chunking_settings() -> None:
    """chunking_settings does not apply to the rerank task type."""
    assert "chunking_settings" not in rerank_endpoint_body()


def test_mapping_carries_all_three_signals() -> None:
    props = index_mapping()["mappings"]["properties"]
    assert props["text"]["type"] == "text"
    assert props["text_semantic"]["type"] == "semantic_text"
    assert props["text_elser"]["type"] == "semantic_text"


def test_semantic_fields_pin_their_inference_endpoint() -> None:
    """Omitting inference_id lets an upgrade silently swap the embedding model,
    which would mean two runs measuring two different systems."""
    props = index_mapping()["mappings"]["properties"]
    assert props["text_semantic"]["inference_id"] == DENSE_ENDPOINT_ID
    assert props["text_elser"]["inference_id"] == ELSER_ENDPOINT_ID


def test_mapping_is_strict() -> None:
    """An unexpected field is an ingest bug and must fail, not be indexed."""
    assert index_mapping()["mappings"]["dynamic"] == "strict"


def test_mapping_carries_chunk_identity_fields() -> None:
    props = index_mapping()["mappings"]["properties"]
    assert props["chunk_id"]["type"] == "keyword"
    assert props["source_id"]["type"] == "keyword"
    assert props["position"]["type"] == "integer"


def test_single_node_friendly_settings() -> None:
    settings = index_mapping()["settings"]
    assert settings["number_of_replicas"] == 0


def test_custom_endpoints_are_threaded_through() -> None:
    props = index_mapping(elser_endpoint="x-elser", dense_endpoint="x-dense")["mappings"][
        "properties"
    ]
    assert props["text_elser"]["inference_id"] == "x-elser"
    assert props["text_semantic"]["inference_id"] == "x-dense"
