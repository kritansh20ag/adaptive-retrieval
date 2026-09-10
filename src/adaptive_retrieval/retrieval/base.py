"""What a retrieval strategy returns, independent of how it retrieved."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RetrievalResult", "RetrievedChunk"]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_id: str = ""
    position: int = 0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Ranked chunks plus the cost of producing them.

    Latency is split per stage rather than reported as one number: a slow
    reranker must be distinguishable from a slow retriever, and "reranking is
    80% of our latency" is a finding, not noise.
    """

    chunks: tuple[RetrievedChunk, ...] = ()
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    graph_ms: float = 0.0
    #: Elasticsearch's own `took`, so network noise is not charged to retrieval.
    took_ms: float = 0.0
    #: Chunk IDs contributed by graph expansion, for auditing A5/A6.
    graph_chunk_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.chunks]

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms + self.graph_ms
