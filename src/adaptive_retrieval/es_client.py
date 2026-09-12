"""Elasticsearch access: provisioning, ingest and search.

Everything that talks to a cluster lives here, so the rest of the package
stays unit-testable without one. The interesting logic is deliberately *not*
here - query bodies are built in ``retrieval.query_builder``, mappings in
``ingest.mapping``.

The one piece of judgement in this module is :meth:`assert_ready`. A retrieval
benchmark's worst failure is not a crash, it is an arm that quietly returns
nothing: RRF over three sources where ELSER is undeployed still produces a
plausible ranked list and a plausible nDCG. So the models are asserted
*started* before any arm runs, and a missing deployment is a loud error.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from adaptive_retrieval.chunking import Chunk
from adaptive_retrieval.ingest.mapping import (
    DENSE_ENDPOINT_ID,
    ELSER_ENDPOINT_ID,
    RERANK_ENDPOINT_ID,
    dense_endpoint_body,
    elser_endpoint_body,
    index_mapping,
    rerank_endpoint_body,
)
from adaptive_retrieval.retrieval.base import RetrievalResult, RetrievedChunk

__all__ = ["EsClient", "StackNotReadyError"]


class StackNotReadyError(RuntimeError):
    """Raised when the cluster cannot serve every arm the config defines."""


class EsClient:
    """Thin wrapper over the Elasticsearch client."""

    def __init__(self, client: Elasticsearch, index: str) -> None:
        self.client = client
        self.index = index

    @classmethod
    def connect(
        cls,
        index: str,
        *,
        hosts: str = "http://localhost:9200",
        api_key: str | None = None,
        request_timeout: float = 120.0,
    ) -> EsClient:
        client = Elasticsearch(
            hosts,
            api_key=api_key,
            request_timeout=request_timeout,
            # Transient errors are retried by the client; the harness records
            # the attempt count so retries stay out of the latency column.
            retry_on_timeout=True,
            max_retries=3,
        )
        return cls(client, index)

    # -- provisioning ------------------------------------------------------

    def ensure_inference_endpoints(self, *, num_allocations: int = 1) -> None:
        """Create the three inference endpoints if they are absent.

        Idempotent: an existing endpoint is left alone rather than recreated,
        because recreating one would re-download and re-deploy the model.
        """
        wanted = [
            ("sparse_embedding", ELSER_ENDPOINT_ID, elser_endpoint_body),
            ("text_embedding", DENSE_ENDPOINT_ID, dense_endpoint_body),
            ("rerank", RERANK_ENDPOINT_ID, rerank_endpoint_body),
        ]
        for task_type, endpoint_id, body_fn in wanted:
            if self._inference_exists(endpoint_id):
                continue
            self.client.inference.put(
                task_type=task_type,
                inference_id=endpoint_id,
                body=body_fn(num_allocations=num_allocations),
            )

    def _inference_exists(self, endpoint_id: str) -> bool:
        try:
            self.client.inference.get(inference_id=endpoint_id)
        except Exception:  # any failure means "not usable yet"
            return False
        return True

    def ensure_index(self, *, recreate: bool = False) -> None:
        if recreate and self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, body=index_mapping())

    def assert_ready(self, *, require_rerank: bool = True) -> None:
        """Fail loudly unless every model an arm needs is deployed and started.

        Without this the benchmark's worst failure mode is silent: an arm whose
        ELSER leg returns nothing still yields a ranked list, a plausible nDCG,
        and a wrong conclusion.
        """
        missing: list[str] = []
        required = [ELSER_ENDPOINT_ID, DENSE_ENDPOINT_ID]
        if require_rerank:
            required.append(RERANK_ENDPOINT_ID)

        for endpoint_id in required:
            if not self._inference_exists(endpoint_id):
                missing.append(f"{endpoint_id} (inference endpoint absent)")

        if missing:
            raise StackNotReadyError(
                "the cluster cannot serve every arm:\n  - "
                + "\n  - ".join(missing)
                + "\nRun `ensure_inference_endpoints()` and wait for the models to deploy. "
                "If deployment fails for lack of memory, see docker-compose.yml: ML "
                "inference runs outside the JVM heap and needs ~8GB to host ELSER and "
                "the reranker together."
            )

    def licence_is_active(self) -> tuple[bool, str]:
        """Return whether the licence permits ML, and its status string.

        A self-generated trial expires after 30 days, after which ELSER and the
        reranker stop and the cluster drops to basic - silently producing
        different numbers mid-project.
        """
        info = self.client.license.get()
        licence = dict(info).get("license", {})
        status = str(licence.get("status", "unknown"))
        kind = str(licence.get("type", "unknown"))
        return status == "active" and kind in {"trial", "platinum", "enterprise"}, (
            f"{kind}/{status}"
        )

    # -- ingest ------------------------------------------------------------

    def index_chunks(
        self, chunks: Iterable[Chunk], *, chunk_size: int = 100
    ) -> tuple[int, list[dict[str, Any]]]:
        """Bulk-index chunks. Returns ``(succeeded, failures)``.

        The Elasticsearch document ``_id`` is the content-addressed chunk ID, so
        re-ingesting unchanged content overwrites in place rather than
        duplicating - and the graph's pointers keep resolving.

        ``raise_on_error=False`` is deliberate. The helper's default aborts the
        whole ingest on the first bad document, after earlier batches have
        already committed, leaving a partially-populated index and no way to
        report which documents failed. A single ELSER inference timeout - the
        common symptom of under-provisioned ML memory - would end the run. A
        partial ingest must be reportable and resumable, not fatal.
        """
        actions = (
            {
                "_index": self.index,
                "_id": chunk.chunk_id,
                "_source": {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "position": chunk.position,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "word_count": chunk.word_count,
                    "oversized": chunk.oversized,
                    "text": chunk.text,
                    "text_semantic": chunk.text,
                    "text_elser": chunk.text,
                },
            }
            for chunk in chunks
        )
        succeeded, failures = bulk(
            self.client,
            actions,
            chunk_size=chunk_size,
            request_timeout=600,
            raise_on_error=False,
            raise_on_exception=False,
        )
        # With raise_on_error=False the helper returns the error list; the
        # int overload only applies when stats_only=True.
        errors: list[dict[str, Any]] = (
            [f for f in failures if isinstance(f, dict)] if isinstance(failures, list) else []
        )
        return int(succeeded), errors

    def indexed_chunk_ids(self) -> set[str]:
        """Every chunk ID already in the index, for resumable ingest.

        Reads ``_id`` with ``_source: false`` - the document id IS the chunk id,
        so fetching the source would move the whole corpus over the wire for
        nothing. The scroll context is always released, including on an
        exception: Elasticsearch caps open scrolls at 500 by default.
        """
        if not self.client.indices.exists(index=self.index):
            return set()
        self.client.indices.refresh(index=self.index)
        found: set[str] = set()
        scroll_id: str | None = None
        try:
            response = self.client.search(
                index=self.index,
                body={"query": {"match_all": {}}, "_source": False, "size": 1000},
                scroll="2m",
            )
            while True:
                scroll_id = response.get("_scroll_id")
                hits = response["hits"]["hits"]
                if not hits:
                    break
                found.update(hit["_id"] for hit in hits)
                response = self.client.scroll(scroll_id=scroll_id, scroll="2m")
        finally:
            if scroll_id:
                self.client.clear_scroll(scroll_id=scroll_id)
        return found

    def count(self) -> int:
        self.client.indices.refresh(index=self.index)
        return int(self.client.count(index=self.index)["count"])

    # -- search ------------------------------------------------------------

    def search(self, body: dict[str, Any]) -> RetrievalResult:
        """Run a prepared ``_search`` body and time it.

        Wall-clock and Elasticsearch's own ``took`` are both recorded, so
        network noise is not charged to retrieval.
        """
        started = time.perf_counter()
        response = self.client.search(index=self.index, body=body)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        chunks = tuple(
            RetrievedChunk(
                chunk_id=hit["_source"]["chunk_id"],
                text=hit["_source"].get("text", ""),
                score=float(hit.get("_score") or 0.0),
                source_id=hit["_source"].get("source_id", ""),
                position=int(hit["_source"].get("position", 0)),
            )
            for hit in response["hits"]["hits"]
        )
        return RetrievalResult(
            chunks=chunks,
            retrieve_ms=elapsed_ms,
            took_ms=float(response.get("took", 0)),
        )

    def fetch_chunks(self, chunk_ids: Sequence[str]) -> tuple[RetrievedChunk, ...]:
        """Fetch chunks by ID - the second half of graph expansion.

        The graph returns chunk IDs; the text still lives only in
        Elasticsearch, which is the whole point of storing pointers.
        """
        if not chunk_ids:
            return ()
        response = self.client.mget(
            index=self.index,
            body={"ids": list(chunk_ids)},
            source=["chunk_id", "source_id", "position", "text"],
        )
        return tuple(
            RetrievedChunk(
                chunk_id=doc["_source"]["chunk_id"],
                text=doc["_source"].get("text", ""),
                score=0.0,
                source_id=doc["_source"].get("source_id", ""),
                position=int(doc["_source"].get("position", 0)),
            )
            for doc in response["docs"]
            if doc.get("found")
        )
