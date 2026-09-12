from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from adaptive_retrieval.config import BenchmarkConfig, load_config
from adaptive_retrieval.generate import AnswerPayload, CitedSentence, GenerationResult
from adaptive_retrieval.golden import GoldenCase
from adaptive_retrieval.graph.store import InMemoryGraph, Triple
from adaptive_retrieval.harness.executor import ArmExecutor, coverage_sufficiency
from adaptive_retrieval.retrieval.base import RetrievalResult, RetrievedChunk

CONFIG = """
corpus: test-corpus
golden_set: golden/v1.jsonl
judges:
  entailment: deberta-v3-large-mnli
defaults:
  index: chunks-v1
  k: 10
arms:
  - id: A0
    kind: closed_book
  - id: A1
    kind: retrieval
    retriever:
      type: bm25
  - id: A4
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
      rank_window_size: 100
  - id: A5
    kind: retrieval
    extends: A4
    graph_expansion: true
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A4, A5]
    retry:
      max: 1
      widen_k: 50
"""

CASE = GoldenCase.model_validate(
    {
        "id": "Q17",
        "class": "multi_hop",
        "question": "How are Acme, Northwind and Globex connected?",
        "gold_chunks": ["g1"],
        "answer": "Through a chain of vendors.",
        "should_abstain": False,
    }
)


class FakeEs:
    """Returns a fixed ranked list, and records every request it was given."""

    def __init__(self, chunk_ids: list[str], extra: dict[str, str] | None = None) -> None:
        self._chunk_ids = chunk_ids
        self._extra = extra or {}
        self.searches: list[dict[str, Any]] = []
        self.fetched: list[list[str]] = []

    def search(self, body: dict[str, Any]) -> RetrievalResult:
        self.searches.append(body)
        size = int(body.get("size", 10))
        chunks = tuple(RetrievedChunk(cid, f"text {cid}", 1.0) for cid in self._chunk_ids[:size])
        return RetrievalResult(chunks=chunks, retrieve_ms=1.0)

    def fetch_chunks(self, chunk_ids: Any) -> tuple[RetrievedChunk, ...]:
        ids = list(chunk_ids)
        self.fetched.append(ids)
        return tuple(RetrievedChunk(cid, self._extra.get(cid, f"text {cid}"), 0.0) for cid in ids)


class FakeGenerator:
    def __init__(self) -> None:
        self.model = "claude-opus-5"
        self.prompts: list[tuple[str, tuple[RetrievedChunk, ...]]] = []

    def answer(self, question: str, chunks: tuple[RetrievedChunk, ...]) -> GenerationResult:
        self.prompts.append((question, chunks))
        return GenerationResult(
            payload=AnswerPayload(
                abstained=False,
                sentences=[CitedSentence(text="An answer.", cited_chunk_ids=["g1"])],
            ),
            served_model="claude-opus-5",
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=0.001,
            cost_usd=0.001,
        )


def _config(tmp_path: Path) -> BenchmarkConfig:
    path = tmp_path / "benchmark.yaml"
    path.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
    return load_config(path)


def _graph() -> InMemoryGraph:
    graph = InMemoryGraph()
    names = ["Acme", "Northwind", "Globex", "Initech", "Umbrella", "Soylent"]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            graph.add(Triple(a, "rel", b, f"g{i}"))
    graph.finalise()
    return graph


def _executor(tmp_path: Path, es: FakeEs, **kwargs: Any) -> ArmExecutor:
    return ArmExecutor(
        config=_config(tmp_path),
        es=es,  # type: ignore[arg-type]
        generator=FakeGenerator(),  # type: ignore[arg-type]
        **kwargs,
    )


# --------------------------------------------------------------------------
# closed book
# --------------------------------------------------------------------------


def test_closed_book_retrieves_nothing(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es)("A0", CASE)
    assert outcome.retrieval.chunks == ()
    assert es.searches == []


# --------------------------------------------------------------------------
# graph expansion must reach the scored window
# --------------------------------------------------------------------------


def test_graph_chunks_land_inside_the_scored_top_k(tmp_path: Path) -> None:
    """THE regression test. Appending graph chunks after position k means the
    metrics - computed at k - slice them off, and A5 becomes arithmetically
    identical to A4. The benchmark could not then detect its own hypothesis.
    """
    es = FakeEs([f"c{i}" for i in range(10)])
    executor = _executor(tmp_path, es, graph=_graph(), graph_slots=3)
    outcome = executor("A5", CASE)

    top_k = outcome.retrieval.chunk_ids[:10]
    graph_ids = set(outcome.retrieval.graph_chunk_ids)
    assert graph_ids, "graph expansion produced nothing"
    assert graph_ids & set(top_k), "graph evidence never reached the scored window"


def test_a5_and_a4_do_not_return_the_same_ranking(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    executor = _executor(tmp_path, es, graph=_graph(), graph_slots=3)
    assert executor("A4", CASE).retrieval.chunk_ids != executor("A5", CASE).retrieval.chunk_ids


def test_result_never_exceeds_k(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es, graph=_graph(), graph_slots=3)("A5", CASE)
    assert len(outcome.retrieval.chunks) <= 10


def test_graph_slots_bounds_the_graph_contribution(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es, graph=_graph(), graph_slots=2)("A5", CASE)
    assert len(outcome.retrieval.graph_chunk_ids) <= 2


def test_graph_expansion_is_timed_separately(tmp_path: Path) -> None:
    """Folding it into retrieve_ms hides where A5's extra cost is paid."""
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es, graph=_graph())("A5", CASE)
    assert outcome.retrieval.graph_ms > 0.0


def test_already_retrieved_chunks_do_not_consume_the_graph_budget(tmp_path: Path) -> None:
    """De-duplication happens before truncation, not after."""
    es = FakeEs(["g0", "g1", "g2"] + [f"c{i}" for i in range(7)])
    outcome = _executor(tmp_path, es, graph=_graph(), graph_slots=2)("A5", CASE)
    for chunk_id in outcome.retrieval.graph_chunk_ids:
        assert chunk_id not in {"g0", "g1", "g2"}


def test_graph_arm_without_a_graph_is_a_loud_error(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    with pytest.raises(RuntimeError, match="no graph was provided"):
        _executor(tmp_path, es)("A5", CASE)


def test_invalid_graph_slots_rejected(tmp_path: Path) -> None:
    es = FakeEs([])
    with pytest.raises(ValueError, match="graph_slots"):
        _executor(tmp_path, es, graph=_graph(), graph_slots=0)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def test_router_records_its_decision(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es, graph=_graph())("A6", CASE)
    assert outcome.route is not None
    assert outcome.route.arm_id in {"A1", "A4", "A5"}
    assert outcome.route_ms >= 0.0


def test_corpus_router_without_a_graph_is_a_loud_error(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    with pytest.raises(RuntimeError, match="no graph was provided"):
        _executor(tmp_path, es)("A6", CASE)


def test_the_router_and_graph_expansion_share_one_extractor(tmp_path: Path) -> None:
    """If they disagreed about the query's entities, the routing decision and
    the expansion it triggers would be about different questions."""
    seen: list[str] = []

    def extractor(query: str) -> list[str]:
        seen.append(query)
        return ["Acme", "Northwind"]

    executor = _executor(tmp_path, FakeEs([f"c{i}" for i in range(10)]), graph=_graph())
    executor.extractor = extractor
    executor("A6", CASE)
    assert seen, "the injected extractor was bypassed"


# --------------------------------------------------------------------------
# the corrective retry
# --------------------------------------------------------------------------


def test_retry_does_not_fire_when_evidence_is_sufficient(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es, graph=_graph())("A6", CASE)
    assert outcome.retried is False
    assert outcome.attempts == 1


def test_retry_fires_and_widens(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    executor = _executor(
        tmp_path, es, graph=_graph(), sufficient=lambda q, chunks: len(chunks) > 5000
    )
    outcome = executor("A6", CASE)
    assert outcome.retried is True
    assert outcome.attempts == 2
    assert es.searches[-1]["size"] == 50  # widen_k


def test_retry_latency_is_the_final_attempt_only(tmp_path: Path) -> None:
    """Summing both attempts contradicts "timed on the final successful
    request" and would inflate exactly the two arms that retry."""
    es = FakeEs([f"c{i}" for i in range(10)])
    executor = _executor(
        tmp_path, es, graph=_graph(), sufficient=lambda q, chunks: len(chunks) > 5000
    )
    outcome = executor("A6", CASE)
    assert outcome.retrieval.retrieve_ms == pytest.approx(1.0)
    assert outcome.retry_overhead_ms > 0.0


def test_retry_can_be_disabled_by_a_permissive_check(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es, graph=_graph(), sufficient=lambda q, chunks: True)("A6", CASE)
    assert outcome.retried is False


def test_coverage_sufficiency_only_detects_total_failure(tmp_path: Path) -> None:
    """Documented limitation: it asks "did anything come back", not "is this
    enough to ground an answer". An NLI check is the real one."""
    check = coverage_sufficiency()
    assert check("q", ()) is False
    assert check("q", (RetrievedChunk("c1", "t", 1.0),)) is True


# --------------------------------------------------------------------------
# plain retrieval arms
# --------------------------------------------------------------------------


def test_retrieval_arm_passes_k_through(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(20)])
    _executor(tmp_path, es)("A1", CASE)
    assert es.searches[0]["size"] == 10


def test_generation_receives_the_retrieved_chunks(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    executor = _executor(tmp_path, es)
    outcome = executor("A1", CASE)
    generator: Any = executor.generator
    assert generator.prompts[0][1] == outcome.retrieval.chunks


def test_citations_are_scored_when_a_judge_is_supplied(tmp_path: Path) -> None:
    es = FakeEs(["g1"] + [f"c{i}" for i in range(9)])
    outcome = _executor(tmp_path, es, entails=lambda premise, hypothesis: True)("A1", CASE)
    assert outcome.citations.recall == pytest.approx(1.0)


def test_citations_are_undefined_without_a_judge(tmp_path: Path) -> None:
    es = FakeEs([f"c{i}" for i in range(10)])
    outcome = _executor(tmp_path, es)("A1", CASE)
    assert outcome.citations.precision is None
    assert outcome.citations.recall is None
