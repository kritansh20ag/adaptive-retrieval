"""Assembles a runnable harness from config.

The only module that knows how every other piece fits together, kept separate
so ``ArmExecutor`` stays testable against stubs and the CLI stays thin.

It also builds the two baselines the eval guidance requires **before** any paid
pass:

* the **oracle** - feed the gold chunks straight in. Should approach a perfect
  score. If it does not, the grader is broken, not the model.
* the **null** - return nothing and answer nothing. Should score zero. If it
  does not, the grader is too lenient.

Two runs, minutes of compute, and they catch most wiring bugs before money is
spent. Both are built here rather than improvised at the call site, because a
smoke test assembled differently from the real run tests a different system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_retrieval.config import BenchmarkConfig
from adaptive_retrieval.generate import AnswerPayload, CitedSentence, GenerationResult
from adaptive_retrieval.golden import GoldenCase
from adaptive_retrieval.graph.store import GraphStore, InMemoryGraph
from adaptive_retrieval.harness.runner import ArmOutcome
from adaptive_retrieval.metrics.citations import CitationScores, EntailmentFn, score_citations
from adaptive_retrieval.retrieval.base import RetrievalResult, RetrievedChunk

__all__ = ["NullArm", "OracleArm", "load_graph"]


def load_graph(path: str | None) -> GraphStore | None:
    """Load a saved graph, or return ``None`` when the run has no graph."""
    if not path:
        return None
    return InMemoryGraph.load(path)


@dataclass
class OracleArm:
    """Hands the grader the gold chunks and a perfect answer.

    Not a model and not a retriever - a probe of the *grading pipeline*. If
    this does not score near 1.0, the fault is in the metrics, the golden set
    or the judge, and no amount of retrieval work will fix it.
    """

    config: BenchmarkConfig
    entails: EntailmentFn | None = None

    def __call__(self, arm_id: str, case: GoldenCase) -> ArmOutcome:
        chunks = tuple(
            RetrievedChunk(chunk_id=cid, text=case.answer or "", score=1.0)
            for cid in case.gold_chunks
        )
        sentences = (
            []
            if case.should_abstain
            else [CitedSentence(text=case.answer or "", cited_chunk_ids=list(case.gold_chunks))]
        )
        payload = AnswerPayload(abstained=case.should_abstain, sentences=sentences)
        generation = GenerationResult(
            payload=payload,
            served_model=self.config.generator.model,
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=0.001,
            cost_usd=0.0,
        )
        chunk_texts = {c.chunk_id: c.text for c in chunks}
        citations = (
            score_citations(generation.statements, chunk_texts, self.entails)
            if self.entails is not None
            else CitationScores(precision=None, recall=None, n_statements=0, n_citations=0)
        )
        return ArmOutcome(
            retrieval=RetrievalResult(chunks=chunks, retrieve_ms=0.001),
            generation=generation,
            citations=citations,
        )


@dataclass
class NullArm:
    """Retrieves nothing and answers nothing.

    Deliberately abstains on every question, including answerable ones - so a
    grader that scores this above zero on retrieval, or above chance on
    abstention, is too lenient.
    """

    config: BenchmarkConfig

    def __call__(self, arm_id: str, case: GoldenCase) -> ArmOutcome:
        generation = GenerationResult(
            payload=AnswerPayload(abstained=True, sentences=[]),
            served_model=self.config.generator.model,
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=0.001,
            cost_usd=0.0,
        )
        return ArmOutcome(
            retrieval=RetrievalResult(retrieve_ms=0.001),
            generation=generation,
            citations=CitationScores(precision=None, recall=None, n_statements=0, n_citations=0),
        )


def smoke_verdict(summaries: list[Any]) -> tuple[bool, list[str]]:
    """Judge an oracle/null smoke run. Returns ``(passed, complaints)``."""
    complaints: list[str] = []
    by_arm = {s.arm: s for s in summaries}

    oracle = by_arm.get("ORACLE")
    if oracle is None:
        complaints.append("no ORACLE rows were produced")
    elif oracle.ndcg is None or oracle.ndcg < 0.99:
        complaints.append(
            f"oracle nDCG is {oracle.ndcg} - the grader cannot score a perfect answer, "
            f"so the fault is in the metrics, the golden set or the judge"
        )

    null = by_arm.get("NULL")
    if null is None:
        complaints.append("no NULL rows were produced")
    elif null.ndcg not in (None, 0.0):
        complaints.append(
            f"null nDCG is {null.ndcg} - the grader rewards an empty answer, so it is "
            f"too lenient to distinguish any arm from nothing"
        )
    return (not complaints, complaints)
