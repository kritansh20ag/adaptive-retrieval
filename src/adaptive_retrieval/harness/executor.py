"""Executes one arm against one question.

This is where the engine actually is: route, retrieve, optionally expand
through the graph, check the evidence, retry at most once, generate.

The sufficiency check deserves a note, because it is the second place an LLM
could quietly get into the hot path. It is deliberately a **cheap, non-model**
check by default - the same entailment judge used for citations, or a coverage
heuristic. An LLM call here would add ~400-600ms to every query, which is the
same mistake as putting one in the router and would undo the cost argument the
whole project rests on.

The retry is capped at one by config, and the cap is structural rather than a
default: an uncapped corrective loop is how a 400ms query becomes a 30s one,
and p95 is a reported metric.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from adaptive_retrieval.config import (
    BenchmarkConfig,
    ClosedBookArm,
    RetrievalArm,
    RouterArm,
)
from adaptive_retrieval.es_client import EsClient
from adaptive_retrieval.generate import Generator
from adaptive_retrieval.golden import GoldenCase
from adaptive_retrieval.graph.store import GraphStore
from adaptive_retrieval.harness.runner import ArmOutcome
from adaptive_retrieval.judge import LlmQualityJudge, SecondOpinionEntailment
from adaptive_retrieval.metrics.citations import CitationScores, EntailmentFn, score_citations
from adaptive_retrieval.retrieval.base import RetrievalResult, RetrievedChunk
from adaptive_retrieval.retrieval.query_builder import FieldNames, build_search_body
from adaptive_retrieval.router.routers import (
    CorpusRouter,
    EntityExtractor,
    QueryRouter,
    RegexEntityExtractor,
    Router,
)

__all__ = [
    "ArmExecutor",
    "SufficiencyCheck",
    "coverage_sufficiency",
    "entailment_sufficiency",
]

#: ``(question, chunks) -> evidence is enough to ground an answer``.
SufficiencyCheck = Callable[[str, tuple[RetrievedChunk, ...]], bool]


def coverage_sufficiency(min_chunks: int = 1) -> SufficiencyCheck:
    """Did we retrieve anything at all.

    **Not the default, and not a real sufficiency check.** It fires only when
    retrieval returned nothing - and re-issuing the same query with a larger
    ``size`` cannot make a query that matched zero documents match some. As the
    default it made the corrective retry a guaranteed no-op, and ``retry_rate``
    would have read ~0.000, which a reader would take as "the evidence was
    almost always sufficient" rather than "the check can only detect total
    failure". Kept for the cost-free smoke path; use
    :func:`entailment_sufficiency` for a real run.
    """

    def check(question: str, chunks: tuple[RetrievedChunk, ...]) -> bool:
        return len(chunks) >= min_chunks

    return check


def entailment_sufficiency(
    entails: EntailmentFn, *, top_n: int = 5, min_chunks: int = 1
) -> SufficiencyCheck:
    """Is the retrieved evidence enough to ground an answer to this question.

    Asks the entailment judge whether the top passages support the question's
    subject matter at all. This is the check the retry is supposed to be
    driven by: it measures groundedness, not cardinality, so a retrieval that
    returned ten irrelevant chunks correctly triggers a broader retry.

    Deliberately reuses the citation judge rather than making an LLM call - an
    NLI model runs in ~30-60ms, where an LLM would add 400-600ms to every
    query and undo the cost argument the whole project rests on.
    """

    def check(question: str, chunks: tuple[RetrievedChunk, ...]) -> bool:
        if len(chunks) < min_chunks:
            return False
        premise = "\n\n".join(chunk.text for chunk in chunks[:top_n])
        return entails(premise, question)

    return check


@dataclass
class ArmExecutor:
    """Runs any arm the config defines."""

    config: BenchmarkConfig
    es: EsClient
    generator: Generator
    graph: GraphStore | None = None
    entails: EntailmentFn | None = None
    #: Answer-quality judge. Metered separately from the arm's own cost.
    quality: LlmQualityJudge | None = None
    #: Records the primary/second-opinion disagreement rate, if one is wired.
    second_opinion: SecondOpinionEntailment | None = None
    sufficient: SufficiencyCheck | None = None
    fields: FieldNames | None = None
    graph_hops: int = 2
    #: How many of the k scored slots graph evidence may occupy. Published with
    #: the results: it is the rule that decides how much say the graph gets.
    graph_slots: int = 3
    #: Shared with the router, so the routing decision and the expansion it
    #: triggers cannot disagree about what the query's entities are.
    extractor: EntityExtractor = field(default_factory=RegexEntityExtractor)

    def __post_init__(self) -> None:
        self._routers: dict[str, Router] = {}
        self._sufficient = self.sufficient or coverage_sufficiency()
        if not 1 <= self.graph_slots <= self.config.defaults.k:
            raise ValueError(
                f"graph_slots must be in [1, k={self.config.defaults.k}], got {self.graph_slots}"
            )

    # -- routing -----------------------------------------------------------

    def _router_for(self, arm: RouterArm) -> Router:
        if arm.id in self._routers:
            return self._routers[arm.id]
        if arm.signal == "corpus":
            if self.graph is None:
                raise RuntimeError(
                    f"arm {arm.id!r} routes on the corpus signal but no graph was provided; "
                    f"build the graph first or run only the query-signal arms"
                )
            router: Router = CorpusRouter(list(arm.routes), self.graph, extractor=self.extractor)
        else:
            router = QueryRouter(list(arm.routes))
        self._routers[arm.id] = router
        return router

    # -- retrieval ---------------------------------------------------------

    def _retrieve(self, arm: RetrievalArm, question: str, k: int) -> RetrievalResult:
        body = build_search_body(arm, question, k, self.fields)
        result = self.es.search(body)

        if not arm.graph_expansion:
            return result
        if self.graph is None:
            raise RuntimeError(f"arm {arm.id!r} requests graph expansion but no graph was provided")

        # Graph expansion is a SECOND retrieval, timed separately. Folding it
        # into the first would hide where the extra cost is paid, which is the
        # question A5 exists to answer.
        started = time.perf_counter()
        seeds = self.extractor(question)
        # Over-fetch, then de-duplicate against what Elasticsearch already
        # returned, then truncate. Truncating before de-duplication lets
        # already-retrieved chunks consume the whole graph budget.
        expanded_ids = self.graph.expand(seeds, hops=self.graph_hops, limit=k * 3)
        already = {chunk.chunk_id for chunk in result.chunks}
        new_ids = [cid for cid in expanded_ids if cid not in already][: self.graph_slots]
        # The graph returns ids; the text still lives only in Elasticsearch.
        extra = self.es.fetch_chunks(new_ids)
        graph_ms = (time.perf_counter() - started) * 1000.0

        # THE POINT. Retrieval metrics are computed at k, so graph evidence
        # appended after position k is sliced off before scoring and A5 becomes
        # arithmetically identical to A4 - the benchmark could not detect its
        # own hypothesis. Graph chunks are interleaved into the scored window
        # instead: the top (k - graph_slots) lexical hits keep their order and
        # the graph's best candidates take the remaining slots.
        keep = max(k - len(extra), 0)
        merged = (result.chunks[:keep] + extra)[:k]

        return replace(
            result,
            chunks=merged,
            graph_ms=graph_ms,
            graph_chunk_ids=tuple(chunk.chunk_id for chunk in extra),
        )

    # -- the arm -----------------------------------------------------------

    def __call__(self, arm_id: str, case: GoldenCase) -> ArmOutcome:
        arm = self.config.resolved_arm(arm_id)
        k = self.config.defaults.k

        route_ms = 0.0
        retry_overhead_ms = 0.0
        route = None
        retried = False
        attempts = 1

        if isinstance(arm, ClosedBookArm):
            # A0: no retrieval at all. Controls for questions the model can
            # answer from parametric memory, which would otherwise inflate
            # every other arm's apparent contribution.
            retrieval = RetrievalResult()

        elif isinstance(arm, RouterArm):
            started = time.perf_counter()
            decision = self._router_for(arm).route(case.question)
            route_ms = (time.perf_counter() - started) * 1000.0
            route = decision

            target = self.config.resolved_arm(decision.arm_id)
            assert isinstance(target, RetrievalArm)  # guaranteed by config validation
            retrieval = self._retrieve(target, case.question, k)

            if arm.retry.max > 0 and not self._sufficient(case.question, retrieval.chunks):
                # One retry, broader. Capped by config, because an uncapped
                # loop is how p95 dies.
                retried = True
                attempts = 2
                first_ms = retrieval.total_ms
                # The FINAL attempt's latency only. Summing both attempts into
                # the retrieval column contradicts "timed on the final
                # successful request", and it would inflate exactly the two
                # arms that retry - A6 and A7, the contribution arms. The
                # discarded attempt is reported as retry_overhead_ms instead,
                # so the retried and non-retried distributions stay separable.
                retrieval = self._retrieve(target, case.question, arm.retry.widen_k)
                retry_overhead_ms = first_ms

        elif isinstance(arm, RetrievalArm):
            retrieval = self._retrieve(arm, case.question, k)

        else:  # pragma: no cover - config validation makes this unreachable
            raise TypeError(f"unknown arm kind for {arm_id!r}")

        generation = self.generator.answer(case.question, retrieval.chunks)

        chunk_texts = {chunk.chunk_id: chunk.text for chunk in retrieval.chunks}
        if self.second_opinion is not None:
            self.second_opinion.reset()
        citations = (
            score_citations(generation.statements, chunk_texts, self.entails)
            if self.entails is not None
            else CitationScores(precision=None, recall=None, n_statements=0, n_citations=0)
        )

        answer_quality = None
        judge_usage = None
        if self.quality is not None and not generation.payload.abstained:
            answer_text = " ".join(s.text for s in generation.payload.sentences)
            answer_quality = self.quality.judge(
                case.question, answer_text, [c.text for c in retrieval.chunks]
            )
            judge_usage = self.quality.usage

        return ArmOutcome(
            retrieval=retrieval,
            generation=generation,
            citations=citations,
            route=route,
            retried=retried,
            attempts=attempts,
            route_ms=route_ms,
            retry_overhead_ms=retry_overhead_ms,
            faithful=answer_quality.faithful if answer_quality else None,
            relevant=answer_quality.relevant if answer_quality else None,
            judge_model=judge_usage.model if judge_usage else None,
            judge_cost_usd=judge_usage.cost_usd if judge_usage else 0.0,
            judge_input_tokens=judge_usage.input_tokens if judge_usage else 0,
            judge_output_tokens=judge_usage.output_tokens if judge_usage else 0,
            judge_disagreement=(
                self.second_opinion.disagreed if self.second_opinion is not None else None
            ),
        )
