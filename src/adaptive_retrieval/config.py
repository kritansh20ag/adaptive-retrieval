"""Load and validate ``benchmark.yaml``.

The harness's central promise is that an arm is selected by config and nothing
else - no code changes between runs. That promise is only worth anything if a
malformed config fails loudly instead of silently running something other than
what was written down. Every model here sets ``extra="forbid"``, so a typo in
a key is an error rather than a silently ignored setting.

Constraints marked "[ES]" are Elasticsearch's own documented requirements,
verified against the retriever reference on 2026-09-05. Constraints marked
"[ours]" are this project's, and say why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from adaptive_retrieval.chunking import ChunkingConfig

__all__ = [
    "ArmSpec",
    "BenchmarkConfig",
    "ConfigError",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a config file is malformed or internally inconsistent."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


class ChunkingSettings(_Base):
    """Chunking parameters, held constant across every arm.

    Converted to a :class:`~adaptive_retrieval.chunking.ChunkingConfig` by
    :meth:`to_chunking_config`, so there is exactly one place these bounds are
    enforced rather than two definitions drifting apart.
    """

    strategy: Literal["sentence"] = "sentence"
    # [ES] sentence strategy: max_chunk_size >= 20 words, default 250.
    max_words: int = Field(default=250, ge=20)
    # [ES] sentence_overlap is 0 or 1. [ours] We default to 0, unlike Elastic:
    # overlap puts every sentence in two chunks with two different IDs, which
    # makes gold_chunks ambiguous and duplicates entities into the graph,
    # inflating the node degree that A6's routing signal is built on.
    sentence_overlap: Literal[0, 1] = 0
    # [ours] Hard character ceiling. A script without spaces has a word count of
    # 1 however long it is, and an over-long chunk is truncated by the embedding
    # models while BM25 still scores all of it - so A1 and A2 would see
    # different amounts of the same chunk.
    max_chars: int = Field(default=2000, ge=100)

    def to_chunking_config(self) -> ChunkingConfig:
        """Build the runtime chunker config, enforcing the shared bounds."""
        return ChunkingConfig(
            max_words=self.max_words,
            sentence_overlap=self.sentence_overlap,
            max_chars=self.max_chars,
        )


# --------------------------------------------------------------------------
# retrievers
# --------------------------------------------------------------------------

LeafSource = Literal["bm25", "dense", "elser"]


class Bm25Retriever(_Base):
    type: Literal["bm25"]
    field: str = "text"


class DenseRetriever(_Base):
    type: Literal["dense"]
    field: str = "text_semantic"


class ElserRetriever(_Base):
    type: Literal["elser"]
    field: str = "text_elser"


class RrfFusion(_Base):
    """Reciprocal rank fusion over two or more leaf retrievers."""

    type: Literal["rrf"]
    sources: list[LeafSource]
    # [ES] rank_constant must be >= 1; Elasticsearch defaults it to 60.
    rank_constant: int = Field(default=60, ge=1)
    # [ours] No default. Elasticsearch defaults rank_window_size to 10, which is
    # far too shallow for a retrieval benchmark and would silently cripple every
    # fusion arm. Requiring it here forces the value to be chosen and published.
    rank_window_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _at_least_two_sources(self) -> RrfFusion:
        # [ours] Fusing a single ranked list is a no-op that would quietly
        # masquerade as a hybrid arm.
        if len(self.sources) < 2:
            raise ValueError(f"rrf needs at least 2 sources, got {self.sources}")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"rrf sources must be unique, got {self.sources}")
        return self


class LinearFusion(_Base):
    """Score-normalised weighted fusion - the published alternative to RRF."""

    type: Literal["linear"]
    sources: list[LeafSource]
    # [ES] normalizer is one of none | minmax | l2_norm.
    normalizer: Literal["none", "minmax", "l2_norm"] = "minmax"
    # [ES] per-retriever weight must be >= 0; Elasticsearch defaults it to 1.0.
    weights: dict[LeafSource, float] = Field(default_factory=dict)
    rank_window_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_sources_and_weights(self) -> LinearFusion:
        if len(self.sources) < 2:
            raise ValueError(f"linear needs at least 2 sources, got {self.sources}")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"linear sources must be unique, got {self.sources}")
        unknown = set(self.weights) - set(self.sources)
        if unknown:
            raise ValueError(f"weights reference sources not in this arm: {sorted(unknown)}")
        negative = sorted(k for k, v in self.weights.items() if v < 0)
        if negative:
            raise ValueError(
                f"weights must be >= 0 (Elasticsearch requirement); negative: {negative}"
            )
        return self


RetrieverSpec = Annotated[
    Bm25Retriever | DenseRetriever | ElserRetriever | RrfFusion | LinearFusion,
    Field(discriminator="type"),
]


class RerankSpec(_Base):
    """Second-stage cross-encoder reranking."""

    # [ES] text_similarity_reranker defaults inference_id to
    # ".rerank-v1-elasticsearch"; we require it explicitly so the run records
    # which reranker produced the numbers.
    inference_id: str
    field: str = "text"
    # [ours] No default, for the same reason as fusion: Elasticsearch defaults
    # this to 10, and rerank depth is the single biggest driver of both the
    # quality gain and the cost of this stage.
    depth: int = Field(ge=1)
    min_score: float | None = None


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


class RetrySpec(_Base):
    """Corrective retry. Capped, because an uncapped loop destroys p95."""

    max: int = Field(default=1, ge=0, le=1)
    widen_k: int = Field(ge=1)


class ClosedBookArm(_Base):
    """A0: no retrieval at all. Controls for questions answerable from memory."""

    id: str
    kind: Literal["closed_book"]


class RetrievalArm(_Base):
    id: str
    kind: Literal["retrieval"]
    retriever: RetrieverSpec
    rerank: RerankSpec | None = None
    graph_expansion: bool = False


class RouterArm(_Base):
    id: str
    kind: Literal["router"]
    #: ``corpus`` probes the graph for entity presence and local connectivity.
    #: ``query`` classifies on query phrasing only - the baseline A6 must beat.
    signal: Literal["corpus", "query"]
    routes: list[str]
    retry: RetrySpec

    @model_validator(mode="after")
    def _routes_non_empty(self) -> RouterArm:
        if len(self.routes) < 2:
            raise ValueError(f"a router needs at least 2 routes, got {self.routes}")
        if len(set(self.routes)) != len(self.routes):
            raise ValueError(f"router routes must be unique, got {self.routes}")
        return self


ArmSpec = Annotated[
    ClosedBookArm | RetrievalArm | RouterArm,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


class GeneratorSettings(_Base):
    """Identical for every arm, so no arm can win by using a better model."""

    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = Field(default=4096, ge=1)


class JudgeSettings(_Base):
    #: Primary judge. An NLI entailment model, not an LLM - deterministic and
    #: not subject to position/verbosity/self-preference bias.
    entailment: str
    #: Second opinion. Disagreement between the two is recorded and reported.
    second_opinion: str | None = None


class Defaults(_Base):
    index: str
    k: int = Field(ge=1)


class RunSettings(_Base):
    # [ours] One rep is a point estimate with no error bar. Allowed, but the
    # loader warns, because differences smaller than run-to-run spread are not
    # meaningful and a single-rep run cannot show that.
    reps: int = Field(default=2, ge=1)
    seed: int = 20260905
    #: Run question-major (every arm sees question 1, then question 2, ...) so
    #: API load and cache warmth spread evenly instead of landing on one arm.
    interleave: bool = True
    max_case_seconds: float = Field(default=180.0, gt=0)


class BenchmarkConfig(_Base):
    corpus: str
    golden_set: str
    generator: GeneratorSettings = GeneratorSettings()
    judges: JudgeSettings
    defaults: Defaults
    chunking: ChunkingSettings = ChunkingSettings()
    run: RunSettings = RunSettings()
    arms: list[ArmSpec]

    @model_validator(mode="after")
    def _check_arms(self) -> BenchmarkConfig:
        if not self.arms:
            raise ValueError("at least one arm must be defined")

        ids = [arm.id for arm in self.arms]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate arm ids: {duplicates}")

        known = set(ids)
        for arm in self.arms:
            if isinstance(arm, RouterArm):
                missing = [r for r in arm.routes if r not in known]
                if missing:
                    raise ValueError(
                        f"arm {arm.id!r} routes to undefined arm(s): {missing}. "
                        f"Defined arms: {sorted(known)}"
                    )
                if arm.id in arm.routes:
                    raise ValueError(f"arm {arm.id!r} routes to itself")

            # [ES] rank_window_size must be >= the requested size (`k`).
            if isinstance(arm, RetrievalArm):
                retriever = arm.retriever
                if (
                    isinstance(retriever, RrfFusion | LinearFusion)
                    and retriever.rank_window_size < self.defaults.k
                ):
                    raise ValueError(
                            f"arm {arm.id!r}: rank_window_size "
                            f"({retriever.rank_window_size}) must be >= defaults.k "
                            f"({self.defaults.k}) - Elasticsearch rejects a window "
                            f"smaller than the requested size"
                        )
                if arm.rerank is not None and arm.rerank.depth < self.defaults.k:
                    raise ValueError(
                        f"arm {arm.id!r}: rerank.depth ({arm.rerank.depth}) must be "
                        f">= defaults.k ({self.defaults.k})"
                    )
        return self

    def arm(self, arm_id: str) -> ArmSpec:
        for candidate in self.arms:
            if candidate.id == arm_id:
                return candidate
        raise KeyError(f"no arm with id {arm_id!r}; defined: {[a.id for a in self.arms]}")


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark config, raising ``ConfigError`` on any problem."""
    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(
            f"{config_path} must contain a YAML mapping at the top level, "
            f"got {type(parsed).__name__}"
        )

    try:
        return BenchmarkConfig.model_validate(parsed)
    except ValidationError as exc:
        raise ConfigError(f"{config_path} is invalid:\n{exc}") from exc
