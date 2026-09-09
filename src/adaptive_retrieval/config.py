"""Load and validate ``benchmark.yaml``.

The harness's central promise is that an arm is selected by config and nothing
else - no code changes between runs. That promise is only worth anything if a
malformed config fails loudly instead of silently running something other than
what was written down. Every model sets ``extra="forbid"``, so a typo in a key
is an error rather than a silently ignored setting.

Constraints marked "[ES]" are Elasticsearch's own documented requirements,
verified against the retriever reference on 2026-09-05. Constraints marked
"[ours]" are this project's, and say why.

Sequence fields are tuples, not lists. ``frozen=True`` in pydantic blocks
attribute assignment but not mutation of a list *inside* the model - so with
lists, ``config.arms.append(...)`` silently bypassed every validator that had
already run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Literal, Self

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
    """Chunking parameters, held constant across every arm."""

    strategy: Literal["sentence"] = "sentence"
    # [ES] sentence strategy: max_chunk_size >= 20 words, default 250.
    max_words: int = Field(default=250, ge=20)
    # [ES] sentence_overlap is 0 or 1. [ours] We default to 0, unlike Elastic:
    # overlap puts every sentence in two chunks with two different IDs, which
    # makes gold_chunks ambiguous and duplicates entities into the graph,
    # inflating the node degree A6's routing signal is built on.
    sentence_overlap: Literal[0, 1] = 0
    # [ours] Hard character ceiling. A script without spaces has a word count of
    # 1 however long it is, and an over-long chunk is truncated by the embedding
    # models while BM25 still scores all of it.
    max_chars: int = Field(default=2000, ge=100)

    def to_chunking_config(self) -> ChunkingConfig:
        """Build the runtime chunker config, enforcing the shared bounds once."""
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
    sources: tuple[LeafSource, ...]
    # [ES] rank_constant must be >= 1; Elasticsearch defaults it to 60.
    rank_constant: int = Field(default=60, ge=1)
    # [ours] No default. Elasticsearch defaults rank_window_size to 10, far too
    # shallow for a retrieval benchmark; it must be chosen and published.
    rank_window_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _at_least_two_sources(self) -> Self:
        if len(self.sources) < 2:
            raise ValueError(f"rrf needs at least 2 sources, got {list(self.sources)}")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"rrf sources must be unique, got {list(self.sources)}")
        return self


class LinearFusion(_Base):
    """Score-normalised weighted fusion - the published alternative to RRF."""

    type: Literal["linear"]
    sources: tuple[LeafSource, ...]
    # [ES] normalizer is one of none | minmax | l2_norm, and Elasticsearch
    # defaults it to `none`. [ours] No default here, for the same reason as
    # rank_window_size: the normalizer is the biggest determinant of a linear
    # fusion's behaviour - with `none`, raw BM25 scores dwarf cosine
    # similarities and this arm degenerates toward BM25.
    normalizer: Literal["none", "minmax", "l2_norm"]
    # [ES] per-retriever weight must be >= 0; Elasticsearch defaults it to 1.0.
    weights: dict[LeafSource, float] = Field(default_factory=dict)
    rank_window_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_sources_and_weights(self) -> Self:
        if len(self.sources) < 2:
            raise ValueError(f"linear needs at least 2 sources, got {list(self.sources)}")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"linear sources must be unique, got {list(self.sources)}")
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

FusionSpec = RrfFusion | LinearFusion


class RerankSpec(_Base):
    """Second-stage cross-encoder reranking."""

    inference_id: str
    field: str = "text"
    # [ours] No default: Elasticsearch's is 10, and rerank depth is the single
    # biggest driver of both the quality gain and the cost of this stage.
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
    """A retrieval arm, optionally defined as a delta against another arm.

    ``extends`` is what makes "each arm adds exactly one thing to the one above
    it, so any difference is attributable" enforceable rather than aspirational.
    Duplicating a parent's retriever block by hand lets the two drift, and then
    a window-size difference gets reported as a knowledge-graph effect.
    """

    id: str
    kind: Literal["retrieval"]
    #: Inherit ``retriever`` and ``rerank`` from another retrieval arm.
    extends: str | None = None
    retriever: RetrieverSpec | None = None
    rerank: RerankSpec | None = None
    graph_expansion: bool = False

    @model_validator(mode="after")
    def _delta_or_definition(self) -> Self:
        if self.extends is None and self.retriever is None:
            raise ValueError(f"arm {self.id!r} must define a retriever or extend another arm")
        if self.extends is not None:
            if self.retriever is not None:
                raise ValueError(
                    f"arm {self.id!r} extends {self.extends!r} but also defines its own "
                    f"retriever; an extending arm may only add its one declared difference"
                )
            if self.rerank is not None:
                raise ValueError(
                    f"arm {self.id!r} extends {self.extends!r} but also defines its own rerank"
                )
        return self


class RouterArm(_Base):
    id: str
    kind: Literal["router"]
    #: ``corpus`` probes the graph for entity presence and local connectivity.
    #: ``query`` classifies on query phrasing only - the baseline A6 must beat.
    signal: Literal["corpus", "query"]
    routes: tuple[str, ...]
    retry: RetrySpec

    @model_validator(mode="after")
    def _routes_are_sane(self) -> Self:
        if len(self.routes) < 2:
            raise ValueError(f"a router needs at least 2 routes, got {list(self.routes)}")
        if len(set(self.routes)) != len(self.routes):
            raise ValueError(f"router routes must be unique, got {list(self.routes)}")
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
    #: free of position/verbosity/self-preference bias.
    entailment: str
    #: Second opinion. Disagreement between the two is recorded and reported.
    second_opinion: str | None = None


class Defaults(_Base):
    index: str
    k: int = Field(ge=1)


class RunSettings(_Base):
    reps: int = Field(default=2, ge=1)
    seed: int = 20260905
    #: Question-major, so API load and cache warmth spread evenly across arms
    #: instead of landing on whichever arm ran first.
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
    arms: tuple[ArmSpec, ...]

    # -- lookups -----------------------------------------------------------

    def arm(self, arm_id: str) -> ArmSpec:
        for candidate in self.arms:
            if candidate.id == arm_id:
                return candidate
        raise KeyError(f"no arm with id {arm_id!r}; defined: {[a.id for a in self.arms]}")

    def resolved_arm(self, arm_id: str) -> ArmSpec:
        """Return an arm with any ``extends`` inheritance applied."""
        target = self.arm(arm_id)
        if not isinstance(target, RetrievalArm) or target.extends is None:
            return target
        parent = self.resolved_arm(target.extends)
        assert isinstance(parent, RetrievalArm)  # guaranteed by _check_arms
        return target.model_copy(
            update={
                "retriever": parent.retriever,
                "rerank": parent.rerank,
                "extends": None,
            }
        )

    # -- validation --------------------------------------------------------

    @model_validator(mode="after")
    def _check_arms(self) -> Self:
        if not self.arms:
            raise ValueError("at least one arm must be defined")

        ids = [arm.id for arm in self.arms]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate arm ids: {duplicates}")
        known = set(ids)

        # [ours] A0 is the denominator for every reported number - every arm's
        # result is a lift over closed book, not over zero.
        if not any(isinstance(arm, ClosedBookArm) for arm in self.arms):
            raise ValueError(
                "no closed_book arm defined. Every arm's result is reported as lift over "
                "closed book, which controls for questions answerable from parametric "
                "memory; without it there is no denominator."
            )

        # [ours] A model may not judge its own output: self-preference bias
        # lands directly on faithfulness, relevance and judge_disagreement.
        if self.judges.second_opinion == self.generator.model:
            raise ValueError(
                f"judges.second_opinion ({self.judges.second_opinion!r}) is the model under "
                f"test (generator.model). A model may not judge its own output."
            )

        self._check_extends(known)
        self._check_retrieval_arms()
        self._check_routers(known)
        return self

    def _check_extends(self, known: set[str]) -> None:
        for arm in self.arms:
            if not isinstance(arm, RetrievalArm) or arm.extends is None:
                continue
            if arm.extends not in known:
                raise ValueError(
                    f"arm {arm.id!r} extends undefined arm {arm.extends!r}; "
                    f"defined: {sorted(known)}"
                )
            parent = self.arm(arm.extends)
            if not isinstance(parent, RetrievalArm):
                raise ValueError(
                    f"arm {arm.id!r} extends {arm.extends!r}, which is a "
                    f"{parent.kind!r} arm, not a retrieval arm"
                )
            # Walk the chain to catch a cycle before resolution recurses forever.
            seen = {arm.id}
            cursor: RetrievalArm | None = parent
            while cursor is not None and cursor.extends is not None:
                if cursor.id in seen:
                    raise ValueError(f"arm {arm.id!r} has a cyclic extends chain")
                seen.add(cursor.id)
                nxt = self.arm(cursor.extends)
                cursor = nxt if isinstance(nxt, RetrievalArm) else None

    def _check_retrieval_arms(self) -> None:
        for arm in self.arms:
            if not isinstance(arm, RetrievalArm):
                continue
            resolved = self.resolved_arm(arm.id)
            assert isinstance(resolved, RetrievalArm)
            retriever = resolved.retriever
            window = retriever.rank_window_size if isinstance(retriever, FusionSpec) else None

            # [ES] rank_window_size must be >= the requested size (`k`).
            if window is not None and window < self.defaults.k:
                raise ValueError(
                    f"arm {arm.id!r}: rank_window_size ({window}) must be >= defaults.k "
                    f"({self.defaults.k}) - Elasticsearch rejects a window smaller than size"
                )
            if resolved.rerank is not None:
                if resolved.rerank.depth < self.defaults.k:
                    raise ValueError(
                        f"arm {arm.id!r}: rerank.depth ({resolved.rerank.depth}) must be "
                        f">= defaults.k ({self.defaults.k})"
                    )
                # [ours] The reranker can only see what its child produced.
                # Publishing depth=100 while the child yields 20 candidates
                # reports a depth-100 rerank that never happened.
                if window is not None and resolved.rerank.depth > window:
                    raise ValueError(
                        f"arm {arm.id!r}: rerank.depth ({resolved.rerank.depth}) exceeds the "
                        f"child retriever's rank_window_size ({window}); the reranker can only "
                        f"see the candidates the retriever returned"
                    )

    def _check_routers(self, known: set[str]) -> None:
        for arm in self.arms:
            if not isinstance(arm, RouterArm):
                continue
            missing = [r for r in arm.routes if r not in known]
            if missing:
                raise ValueError(
                    f"arm {arm.id!r} routes to undefined arm(s): {missing}. "
                    f"Defined arms: {sorted(known)}"
                )
            if arm.id in arm.routes:
                raise ValueError(f"arm {arm.id!r} routes to itself")

            for route in arm.routes:
                target = self.resolved_arm(route)
                # [ours] Routers must dispatch to concrete arms. A router
                # routing to a router has no cycle check and would recurse
                # until the stack blows, mid-run, after money has been spent.
                if isinstance(target, RouterArm):
                    raise ValueError(
                        f"arm {arm.id!r} routes to {route!r}, which is itself a router. "
                        f"Routers must dispatch to concrete arms."
                    )
                # [ours] The corrective retry must actually broaden. widen_k
                # below defaults.k silently narrows the retry to a fraction of
                # the original, and the arm reports worse-after-retry.
                if arm.retry.max > 0 and arm.retry.widen_k <= self.defaults.k:
                    raise ValueError(
                        f"arm {arm.id!r}: retry.widen_k ({arm.retry.widen_k}) must be > "
                        f"defaults.k ({self.defaults.k}) - a corrective retry must broaden, "
                        f"not narrow"
                    )
                # [ES] A widened k above the target's window is a 400 at query
                # time, mid-run, on the arm that is the contribution.
                if isinstance(target, RetrievalArm) and isinstance(target.retriever, FusionSpec):
                    window = target.retriever.rank_window_size
                    if arm.retry.max > 0 and arm.retry.widen_k > window:
                        raise ValueError(
                            f"arm {arm.id!r}: retry.widen_k ({arm.retry.widen_k}) exceeds "
                            f"route {route!r}'s rank_window_size ({window}); Elasticsearch "
                            f"requires rank_window_size >= size"
                        )


def _rewrite_arm_locations(exc: ValidationError, parsed: dict[str, object]) -> str:
    """Replace ``arms.3`` with ``arm 'A4'`` in error locations.

    On a nine-arm file, counting list entries to find the broken one is a
    needless step in front of a paid run.
    """
    raw_arms = parsed.get("arms")
    lines: list[str] = []
    for error in exc.errors():
        location = list(error["loc"])
        if len(location) >= 2 and location[0] == "arms" and isinstance(location[1], int):
            index = location[1]
            name = f"<index {index}>"
            if isinstance(raw_arms, list) and index < len(raw_arms):
                entry = raw_arms[index]
                if isinstance(entry, dict):
                    name = str(entry.get("id", name))
            location[:2] = [f"arm {name!r}"]
        lines.append(f"  {'.'.join(str(p) for p in location)}: {error['msg']}")
    return "\n".join(lines)


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
        config = BenchmarkConfig.model_validate(parsed)
    except ValidationError as exc:
        raise ConfigError(
            f"{config_path} is invalid:\n{_rewrite_arm_locations(exc, parsed)}"
        ) from exc

    if config.run.reps < 2:
        warnings.warn(
            f"run.reps is {config.run.reps}: a single rep is a point estimate with no error "
            f"bar, and differences smaller than the run-to-run spread will not be "
            f"distinguishable from noise.",
            UserWarning,
            stacklevel=2,
        )
    return config
