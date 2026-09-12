"""The two routers: A6 on the corpus signal, A7 on query phrasing.

They exist as a matched pair on purpose. Beating fixed strategies only proves
routing works, which is already established in the literature. The claim being
tested is narrower: that a **corpus** signal beats a **phrasing** signal. So A7
is not a strawman to be beaten - it is a serious baseline, and it is
implemented as well as A6 is.

Two published results make this a real contest rather than a formality: a
query-only DeBERTa classifier has been reported beating an entity-based NER
router, and TF-IDF + SVM reaches 93.2% on query-complexity routing, with the
authors noting that "surface keyword patterns are strong predictors". Entity
presence alone has already lost once. If A6 does not beat A7, the honest
finding is that phrasing was sufficient on this corpus - and the harness is
built to be able to show that.

Neither router calls an LLM. That is a hard constraint, not an optimisation:
an LLM here would add 300-800ms to every query to save ~600ms on some, which
inverts the entire argument for routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from adaptive_retrieval.graph.signals import RouteSignal, SignalThresholds, probe
from adaptive_retrieval.graph.store import GraphStore

__all__ = [
    "CorpusRouter",
    "EntityExtractor",
    "QueryRouter",
    "RegexEntityExtractor",
    "RouteDecision",
    "Router",
]


class EntityExtractor(Protocol):
    """Pulls candidate entity mentions out of a query. Must not use an LLM."""

    def __call__(self, query: str) -> list[str]: ...


# Capitalised runs, optionally joined by lowercase connectors ("Bank of Acme").
# A deliberately dumb NER: spaCy or Amazon Comprehend drops in behind the same
# protocol, and this keeps the package dependency-free and deterministic.
#
# "and" is NOT a connector. It appears in a few real names ("Johnson and
# Johnson") but overwhelmingly it coordinates two separate entities - and
# swallowing "Northwind and Globex" into one span halves the entity count,
# which directly deflates found_fraction and the density signal.
_CONNECTORS = "of|de|del|da|van|von|der"
_PROPER_NOUN = re.compile(
    rf"\b[A-Z][\w&'-]*(?:\s+(?:{_CONNECTORS})\s+[A-Z][\w&'-]*|\s+[A-Z][\w&'-]*)*"
)
# Words that get capitalised by grammar rather than by being names. Missing
# auxiliaries produced "Is Acme" and "Has Acme" as entities, which are never in
# the graph - and every such miss lowers found_fraction, pushing A6 AWAY from
# the graph route. The bias is directional, so it suppresses exactly the route
# A6 exists to choose.
_STOPWORDS = frozenset(
    {
        # interrogatives
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        # auxiliaries and copulas
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "shall",
        "may",
        "might",
        "must",
        # determiners and sentence-initial prepositions
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "after",
        "before",
        "during",
        "since",
        "until",
        "with",
        "without",
        "between",
        "and",
        "or",
        "but",
        "if",
        "as",
        "to",
        "of",
    }
)


class RegexEntityExtractor:
    """Capitalised-span extractor. Deterministic, dependency-free, ~microseconds."""

    def __call__(self, query: str) -> list[str]:
        found: list[str] = []
        for match in _PROPER_NOUN.finditer(query):
            # Strip trailing punctuation so "Acme." and "Acme" are one entity,
            # matching how the graph normalises names.
            candidate = match.group(0).strip().strip(".,;:!?\"'")
            # Drop a leading sentence-initial question word, which is
            # capitalised by grammar rather than by being a name.
            first, _, rest = candidate.partition(" ")
            if first.casefold() in _STOPWORDS:
                candidate = rest.strip()
            if candidate and candidate.casefold() not in _STOPWORDS:
                found.append(candidate)
        # Order-preserving de-duplication: the signal counts entities, and
        # repeating one would inflate found_fraction.
        seen: set[str] = set()
        unique: list[str] = []
        for name in found:
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(name)
        return unique


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Which arm to run, and enough of why to audit it afterwards."""

    arm_id: str
    reason: str
    signal: dict[str, object]


class Router(Protocol):
    def route(self, query: str) -> RouteDecision: ...


@dataclass(frozen=True, slots=True)
class _Routes:
    """The three tiers a router chooses between, cheapest first."""

    lexical: str
    hybrid: str
    graph: str

    @classmethod
    def from_list(cls, routes: list[str]) -> _Routes:
        if len(routes) != 3:
            raise ValueError(
                f"a router needs exactly 3 routes (lexical, hybrid, graph), got {routes}"
            )
        return cls(lexical=routes[0], hybrid=routes[1], graph=routes[2])


class CorpusRouter:
    """A6. Routes on what the graph knows about this query's entities.

    The decision is a structural lookup and nothing else - no LLM, no
    embedding, no retrieval. Prior work computes corpus-structural features
    once per corpus, which collapses into a fixed per-dataset mapping; this
    computes them for this query's own neighbourhood at query time.
    """

    def __init__(
        self,
        routes: list[str],
        graph: GraphStore,
        *,
        extractor: EntityExtractor | None = None,
        thresholds: SignalThresholds | None = None,
    ) -> None:
        self.routes = _Routes.from_list(routes)
        self.graph = graph
        self.extractor = extractor or RegexEntityExtractor()
        self.thresholds = thresholds or SignalThresholds()

    def route(self, query: str) -> RouteDecision:
        entities = self.extractor(query)
        signal: RouteSignal = probe(entities, self.graph, self.thresholds)

        if signal.entities_found == 0:
            # Nothing to traverse from. The graph can only add noise, so the
            # question is purely one of lexical vs semantic matching.
            return RouteDecision(
                arm_id=self.routes.hybrid,
                reason="no query entity is present in the graph",
                signal=signal.as_dict(),
            )
        if signal.dense:
            return RouteDecision(
                arm_id=self.routes.graph,
                reason=(
                    f"{signal.in_lcc_count} connected entities at mean degree percentile "
                    f"{signal.mean_degree_percentile:.2f}"
                ),
                signal=signal.as_dict(),
            )
        return RouteDecision(
            arm_id=self.routes.hybrid,
            reason="entities present but sparsely connected - no chain to walk",
            signal=signal.as_dict(),
        )


# Surface cues a phrasing router keys on. This is the honest version of the
# baseline: these are the patterns the literature reports as strong predictors,
# not a strawman.
def _matches(cue: str, text: str) -> bool:
    """Whole-word (or whole-phrase) cue match."""
    return re.search(rf"\b{re.escape(cue.strip())}\b", text) is not None


_MULTIHOP_CUES = (
    "both",
    "also",
    "compare",
    "compared",
    "versus",
    " vs ",
    "difference between",
    "who also",
    "which of",
    "same",
    "earlier",
    "previously",
    "before joining",
    "in addition to",
    "as well as",
    "relationship between",
    "connected",
)
_SUMMARY_CUES = ("summarise", "summarize", "overview", "across all", "in general", "themes")
_LEXICAL_CUES = ("error code", "exact", "verbatim", "quote", "spelled", "number of")


class QueryRouter:
    """A7. Routes on query phrasing alone - the baseline A6 must beat.

    Deliberately never touches the graph or the index. If this wins, the honest
    conclusion is that phrasing was sufficient on this corpus, and that is a
    publishable result provided the comparison was built to be able to show it.
    """

    def __init__(self, routes: list[str]) -> None:
        self.routes = _Routes.from_list(routes)

    def route(self, query: str) -> RouteDecision:
        lowered = query.casefold()
        # Word boundaries, not substrings. Bare `cue in lowered` matched
        # "connected" inside "disconnected", "both" inside "bothered" and
        # "same" inside "Samesake" - uncontrolled false positives in the
        # baseline make the A6-A7 difference uninterpretable in either
        # direction, and A7 is meant to be a serious baseline.
        multihop = [cue for cue in _MULTIHOP_CUES if _matches(cue, lowered)]
        summary = [cue for cue in _SUMMARY_CUES if _matches(cue, lowered)]
        lexical = [cue for cue in _LEXICAL_CUES if _matches(cue, lowered)]
        signal: dict[str, object] = {
            "multihop_cues": multihop,
            "summary_cues": summary,
            "lexical_cues": lexical,
            "query_tokens": len(query.split()),
        }

        if multihop or summary:
            return RouteDecision(
                arm_id=self.routes.graph,
                reason=f"phrasing cues suggest multi-hop or summary: {(multihop + summary)[:3]}",
                signal=signal,
            )
        if lexical:
            return RouteDecision(
                arm_id=self.routes.lexical,
                reason=f"phrasing cues suggest an exact-token lookup: {lexical[:3]}",
                signal=signal,
            )
        return RouteDecision(
            arm_id=self.routes.hybrid,
            reason="no strong phrasing cue",
            signal=signal,
        )
