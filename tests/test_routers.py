from __future__ import annotations

import pytest

from adaptive_retrieval.graph.signals import SignalThresholds
from adaptive_retrieval.graph.store import InMemoryGraph, Triple
from adaptive_retrieval.router.routers import (
    CorpusRouter,
    QueryRouter,
    RegexEntityExtractor,
)

ROUTES = ["A1", "A4", "A5"]  # lexical, hybrid, graph


def _dense_graph() -> InMemoryGraph:
    graph = InMemoryGraph()
    names = ["Acme", "Northwind", "Globex", "Initech", "Umbrella", "Soylent"]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            graph.add(Triple(a, "rel", b, f"c{i}"))
    graph.add(Triple("Fringe Ltd", "rel", "Edge Co", "c99"))
    graph.finalise()
    return graph


# --------------------------------------------------------------------------
# entity extraction
# --------------------------------------------------------------------------


def test_extracts_proper_nouns() -> None:
    found = RegexEntityExtractor()("Did Acme acquire Northwind last year?")
    assert "Acme" in found
    assert "Northwind" in found


def test_drops_sentence_initial_question_words() -> None:
    """"Which" is capitalised by grammar, not by being a name."""
    found = RegexEntityExtractor()("Which outlets covered Acme?")
    assert not any(f.casefold() == "which" for f in found)
    assert "Acme" in found


def test_multiword_entities_stay_together() -> None:
    assert "Bank of Acme" in RegexEntityExtractor()("Did Bank of Acme report earnings?")


def test_repeated_entities_are_deduplicated() -> None:
    """Repeats would inflate found_fraction."""
    found = RegexEntityExtractor()("Acme and Acme again, plus Acme.")
    assert [f.casefold() for f in found].count("acme") == 1


def test_query_with_no_proper_nouns() -> None:
    assert RegexEntityExtractor()("what is the refund window?") == []


# --------------------------------------------------------------------------
# A6 - the corpus router
# --------------------------------------------------------------------------


def test_absent_entities_route_to_hybrid_not_graph() -> None:
    """Nothing to traverse from; the graph could only add noise."""
    router = CorpusRouter(ROUTES, _dense_graph())
    decision = router.route("Did Nonexistent Ltd acquire Missing Corp?")
    assert decision.arm_id == "A4"
    assert "not present" in decision.reason or "no query entity" in decision.reason


def test_densely_connected_entities_route_to_graph() -> None:
    router = CorpusRouter(
        ROUTES, _dense_graph(), thresholds=SignalThresholds(dense_percentile=0.3)
    )
    decision = router.route("How are Acme, Northwind and Globex related?")
    assert decision.arm_id == "A5"


def test_sparsely_connected_entities_route_to_hybrid() -> None:
    """Present but with no chain to walk."""
    router = CorpusRouter(ROUTES, _dense_graph())
    decision = router.route("What did Fringe Ltd announce?")
    assert decision.arm_id == "A4"


def test_decision_records_the_signal_for_auditing() -> None:
    router = CorpusRouter(ROUTES, _dense_graph())
    decision = router.route("Did Acme acquire Northwind?")
    assert set(decision.signal) >= {"entities_found", "found_fraction", "dense"}
    assert decision.reason


def test_corpus_router_never_calls_a_model() -> None:
    """The extractor protocol is called exactly once, and it is not an LLM."""
    calls: list[str] = []

    def extractor(query: str) -> list[str]:
        calls.append(query)
        return ["Acme"]

    CorpusRouter(ROUTES, _dense_graph(), extractor=extractor).route("Anything about Acme?")
    assert calls == ["Anything about Acme?"]


def test_router_requires_exactly_three_routes() -> None:
    with pytest.raises(ValueError, match="exactly 3 routes"):
        CorpusRouter(["A1", "A4"], _dense_graph())


# --------------------------------------------------------------------------
# A7 - the phrasing baseline
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Which outlets covered both the layoffs and the merger?",
        "Compare the two announcements.",
        "Who also worked there previously?",
    ],
)
def test_multihop_phrasing_routes_to_graph(query: str) -> None:
    assert QueryRouter(ROUTES).route(query).arm_id == "A5"


def test_summary_phrasing_routes_to_graph() -> None:
    assert QueryRouter(ROUTES).route("Summarise the themes across all coverage.").arm_id == "A5"


def test_exact_lookup_phrasing_routes_to_lexical() -> None:
    assert QueryRouter(ROUTES).route("What is the exact error code?").arm_id == "A1"


def test_unremarkable_phrasing_routes_to_hybrid() -> None:
    assert QueryRouter(ROUTES).route("What did the company announce?").arm_id == "A4"


def test_query_router_ignores_the_corpus_entirely() -> None:
    """A7 must not touch the graph - that is the whole point of the contrast."""
    router = QueryRouter(ROUTES)
    assert not hasattr(router, "graph")


# --------------------------------------------------------------------------
# the contrast itself
# --------------------------------------------------------------------------


def test_the_two_routers_can_disagree() -> None:
    """If they always agreed, A6 vs A7 would measure nothing.

    Phrasing says multi-hop; the corpus says these entities are not in the
    graph at all, so there is nothing to traverse.
    """
    query = "Which outlets covered both Nonexistent Ltd and Missing Corp?"
    corpus = CorpusRouter(ROUTES, _dense_graph()).route(query)
    phrasing = QueryRouter(ROUTES).route(query)
    assert phrasing.arm_id == "A5"
    assert corpus.arm_id == "A4"
    assert corpus.arm_id != phrasing.arm_id


def test_the_two_routers_can_agree() -> None:
    query = "How are Acme, Northwind and Globex both connected?"
    corpus = CorpusRouter(
        ROUTES, _dense_graph(), thresholds=SignalThresholds(dense_percentile=0.3)
    ).route(query)
    assert corpus.arm_id == QueryRouter(ROUTES).route(query).arm_id == "A5"
