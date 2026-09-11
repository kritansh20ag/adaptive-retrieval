from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_retrieval.graph.signals import SignalThresholds, probe
from adaptive_retrieval.graph.store import InMemoryGraph, Triple, normalise_entity


def _chain_graph() -> InMemoryGraph:
    """Acme -> Mehta -> Northwind -> Vendor, plus an isolated island."""
    graph = InMemoryGraph()
    graph.add_all(
        [
            Triple("Acme", "signed_by", "R. Mehta", "c12"),
            Triple("R. Mehta", "worked_at", "Northwind", "c288"),
            Triple("Northwind", "vendor_of", "Globex", "c903"),
            Triple("Globex", "supplies", "Initech", "c1104"),
            # A separate island: connected, but not to the chain above.
            Triple("Lonely Corp", "owns", "Lonely Sub", "c500"),
        ]
    )
    return graph


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_normalisation_is_case_and_whitespace_insensitive() -> None:
    assert normalise_entity("  Acme   Corp  ") == normalise_entity("acme corp")


def test_normalisation_strips_trailing_punctuation() -> None:
    assert normalise_entity("Acme.") == normalise_entity("Acme")


def test_normalisation_does_not_attempt_coreference() -> None:
    """Documented limitation: "R. Mehta" and "Rohit Mehta" stay distinct."""
    assert normalise_entity("R. Mehta") != normalise_entity("Rohit Mehta")


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------


def test_relations_are_undirected_for_connectivity() -> None:
    """The router asks whether entities are connected; direction is irrelevant."""
    graph = _chain_graph()
    assert graph.lookup(["Acme"])["Acme"].degree == 1
    assert graph.lookup(["R. Mehta"])["R. Mehta"].degree == 2


def test_absent_entity_is_reported_not_dropped() -> None:
    """"Not in the graph" is the most useful thing the router learns."""
    stats = graph_stats = _chain_graph().lookup(["Nonexistent Ltd"])
    assert "Nonexistent Ltd" in graph_stats
    assert stats["Nonexistent Ltd"].present is False
    assert stats["Nonexistent Ltd"].degree == 0


def test_largest_connected_component_excludes_the_island() -> None:
    graph = _chain_graph()
    assert graph.lookup(["Acme"])["Acme"].in_lcc is True
    assert graph.lookup(["Lonely Corp"])["Lonely Corp"].in_lcc is False


def test_degree_percentile_ranks_within_the_corpus() -> None:
    graph = _chain_graph()
    hub = graph.lookup(["R. Mehta"])["R. Mehta"]
    leaf = graph.lookup(["Acme"])["Acme"]
    assert hub.degree_percentile > leaf.degree_percentile


def test_expansion_returns_chunk_ids_never_text() -> None:
    """The graph stores pointers; the text lives only in Elasticsearch."""
    chunk_ids = _chain_graph().expand(["Acme"], hops=3)
    assert set(chunk_ids) >= {"c12", "c288"}
    assert all(cid.startswith("c") for cid in chunk_ids)


def test_expansion_reaches_further_with_more_hops() -> None:
    graph = _chain_graph()
    near = graph.expand(["Acme"], hops=1)
    far = graph.expand(["Acme"], hops=3)
    assert set(near) < set(far)


def test_expansion_respects_the_limit() -> None:
    assert len(_chain_graph().expand(["Acme"], hops=5, limit=2)) == 2


def test_expansion_of_an_unknown_entity_is_empty() -> None:
    assert _chain_graph().expand(["Nonexistent Ltd"]) == []


@pytest.mark.parametrize(("hops", "limit"), [(0, 10), (1, 0)])
def test_expansion_rejects_invalid_bounds(hops: int, limit: int) -> None:
    with pytest.raises(ValueError):
        _chain_graph().expand(["Acme"], hops=hops, limit=limit)


def test_empty_entity_names_are_ignored() -> None:
    graph = InMemoryGraph()
    graph.add_all([Triple("", "rel", "Something", "c1")])
    assert graph.entity_count == 0


def test_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    _chain_graph().save(path)
    restored = InMemoryGraph.load(path)
    assert restored.lookup(["Acme"])["Acme"].in_lcc is True
    assert set(restored.expand(["Acme"], hops=3)) >= {"c12", "c288"}


# --------------------------------------------------------------------------
# the routing signal
# --------------------------------------------------------------------------


def test_absent_entities_are_not_dense() -> None:
    signal = probe(["Nonexistent Ltd", "Also Missing"], _chain_graph())
    assert signal.entities_found == 0
    assert signal.dense is False


def test_no_entities_is_not_an_error() -> None:
    """Plenty of questions name nothing the extractor recognises."""
    signal = probe([], _chain_graph())
    assert signal.entities_total == 0
    assert signal.dense is False


def test_single_entity_is_a_lookup_not_a_chain() -> None:
    """One entity has nowhere to traverse to, whatever its degree."""
    signal = probe(["R. Mehta"], _chain_graph())
    assert signal.dense is False


def test_connected_entities_read_as_dense() -> None:
    graph = InMemoryGraph()
    # A tightly connected cluster: every node has several neighbours.
    names = [f"E{i}" for i in range(6)]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            graph.add(Triple(a, "rel", b, f"c{i}"))
    graph.add(Triple("Fringe", "rel", "Edge", "c99"))
    graph.finalise()

    signal = probe(["E0", "E1", "E2"], graph, SignalThresholds(dense_percentile=0.3))
    assert signal.entities_found == 3
    assert signal.in_lcc_count == 3
    assert signal.dense is True


def test_hubs_are_excluded_from_the_density_calculation() -> None:
    """A supernode is densely connected in every corpus, so it says nothing
    about this query. Without this the router degenerates to always-graph."""
    graph = InMemoryGraph()
    for i in range(50):
        graph.add(Triple("United States", "mentions", f"Thing{i}", f"c{i}"))
    graph.add(Triple("Obscure A", "rel", "Obscure B", "c900"))
    graph.finalise()

    signal = probe(["United States"], graph, SignalThresholds(hub_percentile=0.9))
    assert "United States" in signal.hub_entities
    assert signal.dense is False


def test_found_fraction_is_reported() -> None:
    signal = probe(["Acme", "Nonexistent Ltd"], _chain_graph())
    assert signal.found_fraction == pytest.approx(0.5)


def test_signal_serialises_for_the_result_row() -> None:
    """Every A6 decision must be auditable after the fact."""
    payload = probe(["Acme", "R. Mehta"], _chain_graph()).as_dict()
    assert set(payload) >= {"entities_found", "found_fraction", "dense", "in_lcc_count"}


@pytest.mark.parametrize(
    "kwargs",
    [{"min_entities_found": 1.5}, {"dense_percentile": -0.1}, {"min_connected_entities": 0}],
)
def test_invalid_thresholds_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SignalThresholds(**kwargs)  # type: ignore[arg-type]
