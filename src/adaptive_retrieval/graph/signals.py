"""The routing signal: what the corpus says about this query's entities.

This is the contribution, and it is deliberately small. No LLM, no embedding,
no retrieval - one structural lookup against the graph, targeted under 30ms.
That budget is not a nice-to-have: an LLM here would add 300-800ms to *every*
query to save ~600ms on *some*, which inverts the whole argument for routing.

Prior work computes corpus-structural features **once per corpus**, which
collapses routing into a fixed per-dataset mapping. This computes them for
*this query's own neighbourhood*, at query time. That granularity difference is
the claim, and A7 - the same router on query phrasing alone - is the baseline
it has to beat.

Supernodes are the known failure mode. In any real corpus "United States" or
"Google" accumulates enormous degree, so a raw-degree threshold eventually
calls every query well-connected and the router degenerates into always-graph.
Two mitigations: entities above ``hub_percentile`` are excluded from the
density calculation, and density uses the *percentile* of degree within this
corpus rather than a raw count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaptive_retrieval.graph.store import EntityStats, GraphStore

__all__ = ["RouteSignal", "SignalThresholds", "probe"]


@dataclass(frozen=True, slots=True)
class SignalThresholds:
    """Where the routing boundaries sit. Published with the results."""

    #: Below this fraction of the query's entities found in the graph, the
    #: graph can only add noise.
    min_entities_found: float = 0.5
    #: Mean degree percentile above which the neighbourhood counts as dense.
    dense_percentile: float = 0.6
    #: Entities at or above this percentile are hubs and are excluded from the
    #: density calculation - they say nothing about *this* query.
    hub_percentile: float = 0.99
    #: At least this many non-hub entities must be present for a traversal to
    #: have anywhere to go. One entity is a lookup, not a chain.
    min_connected_entities: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("min_entities_found", self.min_entities_found),
            ("dense_percentile", self.dense_percentile),
            ("hub_percentile", self.hub_percentile),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.min_connected_entities < 1:
            raise ValueError(
                f"min_connected_entities must be >= 1, got {self.min_connected_entities}"
            )


@dataclass(frozen=True, slots=True)
class RouteSignal:
    """The structural facts a routing decision is allowed to use.

    Recorded verbatim on the result row, so every A6 decision is auditable
    after the fact rather than being a black box.
    """

    entities_total: int
    entities_found: int
    found_fraction: float
    #: ``None`` when every found entity was excluded as a hub - there is
    #: no informative neighbourhood to describe, and 0.0 would be a lie
    #: about entities that are in fact the most connected in the corpus.
    mean_degree_percentile: float | None
    in_lcc_count: int
    hub_entities: tuple[str, ...] = field(default_factory=tuple)
    dense: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities_total": self.entities_total,
            "entities_found": self.entities_found,
            "found_fraction": round(self.found_fraction, 4),
            "mean_degree_percentile": (
                round(self.mean_degree_percentile, 4)
                if self.mean_degree_percentile is not None
                else None
            ),
            "in_lcc_count": self.in_lcc_count,
            "hub_entities": list(self.hub_entities),
            "dense": self.dense,
        }


def probe(
    entities: list[str],
    graph: GraphStore,
    thresholds: SignalThresholds | None = None,
) -> RouteSignal:
    """Summarise what the graph knows about this query's entities."""
    limits = thresholds or SignalThresholds()

    if not entities:
        # No entities means nothing to look up. Not an error - plenty of
        # questions name nothing the extractor recognises - but the graph
        # cannot help, and the signal says so.
        return RouteSignal(
            entities_total=0,
            entities_found=0,
            found_fraction=0.0,
            mean_degree_percentile=None,
            in_lcc_count=0,
        )

    stats: dict[str, EntityStats] = graph.lookup(entities)
    found = [s for s in stats.values() if s.present]
    hubs = tuple(sorted(s.name for s in found if s.degree_percentile >= limits.hub_percentile))
    # Hubs are excluded from the density calculation: "United States" is
    # densely connected in every corpus and therefore says nothing about this
    # query. Without this the router degenerates to always-graph at scale.
    informative = [s for s in found if s.degree_percentile < limits.hub_percentile]

    mean_percentile = (
        sum(s.degree_percentile for s in informative) / len(informative) if informative else None
    )
    in_lcc_count = sum(1 for s in informative if s.in_lcc)
    found_fraction = len(found) / len(entities)

    dense = (
        found_fraction >= limits.min_entities_found
        and len(informative) >= limits.min_connected_entities
        and in_lcc_count >= limits.min_connected_entities
        and mean_percentile is not None
        and mean_percentile >= limits.dense_percentile
    )

    return RouteSignal(
        entities_total=len(entities),
        entities_found=len(found),
        found_fraction=found_fraction,
        mean_degree_percentile=mean_percentile,
        in_lcc_count=in_lcc_count,
        hub_entities=hubs,
        dense=dense,
    )
