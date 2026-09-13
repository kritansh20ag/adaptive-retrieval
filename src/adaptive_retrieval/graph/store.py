"""The entity graph.

Entities and relations only. Every node records the **chunk IDs** it was
extracted from, and nothing records the chunk text - that lives in
Elasticsearch and only in Elasticsearch. This is the difference from Amazon
Bedrock Knowledge Bases GraphRAG and from Neo4j's GraphRAG package, both of
which copy chunk text into the graph and then have two stores that can drift.

``InMemoryGraph`` is the default and is a complete implementation, not a stub:
the whole benchmark including A5 and A6 runs on it without provisioning
Neptune, which is worth a lot when Neptune is a fixed ~$0.48/hr whether or not
a query uses it. ``GraphStore`` is the protocol a Neptune adapter satisfies.

Entity normalisation is deliberately simple - casefold and whitespace collapse,
no coreference resolution. Coreference is the known failure mode of graph
extraction (one study cut node duplication 33% by adding it), but resolving it
properly is a research project. We accept imperfect recall and treat the graph
as an *additive* source of candidate chunks, so a missing edge costs a
candidate rather than an answer.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "EntityStats",
    "GraphStore",
    "InMemoryGraph",
    "Triple",
    "normalise_entity",
]

_WHITESPACE = re.compile(r"\s+")


def normalise_entity(name: str) -> str:
    """Canonical key for an entity mention.

    Casefold plus whitespace collapse plus trailing-punctuation strip. That is
    all - see the module docstring on why coreference is out of scope.
    """
    cleaned = _WHITESPACE.sub(" ", name).strip().strip(".,;:!?\"'")
    return cleaned.casefold()


@dataclass(frozen=True, slots=True)
class Triple:
    """One extracted fact, and the chunk it came from."""

    subject: str
    predicate: str
    object: str
    chunk_id: str


@dataclass(frozen=True, slots=True)
class EntityStats:
    """What the router is allowed to know about one entity.

    This is deliberately a small, cheap, structural summary. It contains no
    text and requires no model - the whole point is that the routing decision
    costs a graph lookup rather than an LLM call.
    """

    #: The name as the caller wrote it.
    name: str
    present: bool
    #: The name as first seen in the corpus, for audit trails - "R. Mehta"
    #: rather than the normalised key.
    label: str = ""
    degree: int = 0
    #: True when the entity sits in the graph's largest connected component.
    #: An entity in a tiny island has neighbours but nothing to traverse to.
    in_lcc: bool = False
    #: Degree relative to the corpus, in [0, 1]. A supernode like "United
    #: States" has a huge raw degree in every corpus; the percentile is what
    #: says whether this entity is unusually connected *here*.
    degree_percentile: float = 0.0


class GraphStore(Protocol):
    """What the router and graph expansion need from a graph backend."""

    def lookup(self, names: list[str]) -> dict[str, EntityStats]: ...

    def expand(self, names: list[str], *, hops: int = 2, limit: int = 50) -> list[str]: ...


@dataclass
class InMemoryGraph:
    """A complete in-process entity graph.

    Adjacency is undirected for connectivity purposes: the router asks "are
    these entities connected", and a relation is a connection whichever way it
    was written down.
    """

    _adjacency: dict[str, set[str]] = field(default_factory=dict)
    _chunks: dict[str, set[str]] = field(default_factory=dict)
    _labels: dict[str, str] = field(default_factory=dict)
    _lcc: frozenset[str] = field(default_factory=frozenset)
    _degree_ranks: dict[str, float] = field(default_factory=dict)
    _dirty: bool = False

    # -- construction ------------------------------------------------------

    def add(self, triple: Triple) -> None:
        subject = normalise_entity(triple.subject)
        obj = normalise_entity(triple.object)
        if not subject or not obj:
            return
        self._labels.setdefault(subject, triple.subject.strip())
        self._labels.setdefault(obj, triple.object.strip())
        self._adjacency.setdefault(subject, set()).add(obj)
        self._adjacency.setdefault(obj, set()).add(subject)
        self._chunks.setdefault(subject, set()).add(triple.chunk_id)
        self._chunks.setdefault(obj, set()).add(triple.chunk_id)
        self._dirty = True

    def add_all(self, triples: list[Triple]) -> None:
        for triple in triples:
            self.add(triple)
        self.finalise()

    def finalise(self) -> None:
        """Recompute the derived structure the router reads.

        Kept explicit rather than lazy so the cost is paid once at load time,
        not inside the query path where it would show up as routing latency.
        """
        self._lcc = self._largest_component()
        degrees = sorted(len(neighbours) for neighbours in self._adjacency.values())
        total = len(degrees)
        self._degree_ranks = {}
        for key, neighbours in self._adjacency.items():
            degree = len(neighbours)
            # Midrank percentile: (below + half the ties) / total. A strict
            # "fraction with lower degree" collapses under ties - in a clique
            # every member has the same degree, so none counts any other and
            # a densely connected cluster scores near zero.
            #
            # bisect rather than a scan: this runs once per entity, and the
            # naive form is O(n^2) over the whole corpus.
            below = bisect_left(degrees, degree)
            at_or_below = bisect_right(degrees, degree)
            midrank = (below + at_or_below) / 2.0
            self._degree_ranks[key] = (midrank / total) if total else 0.0
        self._dirty = False

    def _largest_component(self) -> frozenset[str]:
        seen: set[str] = set()
        largest: set[str] = set()
        for start in self._adjacency:
            if start in seen:
                continue
            component: set[str] = set()
            queue = deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbour in self._adjacency.get(node, ()):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            if len(component) > len(largest):
                largest = component
        return frozenset(largest)

    # -- reads -------------------------------------------------------------

    @property
    def entity_count(self) -> int:
        return len(self._adjacency)

    def lookup(self, names: list[str]) -> dict[str, EntityStats]:
        """Structural summary for each name. Absent entities are reported, not dropped.

        "This entity is not in the graph" is the single most useful thing the
        router learns, so a missing entity must come back as a present=False
        record rather than vanish from the result.
        """
        if self._dirty:
            self.finalise()
        stats: dict[str, EntityStats] = {}
        for name in names:
            key = normalise_entity(name)
            neighbours = self._adjacency.get(key)
            if neighbours is None:
                stats[name] = EntityStats(name=name, present=False)
                continue
            stats[name] = EntityStats(
                name=name,
                label=self._labels.get(key, name),
                present=True,
                degree=len(neighbours),
                in_lcc=key in self._lcc,
                degree_percentile=self._degree_ranks.get(key, 0.0),
            )
        return stats

    def expand(self, names: list[str], *, hops: int = 2, limit: int = 50) -> list[str]:
        """Chunk IDs reachable within ``hops`` of any named entity.

        Returns chunk IDs, never text: the caller fetches those from
        Elasticsearch. Results are ordered by hop distance so a truncated list
        keeps the closest evidence.
        """
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        seeds = [normalise_entity(n) for n in names]
        frontier = deque((key, 0) for key in seeds if key in self._adjacency)
        seen = {key for key, _ in frontier}
        # Chunks are collected per node, then drained round-robin. Emitting a
        # node's whole chunk list before moving on lets one frequent entity
        # consume the entire budget: for "Acme AND Northwind", where Acme
        # appears in 30 chunks and Northwind in 3, a limit of 10 returned
        # Acme-only evidence - on precisely the multi-hop query graph expansion
        # exists to serve. Which entity won depended on word order.
        per_node: list[list[str]] = []

        while frontier:
            key, depth = frontier.popleft()
            chunks = sorted(self._chunks.get(key, ()))
            if chunks:
                per_node.append(chunks)
            if depth >= hops:
                continue
            for neighbour in sorted(self._adjacency.get(key, ())):
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, depth + 1))

        ordered: list[str] = []
        emitted: set[str] = set()
        for round_index in range(max((len(c) for c in per_node), default=0)):
            for chunks in per_node:
                if round_index >= len(chunks):
                    continue
                chunk_id = chunks[round_index]
                if chunk_id in emitted:
                    continue
                emitted.add(chunk_id)
                ordered.append(chunk_id)
                if len(ordered) >= limit:
                    return ordered
        return ordered

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "adjacency": {k: sorted(v) for k, v in self._adjacency.items()},
            "chunks": {k: sorted(v) for k, v in self._chunks.items()},
            "labels": self._labels,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls(
            _adjacency={k: set(v) for k, v in payload["adjacency"].items()},
            _chunks={k: set(v) for k, v in payload["chunks"].items()},
            _labels=dict(payload["labels"]),
        )
        graph.finalise()
        return graph
