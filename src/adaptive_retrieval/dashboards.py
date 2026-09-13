"""Generate Kibana saved objects for the scorecard.

Generated rather than hand-maintained, so the panels cannot drift from the row
schema. Run as ``python -m adaptive_retrieval.dashboards``.

Two filters are baked into every quality panel rather than left to whoever
opens the dashboard:

* ``status: ok`` - truncated rows are counted but never averaged into quality,
  because a response cut off at ``max_tokens`` is not a wrong answer.
* ``ndcg_at_k`` must exist - unanswerable questions have no retrieval score,
  and the exclusion should be visible rather than implicit.
"""

from __future__ import annotations

import json
import sys
from typing import Any

__all__ = ["RESULTS_INDEX", "build_saved_objects", "main"]

RESULTS_INDEX = "benchmark-results-v1"
_INDEX_PATTERN_ID = "adaptive-retrieval-results"

#: Applied to every quality panel. See the module docstring.
_QUALITY_FILTER = 'status : "ok" and ndcg_at_k : *'


def _index_pattern() -> dict[str, Any]:
    return {
        "id": _INDEX_PATTERN_ID,
        "type": "index-pattern",
        "attributes": {
            "title": RESULTS_INDEX,
            "timeFieldName": "",
        },
        "references": [],
    }


def _lens_panel(
    panel_id: str,
    title: str,
    description: str,
    *,
    metric: str,
    breakdown: str | None,
    query: str,
) -> dict[str, Any]:
    return {
        "id": panel_id,
        "type": "lens",
        "attributes": {
            "title": title,
            "description": description,
            "visualizationType": "lnsXY",
            "state": {
                "query": {"language": "kuery", "query": query},
                "filters": [],
                "datasourceStates": {
                    "formBased": {
                        "layers": {
                            "layer1": {
                                "columns": {
                                    "metric": {
                                        "label": metric,
                                        "operationType": "average",
                                        "sourceField": metric,
                                        "dataType": "number",
                                        "isBucketed": False,
                                    },
                                    "arm": {
                                        "label": "arm",
                                        "operationType": "terms",
                                        "sourceField": "arm",
                                        "dataType": "string",
                                        "isBucketed": True,
                                        "params": {"size": 20, "orderDirection": "asc"},
                                    },
                                    **(
                                        {
                                            "split": {
                                                "label": breakdown,
                                                "operationType": "terms",
                                                "sourceField": breakdown,
                                                "dataType": "string",
                                                "isBucketed": True,
                                                "params": {"size": 10, "orderDirection": "asc"},
                                            }
                                        }
                                        if breakdown
                                        else {}
                                    ),
                                },
                                "columnOrder": (
                                    ["arm", "split", "metric"] if breakdown else ["arm", "metric"]
                                ),
                                "indexPatternId": _INDEX_PATTERN_ID,
                            }
                        }
                    }
                },
            },
        },
        "references": [
            {
                "id": _INDEX_PATTERN_ID,
                "name": "indexpattern-datasource-layer-layer1",
                "type": "index-pattern",
            }
        ],
    }


def build_saved_objects() -> list[dict[str, Any]]:
    """The four panels, plus the index pattern they read."""
    return [
        _index_pattern(),
        _lens_panel(
            "ar-quality-by-class",
            "Quality by arm and query class",
            (
                "Where each strategy actually wins. An overall mean can hide a strategy "
                "that wins two classes and loses two - that crossover is the whole point."
            ),
            metric="ndcg_at_k",
            breakdown="query_class",
            query=_QUALITY_FILTER,
        ),
        _lens_panel(
            "ar-cost-per-query",
            "Cost per query by arm",
            (
                "Reported beside quality, never alone: a strategy that is two points "
                "better and six times more expensive should lose, and the scorecard is "
                "built so that it visibly does."
            ),
            metric="cost_usd",
            breakdown="query_class",
            query='status : "ok"',
        ),
        _lens_panel(
            "ar-latency-p95",
            "Generation latency by arm",
            (
                "Split by `retried`: a corrective retry roughly doubles the work, so "
                "mixing retried and non-retried rows makes p95 unreadable."
            ),
            metric="latency_ms.generate",
            breakdown="retried",
            query='status : "ok"',
        ),
        _lens_panel(
            "ar-router-audit",
            "Router decision audit",
            (
                "What A6 chose, per query class. Compare against the best-scoring arm "
                "for the same question to read the oracle gap - published routers sit "
                "well short of a perfect one, so a gap is expected."
            ),
            metric="ndcg_at_k",
            breakdown="route_taken",
            query=f"{_QUALITY_FILTER} and route_taken : *",
        ),
    ]


def main() -> int:
    for obj in build_saved_objects():
        sys.stdout.write(json.dumps(obj, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
