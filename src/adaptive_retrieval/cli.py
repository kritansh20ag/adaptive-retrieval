"""Command line entry points.

    ar check       # is the stack able to serve every arm?
    ar ingest      # chunk the corpus and index it
    ar build-graph # extract entities and build the graph
    ar run         # replay the golden set across every arm
    ar report      # summarise a run, with intervals and the oracle gap

Every command that spends money says what it will cost before it starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from adaptive_retrieval.analysis import compare_arms, load_rows, oracle_gap, summarise
from adaptive_retrieval.config import BenchmarkConfig, ConfigError, RouterArm, load_config
from adaptive_retrieval.es_client import EsClient, StackNotReadyError
from adaptive_retrieval.golden import (
    GoldenSetError,
    check_provenance,
    class_distribution,
    load_golden_set,
)
from adaptive_retrieval.ingest.corpus import CorpusError, chunk_corpus, load_corpus
from adaptive_retrieval.stats import noise_floor

__all__ = ["main"]


def _connect(config: BenchmarkConfig) -> EsClient:
    return EsClient.connect(
        config.defaults.index,
        hosts=os.environ.get("ES_HOSTS", "http://localhost:9200"),
        api_key=os.environ.get("ES_API_KEY"),
    )


def _cmd_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    es = _connect(config)

    active, description = es.licence_is_active()
    print(f"licence      : {description} {'OK' if active else 'INACTIVE - ML will not run'}")
    try:
        es.assert_ready()
        print("inference    : all endpoints present")
    except StackNotReadyError as exc:
        print(f"inference    : NOT READY\n{exc}", file=sys.stderr)
        return 1

    print(f"index        : {config.defaults.index} ({es.count()} chunks)")

    cases = load_golden_set(args.golden or config.golden_set)
    check_provenance(cases, {config.generator.model})
    distribution = class_distribution(cases)
    print(f"golden set   : {len(cases)} cases {distribution}")

    floor = noise_floor(len(cases), config.run.reps)
    print(
        f"noise floor  : +/-{floor:.3f} at {len(cases)} cases x {config.run.reps} reps. "
        f"An effect smaller than this is not distinguishable from run-to-run spread."
    )
    smallest_class = min(distribution.values())
    if smallest_class:
        per_class = noise_floor(smallest_class, config.run.reps)
        print(
            f"             : +/-{per_class:.3f} for the smallest class ({smallest_class} cases) "
            f"- per-class results are descriptive, not findings."
        )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    es = _connect(config)
    es.ensure_inference_endpoints(num_allocations=args.allocations)
    es.ensure_index(recreate=args.recreate)

    documents = load_corpus(args.corpus)
    chunks = list(chunk_corpus(documents, config.chunking.to_chunking_config()))
    print(f"{len(documents)} documents -> {len(chunks)} chunks")

    if args.sample:
        # Measure ELSER throughput on a sample before committing to the full
        # corpus. At ~26 docs/sec per allocation a large corpus is an
        # hours-long job, and finding that out at the end is expensive.
        chunks = chunks[: args.sample]
        print(f"sample mode: indexing {len(chunks)} chunks")

    indexed = es.index_chunks(chunks)
    print(f"indexed {indexed} chunks; index now holds {es.count()}")
    return 0


def _cmd_build_graph(args: argparse.Namespace) -> int:
    from adaptive_retrieval.graph.extract import batch_requests

    config = load_config(args.config)
    documents = load_corpus(args.corpus)
    chunks = list(chunk_corpus(documents, config.chunking.to_chunking_config()))
    requests = batch_requests(chunks, model=args.model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in requests) + "\n", encoding="utf-8")
    print(
        f"wrote {len(requests)} extraction requests to {out}\n"
        f"Submit via the Batches API (50% cost). Results return in ARBITRARY order - "
        f"join them by custom_id, never by position."
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows = load_rows(Path(args.run) / "results.jsonl")
    errors_path = Path(args.run) / "errors.jsonl"
    n_errors = len(errors_path.read_text().splitlines()) if errors_path.exists() else 0

    print(f"{len(rows)} scored rows, {n_errors} errors (attempts run vs attempts scored)\n")

    header = (
        f"{'arm':<6}{'nDCG':>8}{'hit@k':>8}{'cite_r':>8}{'abst':>8}{'$/query':>10}{'p95 ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for summary in summarise(rows):

        def fmt(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "  --  "

        print(
            f"{summary.arm:<6}{fmt(summary.ndcg):>8}{fmt(summary.hit_rate):>8}"
            f"{fmt(summary.citation_recall):>8}{fmt(summary.abstention_accuracy):>8}"
            f"{summary.mean_cost_usd:>10.4f}{summary.p95_latency_ms:>9.0f}"
        )

    routers = [arm.id for arm in config.arms if isinstance(arm, RouterArm)]
    if len(routers) >= 2:
        print(f"\nThe experiment: {routers[0]} vs {routers[1]}")
        for (a, b), result, adjusted in compare_arms(rows, [(routers[0], routers[1])]):
            print(
                f"  {a} - {b} = {result.mean_difference:+.4f} nDCG, "
                f"95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}], "
                f"p={result.p_value:.4f} (Bonferroni {adjusted:.4f}), n={result.n_pairs}"
            )

    for router in routers:
        arm = config.arm(router)
        assert isinstance(arm, RouterArm)
        gap = oracle_gap(rows, router, list(arm.routes))
        if gap is None:
            continue
        print(
            f"\n{router}: {gap.router_score:.3f} | best fixed ({gap.best_fixed_arm}) "
            f"{gap.best_fixed_score:.3f} | oracle {gap.oracle_score:.3f} "
            f"| gap {gap.gap:.3f} over {gap.n_questions} questions"
        )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate the config and golden set without touching a cluster."""
    config = load_config(args.config)
    print(f"config OK: {len(config.arms)} arms -> {[a.id for a in config.arms]}")
    cases = load_golden_set(args.golden or config.golden_set)
    check_provenance(cases, {config.generator.model})
    print(f"golden set OK: {len(cases)} cases {class_distribution(cases)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ar", description=__doc__)
    parser.add_argument("--config", default="config/benchmark.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check config and golden set, no cluster needed")
    validate.add_argument("--golden")
    validate.set_defaults(func=_cmd_validate)

    check = sub.add_parser("check", help="is the stack able to serve every arm?")
    check.add_argument("--golden")
    check.set_defaults(func=_cmd_check)

    ingest = sub.add_parser("ingest", help="chunk the corpus and index it")
    ingest.add_argument("corpus")
    ingest.add_argument("--recreate", action="store_true")
    ingest.add_argument("--allocations", type=int, default=1)
    ingest.add_argument(
        "--sample", type=int, help="index only N chunks, to measure ELSER throughput first"
    )
    ingest.set_defaults(func=_cmd_ingest)

    graph = sub.add_parser("build-graph", help="write Batches requests for entity extraction")
    graph.add_argument("corpus")
    graph.add_argument("--out", default="runs/extraction-requests.jsonl")
    graph.add_argument("--model", default="claude-opus-5")
    graph.set_defaults(func=_cmd_build_graph)

    report = sub.add_parser("report", help="summarise a run")
    report.add_argument("run", help="a run directory containing results.jsonl")
    report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, GoldenSetError, CorpusError, StackNotReadyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
