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
from typing import Any

from adaptive_retrieval.analysis import compare_arms, load_rows, oracle_gap, summarise
from adaptive_retrieval.config import BenchmarkConfig, ConfigError, RouterArm, load_config
from adaptive_retrieval.es_client import EsClient, StackNotReadyError
from adaptive_retrieval.generate import Generator
from adaptive_retrieval.golden import (
    GoldenCase,
    GoldenSetError,
    check_provenance,
    class_distribution,
    load_golden_set,
)
from adaptive_retrieval.harness.executor import ArmExecutor, entailment_sufficiency
from adaptive_retrieval.harness.row import RunWriter
from adaptive_retrieval.harness.runner import Runner
from adaptive_retrieval.harness.wiring import NullArm, OracleArm, load_graph, smoke_verdict
from adaptive_retrieval.ingest.corpus import CorpusError, chunk_corpus, load_corpus
from adaptive_retrieval.judge import (
    KeywordEntailment,
    LlmEntailmentJudge,
    LlmQualityJudge,
    SecondOpinionEntailment,
)
from adaptive_retrieval.metrics.cost import is_priceable
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

    # Pricing coverage BEFORE money is spent, rather than one unpriceable
    # row at a time after.
    if not is_priceable(config.generator.model):
        print(
            f"pricing      : NO published price for {config.generator.model!r}. "
            f"Add it to MODEL_PRICING or the cost column will be empty.",
            file=sys.stderr,
        )
        return 1
    print(f"pricing      : {config.generator.model} priceable")

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

    if args.resume:
        already = es.indexed_chunk_ids()
        before = len(chunks)
        chunks = [c for c in chunks if c.chunk_id not in already]
        print(f"resume: {before - len(chunks)} chunks already indexed, {len(chunks)} to go")

    indexed, failures = es.index_chunks(chunks)
    print(f"indexed {indexed} chunks; index now holds {es.count()}")
    if failures:
        print(f"WARNING: {len(failures)} documents failed. First: {failures[0]}", file=sys.stderr)
        print("Re-run with --resume to retry only the missing chunks.", file=sys.stderr)
        return 1
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


def _anthropic_client() -> Any:
    """Construct the Claude client, deferring the import so the rest of the CLI
    works with no credentials and no SDK call."""
    import anthropic

    return anthropic.Anthropic()


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cases = load_golden_set(args.golden or config.golden_set)
    check_provenance(cases, {config.generator.model})

    if not is_priceable(config.generator.model):
        print(
            f"error: no published price for {config.generator.model!r}; the cost column "
            f"would be empty. Add it to MODEL_PRICING before spending.",
            file=sys.stderr,
        )
        return 1

    run_id = args.run_id or _run_id()
    run_dir = Path(args.out) / run_id

    trials = len(cases) * len(config.arms) * config.run.reps
    print(f"run {run_id}: {len(cases)} cases x {len(config.arms)} arms x {config.run.reps} reps")
    print(f"           = {trials} trials, each one paid model call. Writing to {run_dir}")
    floor = noise_floor(len(cases), config.run.reps)
    print(f"           noise floor +/-{floor:.3f} - an effect smaller than this is not real")
    if not args.yes:
        print("\nRe-run with --yes to start. Nothing has been spent.")
        return 0

    es = _connect(config)
    es.assert_ready(
        require_rerank=any(
            getattr(config.resolved_arm(a.id), "rerank", None) is not None for a in config.arms
        )
    )

    client = _anthropic_client()
    generator = Generator(
        client,
        model=config.generator.model,
        effort=config.generator.effort,
        max_tokens=config.generator.max_tokens,
    )

    # NLI would be the primary judge; without one wired, the keyword judge
    # keeps the pipeline honest about what it is rather than silently scoring
    # citations with an LLM.
    primary = KeywordEntailment()
    second = (
        LlmEntailmentJudge(client, model=config.judges.second_opinion)
        if config.judges.second_opinion
        else None
    )
    entails = SecondOpinionEntailment(primary=primary, second=second)
    quality = (
        LlmQualityJudge(client, model=config.judges.second_opinion)
        if config.judges.second_opinion
        else None
    )

    executor = ArmExecutor(
        config=config,
        es=es,
        generator=generator,
        graph=load_graph(args.graph),
        entails=entails,
        quality=quality,
        second_opinion=entails,
        sufficient=entailment_sufficiency(entails),
    )

    with RunWriter(run_dir) as writer:
        runner = Runner(config, writer, executor, run_id=run_id)
        try:
            rows, errors = runner.run(cases, resume=not args.no_resume)
        finally:
            runner.close()

    print(f"\n{rows} rows scored, {errors} errors. Report with:\n  ar report {run_dir}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Oracle and null baselines. No cluster, no model, no spend."""
    config = load_config(args.config)
    cases = load_golden_set(args.golden or config.golden_set)

    run_id = f"smoke-{_run_id()}"
    run_dir = Path(args.out) / run_id
    entails = KeywordEntailment()

    class _Pair:
        """Two arms only: the perfect answer and the empty one."""

        def __init__(self) -> None:
            self.oracle = OracleArm(config, entails=entails)
            self.null = NullArm(config)

        def __call__(self, arm_id: str, case: GoldenCase) -> Any:
            return self.oracle(arm_id, case) if arm_id == "ORACLE" else self.null(arm_id, case)

    smoke_config = config.model_copy(update={"run": config.run.model_copy(update={"reps": 1})})

    with RunWriter(run_dir) as writer:
        runner = Runner(smoke_config, writer, _Pair(), run_id=run_id, save_trajectories=False)
        runner.arm_ids = lambda: ["ORACLE", "NULL"]  # type: ignore[method-assign]
        try:
            runner.run(cases, resume=False)
        finally:
            runner.close()

    summaries = summarise(load_rows(run_dir / "results.jsonl"))
    for summary in summaries:
        print(f"  {summary.arm:<8} nDCG={summary.ndcg}  abstention={summary.abstention_accuracy}")

    passed, complaints = smoke_verdict(summaries)
    if not passed:
        for complaint in complaints:
            print(f"FAIL: {complaint}", file=sys.stderr)
        return 1
    print("\nsmoke OK: the grader scores a perfect answer and rejects an empty one.")
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
    ingest.add_argument("--resume", action="store_true", help="skip chunks already in the index")
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

    run = sub.add_parser("run", help="replay the golden set across every arm")
    run.add_argument("--golden")
    run.add_argument("--graph", help="path to a saved graph, for A5/A6")
    run.add_argument("--out", default="runs")
    run.add_argument("--run-id")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument(
        "--yes", action="store_true", help="actually spend money; without it, only estimates"
    )
    run.set_defaults(func=_cmd_run)

    smoke = sub.add_parser(
        "smoke", help="oracle and null baselines - no cluster, no model, no spend"
    )
    smoke.add_argument("--golden")
    smoke.add_argument("--out", default="runs")
    smoke.set_defaults(func=_cmd_smoke)

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
