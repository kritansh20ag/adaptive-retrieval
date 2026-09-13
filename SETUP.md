# Setting up and running locally

Two paths. **Path A** needs nothing but Python and gets you as far as
validating the config, running the full test suite, and exercising every pure
component. **Path B** adds Docker and gets you a real index and real arms.

Nothing here needs AWS. Neptune is optional — the graph has a complete
in-memory implementation, so A5 and A6 run without it.

---

## Path A — no Docker, no cloud (2 minutes)

```bash
git clone https://github.com/kritansh20ag/adaptive-retrieval.git
cd adaptive-retrieval

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest          # 416 tests
ruff check src tests
mypy src
ar validate     # config + golden set, no cluster needed
ar smoke        # oracle and null baselines - no cluster, no model, no spend
```

`ar smoke` is the one to trust. It runs the whole grading pipeline end to end
against two synthetic arms: an **oracle** that hands the grader the gold chunks
and a perfect answer, and a **null** that returns nothing. The oracle must score
nDCG 1.0 and the null must score 0.0. If it fails, the fault is in the metrics,
the golden set or the judge - and no amount of retrieval work will fix it.

`ar validate` should print the eleven arms and the golden-set class distribution.
If it does, the config is internally consistent and every invariant the harness
depends on is being enforced.

For an exactly reproducible environment, install from the lockfile instead:

```bash
pip install -r requirements.lock && pip install -e . --no-deps
```

---

## Path B — the real stack

### 1. Give Docker enough memory

**This is the step people get wrong, and the failure is silent.**

ML inference (ELSER, the dense encoder, the reranker) runs in native processes
*outside* the JVM heap, and Elasticsearch caps their total memory. Elastic
documents a 4 GB ML node for ELSER alone, and **8 GB to run the Elastic Rerank
model alongside ELSER** — which arms A4 and A5 both require.

Under-provision it and you do not get an error. You get an ELSER leg that
returns nothing, an RRF fusion that still produces a plausible ranked list, and
a plausible nDCG that is wrong.

- **Docker Desktop:** Settings → Resources → Memory → **at least 14 GB**
- **Colima:** `colima start --cpu 4 --memory 14 --disk 60`

### 2. Start the stack

```bash
docker compose up -d          # or: docker-compose up -d
docker compose ps             # wait for elasticsearch to report (healthy)
```

Elasticsearch is on `http://localhost:9200`, Kibana on `http://localhost:5601`.
Both are bound to **loopback only** — security is disabled for local dev, so
publishing them on `0.0.0.0` would expose the corpus and every result to the
network.

First start takes a few minutes: the images are large and Elasticsearch has to
download the ML models.

### 3. Check the stack can serve every arm

```bash
ar check
```

This asserts the licence is active (a self-generated trial **expires after 30
days**, after which ML silently stops), that every inference endpoint exists,
and prints the noise floor for your golden set — the effect size below which a
difference between arms is indistinguishable from run-to-run spread.

If `ar check` fails on inference endpoints, it is almost always step 1.

### 4. Ingest a corpus

Get MultiHop-RAG (609 news articles, ODC-BY licensed):

```bash
# https://huggingface.co/datasets/yixuantt/MultiHopRAG
mkdir -p data && curl -L -o data/corpus.jsonl <corpus-url>
```

Then **measure throughput on a sample before committing to the whole thing** —
ELSER indexes at roughly 26 docs/sec per allocation, so a large corpus is an
hours-long job and you want to know that at the start:

```bash
ar ingest data/corpus.jsonl --sample 1000    # time this
ar ingest data/corpus.jsonl                  # then the full run
```

Ingest is resumable and idempotent: the Elasticsearch `_id` is the
content-addressed chunk ID, so re-running overwrites in place rather than
duplicating.

### 5. Build the graph (optional — needed for A5 and A6)

```bash
ar build-graph data/corpus.jsonl --out runs/extraction-requests.jsonl
```

This writes Batches API requests rather than calling the model directly:
extraction is ~75% of index cost and has no latency requirement, so the Batches
API's 50% discount is the right surface. **Results return in arbitrary order —
join them by `custom_id`, never by position.**

### 6. Run and report

```bash
ar run                 # prints the trial count, the cost shape and the noise
                       # floor, then STOPS. Nothing is spent.
ar run --yes           # actually runs
ar run --yes --graph runs/graph.json   # with A5/A6
ar report runs/<run-id>
```

`ar run` without `--yes` is an estimate only: it validates the config and the
golden set, refuses to start if the generator model has no published price, and
prints the noise floor so you can see whether the effect you are chasing is even
detectable at this sample size. It touches no cluster and no model.

Runs are resumable. An attempt that errored is *not* complete and is retried;
only scored rows are skipped. Per-case trajectories - every retrieved chunk, the
answer, the citations, the routing decision - are written to
`runs/<run-id>/trajectories/`, so a surprising score can be explained without
re-running it.

The report prints per-arm quality, cost per query, p95 latency, the A6-vs-A7
comparison with a bootstrap confidence interval and a Bonferroni-adjusted
p-value, and each router's gap against a perfect router.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ES_HOSTS` | `http://localhost:9200` | Elasticsearch endpoint |
| `ES_API_KEY` | unset | For a secured cluster |
| `ANTHROPIC_API_KEY` | unset | Or use `ant auth login`, which the SDK picks up automatically |

Never put credentials in `benchmark.yaml` — it is committed, and it is the file
published alongside the results.

---

## Troubleshooting

**`ar check` says an inference endpoint is missing.** Memory. See step 1.

**ELSER will not deploy on an Apple Silicon Mac.** Elastic's preconfigured
endpoints point at `linux-x86_64`-optimised model builds that do not run on
arm64. This repo does not use them — it creates its own endpoints against the
platform-agnostic model IDs. If you deployed the preconfigured ones by hand,
delete them.

**The cluster never reaches green.** It never will. A single-node cluster stays
yellow while any index has replicas, which is why the health check waits for
yellow.

**Numbers changed and nothing else did.** Check the licence. A trial expires
after 30 days, ML stops, and arms that depend on ELSER or the reranker quietly
degrade rather than failing.

**`ModuleNotFoundError: adaptive_retrieval` when running pytest.** You are
outside the venv, or the package is not installed. `pytest` is configured with
`pythonpath = ["src"]` so a fresh clone works, but the `ar` command needs
`pip install -e .`.
