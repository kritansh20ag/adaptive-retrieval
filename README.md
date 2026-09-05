# Adaptive Retrieval

A corpus-aware adaptive retrieval engine and a reproducible benchmark harness, built on Elasticsearch and AWS.

**Forge the Future 2026 — Intelligent Search & AI Platforms**

---

## 1. The problem

Production RAG systems pick one retrieval strategy and apply it to every query. Published benchmarks show that is the wrong default:

- Plain vector retrieval wins on **single-hop factual** questions.
- Graph-augmented retrieval wins on **multi-hop relational** and **corpus-level** questions.
- Measured on the same corpus, Microsoft GraphRAG's global search consumes **331,375 tokens per query** against roughly **900** for plain vector RAG.

Running one strategy uniformly either burns cost on easy queries or silently fails on hard ones — and teams have no reproducible way to find out which is happening on their corpus.

## 2. What we are building

**An adaptive retrieval engine.** A router inspects each query and dispatches it to the cheapest strategy that will actually answer it — lexical, hybrid, hybrid+rerank, or graph-expanded — with one corrective retry when the returned evidence is too thin to ground an answer.

**An open benchmark harness.** The same golden query set is replayed across every strategy from a single YAML config, producing a retrieval-quality, citation-quality, latency and cost-per-query scorecard, indexed back into Elasticsearch and dashboarded in Kibana.

### The specific contribution

Not "adaptive routing" — that exists. Ours is narrower and defensible:

1. **Per-query, index-grounded routing signal.** Prior work (RAGRouter-Bench, 2026) computes corpus-structural features *once per corpus*, which collapses routing into a fixed per-dataset mapping. We link *this query's* entities into the live graph and measure their *local* neighbourhood — presence, degree, membership of the largest connected component.
2. **A routing catalogue that includes a lexical tier and a reranking tier.** Published routing benchmarks catalogue LLM-only / NaiveRAG / GraphRAG / HybridRAG / IterativeRAG. None includes BM25 or a cross-encoder stage — the two cheapest things that work.
3. **One per-query row joining relevance, citation quality, latency and cost, sliced by query class.** Quality frameworks (RAGAS, BEIR, ranx) emit relevance without cost. Observability tools (TruLens, Phoenix, LangSmith) emit cost without relevance judgements. Nobody ships the join.

### What we are explicitly not claiming

Graph retrieval, hybrid fusion, reranking and adaptive routing are all established. `HippoRAG 2` (ICML 2025) already stores phrase nodes pointing at passage nodes without duplicating text — our storage design follows it rather than inventing anything. We cite prior art up front and position against it.

---

## 3. Architecture

### Offline (once)

1. Documents land in **S3**.
2. Step Functions + Lambda chunk and embed them.
3. Chunk text is written to **Elasticsearch** — one ES document per chunk, indexed three ways (BM25, dense `semantic_text`, ELSER sparse).
4. **Claude on Bedrock** extracts entities and relations from each chunk; these become nodes and edges in **Neptune Analytics**. Each node stores the **chunk IDs** it was extracted from — never the text.

### Online (per query)

5. The router extracts query entities and probes Neptune: do these entities exist, how connected are they? **Cheap, non-LLM, target <30 ms.**
6. Route: absent/isolated → hybrid+rerank; present and densely connected → graph-expanded; trivially lexical → BM25.
7. Retrieval executes in Elasticsearch. Graph hits return chunk IDs, which are fetched from the same index.
8. Sufficiency check → at most **one** broader retry → answer with per-sentence citations.
9. One result row is indexed for the scorecard.

### The two load-bearing decisions

| Decision | Why | What breaks without it |
|---|---|---|
| Graph stores **chunk ID pointers, not text** | One source of truth; no drift on re-ingest | Two copies of the corpus diverging. (Bedrock KB GraphRAG and Neo4j both duplicate the text — this is our differentiator against the managed baseline.) |
| Chunk IDs are **content hashes**, never Elasticsearch `_id` | Pointers must survive a reindex | One reindex silently breaks every graph edge's pointer |

---

## 4. Verified API notes

Checked against live documentation on **2026-09-05**. Do not code from memory; re-verify before touching these.

### Elasticsearch retrievers

**`rrf`** — [docs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever)
- `rank_constant` default **60** (must be ≥ 1)
- `rank_window_size` default **10** — must be ≥ `size`. **Never leave at the default for a benchmark.**
- `retrievers` and `query` are mutually exclusive; one is required
- Per-retriever weights (ES 9.2+): `weight` is a *sibling of the retriever type key* — `{ "weight": 2.0, "standard": {...} }`
- Score: `rrf_score = Σ (weight_i × rrf_score_i)`

**`linear`** — [docs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever)
- Weight shape **differs from `rrf`**: `{ "retriever": {...}, "weight": 5, "normalizer": "minmax" }`
- `normalizer`: `none` | `minmax` | `l2_norm`. Top-level `normalizer` (9.2+) is the default for children; per-retriever overrides. Pre-9.2 default is `none`
- `minmax` is `score = (score - min) / (max - min)`

**`text_similarity_reranker`** — [docs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever)
- Required: `retriever`, `field`, `inference_text`
- `inference_id` default `.rerank-v1-elasticsearch`
- `rank_window_size` default **10** — this is rerank depth, far too shallow by default
- Optional `min_score`, `filter`, `chunk_rescorer`
- Valid rerank endpoints: Elastic Rerank, Cohere Rerank, Google Vertex AI, Eland-uploaded `text_similarity` models

**`semantic_text`** — [docs](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text)
- Auto-chunks at index time; configurable via `chunking_settings`
- **Always set `inference_id` explicitly** — omitting it means a version upgrade can give new indices a different embedding model than existing ones
- We chunk ourselves regardless, because the graph needs stable chunk IDs to point at

### Claude models

Per the bundled Claude API skill (cached 2026-06-24):

- **Default model: `claude-opus-5`.** 1M context, $5/$25 per MTok. Not date-suffixed.
- **On Bedrock**, use the Mantle client — Python `AnthropicBedrockMantle(aws_region=...)` — and model IDs take an `anthropic.` prefix: `anthropic.claude-opus-5`. `AnthropicBedrock` (without `Mantle`) is the legacy InvokeModel path.
- **Thinking:** `thinking: {"type": "adaptive"}`. `budget_tokens` is **rejected with a 400** on Opus 5. Depth is controlled by `output_config: {"effort": ...}`, not a token budget.
- **No assistant prefill** — returns 400. Use structured outputs (`output_config.format`) instead.
- **`stop_reason: "refusal"`** is an HTTP 200 outcome. Always check `stop_reason` before reading `content`.
- **Batches** run at 50% cost and are the right surface for entity extraction over a whole corpus. Results return in **any order** — key by `custom_id`, never position.
- **Prompt caching** is prefix-match: render order is `tools` → `system` → `messages`. Put the stable extraction instructions first and the varying chunk last. Verify with `usage.cache_read_input_tokens`; if it is zero across repeated calls, something is silently invalidating the prefix.
- Bulk extraction and the LLM judge may run on a cheaper model (`claude-sonnet-5` / `claude-haiku-4-5`) — **but that is a cost decision for the repo owner to make explicitly**, not a default. See `WHAT-I-NEED-FROM-YOU.md`.

---

## 5. The arms

Each arm adds exactly one thing to the one above it, so any difference is attributable.

| Arm | Strategy | Isolates |
|---|---|---|
| `A0` | **Closed book — no retrieval at all** | How much is answerable from memory (see §6) |
| `A1` | BM25 lexical only | Classical baseline |
| `A2` | Dense vectors only | Naive RAG baseline |
| `A3` | BM25 + dense + ELSER, fused with RRF | Signal fusion |
| `A3b` | Same three signals, score-normalised `linear` fusion | Is RRF the weak default the literature says it is? |
| `A4` | Best fusion + cross-encoder rerank | Second-stage ranking |
| `A5` | A4 + graph expansion over entity relations | The knowledge graph |
| **`A6`** | **Router on corpus signal, one corrective retry** | **Routing** |
| **`A7`** | **Same router, query-phrasing classifier only** | **Does the corpus signal beat phrasing?** |
| `A8` | A4 re-indexed with contextual retrieval | Indexing vs routing (stretch) |
| `A9` | **Agentic**: Claude chooses retrievers via tool use, unbounded | The ceiling — what unlimited budget buys |

### Why the agentic design is an arm, not the engine

Giving Claude `search_lexical` / `search_hybrid` / `search_graph` as tools and letting it choose is a legitimate architecture, and it is the obvious alternative to our router. We reject it as *the design* for two reasons and keep it as *a measurement*:

1. **It puts an LLM call in the routing path.** That adds ~400–800 ms and ~$0.0125 per query on `claude-opus-5` — roughly 45% on top of generation cost — to decide *how* to retrieve, before retrieving anything. Agentic search is reported at 2–10 s and 3–10× the token cost of advanced RAG.
2. **It collapses A6 vs A7.** An LLM router reads the query text, so its decision cannot be attributed to the structural signal rather than to phrasing. The structural probe is deliberately dumb — presence, degree, `in_lcc` — precisely so that when it wins we know what won.

Tools also do not buy determinism, which is the usual argument for them here: which tool is called and how often remains model-decided, and on Opus 5 `temperature`/`top_p`/`top_k` are **removed and return a 400**, so the "set temperature to 0" escape hatch does not exist.

As an arm it is genuinely valuable: if `A6` lands within a couple of points of `A9` at a fraction of the cost, that is a stronger result than `A6` beating fixed arms — it says a cheap structural signal recovers most of what an expensive reasoning loop achieves.

### A6 vs A7 is the experiment

Beating fixed strategies only proves routing works, which is already established. Our claim is that the *corpus* signal beats the *phrasing* signal, so A7 is the baseline that matters.

This is not a formality. A 2026 study reports a query-only DeBERTa classifier **outperforming an entity-based NER router**, and another reports TF-IDF + SVM at 93.2% accuracy on query-complexity routing, concluding "surface keyword patterns are strong predictors." Entity presence alone has already lost once. **Local connectivity has to be what makes the difference** — and if it doesn't, that is the honest finding and we report it.

---

## 6. Harness engineering standard

Derived from [Anthropic's eval guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) and the eval-health checklist bundled with the Claude API skill. These are **construction requirements**, not review items — the runner and grader satisfy them by default.

### The threat that reshaped the arm list

**"Answerable from memory."** MultiHop-RAG is real news about real entities. Claude may answer a meaningful fraction of the golden set from parametric memory *without retrieving anything*, which compresses the dynamic range of every arm and makes retrieval look less valuable than it is — or more, depending on which way it cuts. We therefore run **`A0`, closed book**, and report every arm's lift *over closed book*, not over zero. This single control is cheap and no published RAG comparison in our research had one.

### Separation of plumbing from model failure

The central failure mode is **conflation** — an infra error landing in the same column as a genuine result.

- Attempts that never produced a scorable output go to **`errors.jsonl`** with a failure class (harness error, serving error, timeout, model mismatch) — **never** into `results.jsonl`, where they would occupy the `(case, arm, rep)` slot, block resume, and score plumbing as a model failure.
- Rows that did produce output carry `stop_reason` and `status: "truncated"` when the response hit `max_tokens`, so a clipped answer is counted and shown but not averaged in as wrong.

### "Abstained" is not "produced nothing"

**This is the most dangerous conflation in this particular harness**, because a whole query class is unanswerable. A correct refusal and a crashed run must never land on the same label:

- `abstained: true` requires an **explicitly parsed refusal**, via structured output.
- An empty, unparseable, or truncated answer is `status: "error"` — it never counts as an abstention, and it never counts as a wrong answer either.
- A runner that errors on every input must not score identically to one that correctly found nothing.

### Before the first paid pass — three cheap tests

1. **Oracle baseline.** Feed the gold answers and gold chunks through the grader. Should approach 100%. If it doesn't, the grader is broken, not the model.
2. **Null baseline.** Feed empty output and a constant answer. Should score 0. If it doesn't, the grader is too lenient.
3. **Mechanism-wired test.** Disable the graph and confirm `A5`'s score drops; enable it and confirm traversal appears in the transcript. If the score barely moves either way, the eval is not measuring the lever we plan to pull.

These are two runs and minutes of compute, and they catch most wiring bugs before a full pass costs real money.

### Measurement hygiene

| Rule | Why |
|---|---|
| Tokens from the API `usage` block, never estimated from string length | Estimates are off by enough to reverse a cost comparison |
| Cost derived from recorded tokens **and the row's actual model**, including cache rates | A flat assumed rate hides the thing we are measuring |
| **Judge cost recorded separately** (`judge_model`, `judge_usage`) | Otherwise the judge's spend dampens differences between arms |
| Latency timed on the **final successful request only** | Retries, backoff sleeps and local queueing must stay out of the model-latency column, or whichever arm hit more 429s looks slower |
| Per-call breakdown, not just per-episode | A slow reranker must be distinguishable from a slow model |
| Assert the **served** model matches the requested model on every call | A provider fallback or capacity reroute silently invalidates the run |
| Compare cache-read share across arms | If one arm runs warm and another cold, part of the cost difference is run order |
| Retries use jittered backoff, capped, with attempt count recorded per row | A zero-delay retry loop multiplies cost invisibly |
| Hard per-case wall-clock ceiling, independent of stream liveness | A hung stream emitting keepalives defeats inactivity timers forever |

### Grader design

- **Atomic checks, not a blended score.** One judge call per property (`correct`, `grounded`, `abstained_correctly`), never one prompt scoring everything.
- **NLI primary, LLM second opinion**, with `judge_disagreement` recorded. Never use the model under test as its own judge.
- **Calibrate the judge against human labels** on a few dozen cases and report agreement. Below ~90% on clear-cut cases, the judge prompt needs another iteration before its scores can steer anything.
- **Test the judge on known negatives**: an empty string, "I don't know", and a confident answer to the wrong question. It must fail all three.
- **Give the judge a way out** — an explicit `Unknown` option when it lacks information — rather than forcing a guess.
- **Grade outcomes, not paths.** Do not require a particular route through the graph; check the answer is correct and grounded.
- **Treat candidate text as untrusted data**, not instructions, in every judge prompt.

### Reproducibility

- Full per-case trajectories saved for every `(case, arm, rep)` — every retrieval result, prompt, response and grader input/output — so a surprising score can be traced without re-running.
- Pinned seeds; sorted iteration anywhere order reaches the model or grader.
- Case set and grader versioned together. Scores from before and after a grader change are not comparable.
- The runner calls the app's **real entry point**. A re-implemented call silently measures a different system.

---

## 7. Statistics, and whether we can even see the effect

**Reps and the noise floor.** One rep is a point estimate with no error bar. The noise floor on a pass-rate metric is roughly `1 / sqrt(n · R)`:

| Cases | Reps | Approx. noise floor |
|---|---|---|
| 120 | 1 | ±9 points |
| 120 | 2 | ±6.5 points |
| 120 | 3 | ±5 points |
| 30 (one class) | 2 | ±13 points |

**Compute this before the first full pass and compare it against the A6 − A7 difference we hope to detect.** If the noise floor exceeds the effect, add reps before adding features — reps are the cheapest lever, and paired designs stretch them further.

**Paired comparisons.** Every arm answers the same questions, so use the per-question *differences*, not two averages. Paired bootstrap (10,000 resamples) over per-query differences, with Bonferroni correction across pairwise arm comparisons. Use [`ranx`](https://github.com/AmenRa/ranx) rather than hand-rolling.

**Report effect size with bootstrap CIs, not bare p-values.** "A6 beats A7 by 0.04 nDCG, 95% CI [0.01, 0.07]" survives scrutiny; "p < 0.05" invites it.

**Non-determinism.** Report `pass@k` where one success suffices and `pass^k` where consistency matters — at a 75% per-trial rate, all three trials pass only 42% of the time, and that gap is worth showing.

**Sample size, stated honestly.** ~120 questions across four classes is ~30 per class; per-class differences will mostly not reach significance. Comparable work uses 824 (FRAMES) or 2,556 (MultiHop-RAG). Power the overall comparison, report per-class as descriptive with intervals, and state the minimum detectable effect up front.

---

## 8. Metrics

| Dimension | Metrics | Judge |
|---|---|---|
| Retrieval quality | hit-rate@k, MRR, nDCG@10 | Deterministic, vs `gold_chunks` |
| Answer quality | faithfulness, answer relevance | LLM, atomic calls |
| Trust | citation precision/recall (ALCE definitions), abstention accuracy | **NLI primary**, LLM second opinion |
| Operations | p50/p95 latency per stage, tokens, cost per query | Measured from `usage` |

**Citation metrics follow [ALCE](https://aclanthology.org/2023.emnlp-main.398.pdf) verbatim.** A citation is *irrelevant* if it alone does not entail the sentence **and** removing it changes nothing. ALCE's own human agreement is κ 0.698 (recall) and 0.525 (precision), and its NLI judge cannot detect *partial* support — so precision is systematically under-reported. We publish the NLI/LLM disagreement rate rather than picking one judge.

---

## 9. Corpus and golden set

**Corpus: MultiHop-RAG.** 609 news articles, 2,556 queries, **ODC-BY** licensed. Natively covers inference / comparison / temporal classes and ships **null (unanswerable) queries as a first-class ~12% class** — the abstention slice, free, with a published baseline. We add ~20 summarisation questions.

Meta's CRAG has better stratification (8 types including false premise, hallucination scored −1) but is **CC BY-NC** — a licence problem for a corporate submission. Cite it for the stratification scheme; do not use it as the corpus.

**Golden set entry:**
```json
{"id": "Q17", "class": "multi_hop",
 "question": "Which outlets covered both the Acme layoffs and the Northwind merger?",
 "gold_chunks": ["c12", "c88"],
 "answer": "Reuters and the Financial Times.",
 "should_abstain": false,
 "gold_provenance": "dataset"}
```

`gold_provenance` records where ground truth came from — `dataset`, `human`, or `model:<id>`. **Never use a model's outputs as gold when that model is under comparison.** Every human-written or human-verified case is tagged as such.

Without `gold_chunks` you cannot compute nDCG at all — you can only ask an LLM whether the answer looked nice, which measures the judge, not the retriever.

**Start small.** Per Anthropic's guidance, begin with 20–50 cases drawn from real failure shapes and get the pipeline working end to end before scaling to the full set. Do not spend a week labelling before the harness has ever run.

---

## 10. Planned repository layout

```
adaptive-retrieval/
├── README.md                     this plan — the source of truth
├── SETUP.md                      local setup + run instructions (written last)
├── WHAT-I-NEED-FROM-YOU.md       credentials, decisions, human input (written last)
├── docker-compose.yml            local Elasticsearch + Kibana
├── pyproject.toml
├── config/
│   └── benchmark.yaml            selects every arm; no code changes between runs
├── golden/
│   └── v1.jsonl
├── src/adaptive_retrieval/
│   ├── config.py                 load + validate benchmark.yaml
│   ├── chunking.py               chunking + content-hash chunk IDs
│   ├── ingest/
│   │   ├── corpus.py             MultiHop-RAG loader
│   │   ├── mapping.py            index mapping: text + semantic_text + sparse_vector
│   │   └── pipeline.py           chunk → bulk index
│   ├── retrieval/
│   │   ├── base.py               Retriever protocol → RetrievalResult
│   │   ├── bm25.py / dense.py / elser.py
│   │   ├── fusion.py             rrf + linear
│   │   ├── rerank.py             text_similarity_reranker
│   │   └── graph_expand.py
│   ├── graph/
│   │   ├── extract.py            Claude entity/relation extraction (Batches, cached)
│   │   ├── store.py              Neptune load
│   │   └── signals.py            presence, degree, in_lcc  ← the routing signal
│   ├── router/
│   │   ├── corpus_router.py      A6
│   │   └── query_router.py       A7
│   ├── generate.py               answer + per-sentence citations
│   ├── metrics/
│   │   ├── retrieval.py          hit_rate@k, MRR, nDCG@10
│   │   ├── citations.py          ALCE precision/recall via NLI
│   │   └── cost.py               tokens, latency, USD
│   ├── harness/
│   │   ├── runner.py             the run loop
│   │   ├── row.py                result row schema
│   │   ├── errors.py             errors.jsonl sidecar + failure classes
│   │   └── results_index.py      write to Elasticsearch
│   └── stats.py                  paired bootstrap + Bonferroni + noise floor
├── runs/                         per-run: results.jsonl, errors.jsonl, trajectories/
├── dashboards/                   Kibana saved objects
└── tests/
```

---

## 11. Implementation phases

Each phase has an **exit criterion**. A phase is not done until it is met, reviewed, and pushed.

### Phase 0 — Skeleton
- [ ] `docker-compose.yml` with Elasticsearch + Kibana for local dev
- [ ] `pyproject.toml`, package skeleton, pinned dependencies, test harness
- [ ] `config.py` — load and **validate** `benchmark.yaml`; fail loudly on unknown arm keys
- [ ] `chunking.py` — content-hash chunk IDs
- **Exit:** `pytest` green; a config with a bad arm raises a clear error.

### Phase 1 — Ingest and the fixed arms
- [ ] Corpus loader (MultiHop-RAG)
- [ ] Index mapping: BM25 text + `semantic_text` dense + ELSER `sparse_vector`, one ES document per chunk
- [ ] Bulk ingest with progress + resumability
- [ ] `A0`, `A1`, `A2`, `A3`, `A3b` driven entirely from config
- [ ] **Measure ELSER ingest throughput on 1,000 chunks and extrapolate** before running the full corpus
- **Exit:** all arms return ranked chunk IDs; no arm requires a code change to select.

### Phase 2 — The harness (already a shippable submission)
- [ ] `A4` reranking
- [ ] Run loop, **interleaved** (question × all arms, not arm × all questions), with reps
- [ ] Result row + `errors.jsonl` sidecar + saved trajectories
- [ ] Retrieval metrics — no LLM involved, so these are the trustworthy ones
- [ ] Citation metrics via NLI, LLM second opinion, `judge_disagreement` recorded
- [ ] **Oracle and null baseline runs** — both must behave as specified in §6
- [ ] `stats.py` — noise floor, paired bootstrap, Bonferroni
- [ ] Kibana panels
- **Exit:** one command produces a scorecard with confidence intervals, and the oracle/null tests pass.

### Phase 3 — The graph
- [ ] Entity/relation extraction via Claude **Batches**, cached on content hash
- [ ] Neptune load with chunk-ID pointers
- [ ] `A5` graph expansion
- [ ] **Mechanism-wired test** — graph off, score drops
- **Exit:** a multi-hop question A4 fails is answered by A5, traceable through the graph path.

### Phase 4 — The claim
- [ ] Routing signals: presence, degree, `in_lcc` — **non-LLM entity extraction**
- [ ] `A6` corpus router + one corrective retry
- [ ] `A7` query-phrasing router
- [ ] Oracle-router panel: best-possible route per query vs what A6 chose
- [ ] Demo UI (stretch)
- **Exit:** A6 vs A7 reported with confidence intervals, and the routing oracle gap quantified.

### Cut order under time pressure
Cut `A8`, then the demo UI, then Phase 3. **Never cut `A7`** — without it nothing tests the claim.

---

## 12. Known risks

| Risk | Mitigation |
|---|---|
| **Router uses an LLM for entity extraction** | Architecture-fatal: adds 300–800 ms to *every* query to save ~600 ms on *some*. Use spaCy/Comprehend NER, target <30 ms. Non-negotiable. |
| **Questions answerable from parametric memory** | `A0` closed-book control; report lift over closed book, not over zero. |
| **Supernodes degrade the signal at scale** | Common entities accumulate huge degree, so "densely connected" becomes true for everything and the router degenerates to always-graph. Mitigate with IDF-weighted entities, normalised degree, drop top-N hubs. |
| **Entity resolution decays as the corpus grows** | More documents → more name collisions. One study cut node duplication 33% with coreference alone. We skip full coreference, use simple normalisation, and treat the graph as an *additive* candidate source so a missing edge costs a candidate, not an answer. |
| **Extraction cost/time at ingest** | ~75% of index cost in GraphRAG. Use the Batches API (50% cost), cache on content hash, start day one. LinearRAG built a competitive entity graph with **zero** LLM tokens — the fallback if cost runs away. |
| **Corrective retry destroys p95** | A retry roughly doubles retrieval *and* generation. Cap at one; report retried and non-retried latency as separate distributions. |
| **Neptune is a fixed cost** | ~$0.48/hr provisioned (~$350/mo) regardless of traffic. The graph must earn that, and the benchmark must be capable of saying it didn't. |
| **Chunk IDs tied to ES `_id`** | One reindex breaks every graph pointer, silently. Content hashes only. |
| **Infra errors scored as model failures** | `errors.jsonl` sidecar, failure classes, `status` field. See §6. |

---

## 13. Working protocol

1. Work proceeds **one section at a time**, in the phase order above.
2. On completing a section, a **fresh review agent** is spawned and instructed to be maximally harsh. Findings are triaged and fixed before moving on.
3. Every section is committed and pushed. This README is the source of truth — re-read at each section boundary and updated when reality diverges from the plan.
4. **No API is coded from memory.** Docs are fetched and verified first, and the verified facts are recorded in §4 with the date checked.
5. Every run that exercises a model costs real money. Cost is estimated and approved before any full pass.

---

## 14. Companion documents

- Architecture walkthrough, stage by stage — https://claude.ai/code/artifact/07f22f45-84d3-4339-8e72-5a2fac052e7a
- Benchmark harness deep-dive — https://claude.ai/code/artifact/f4b3cacc-496a-4864-b4f5-576cb98a1a87

## 15. Key sources

| | |
|---|---|
| Eval design | [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic |
| Fusion | [Bruch et al., TOIS](https://arxiv.org/abs/2210.11934) — convex combination beats RRF on 9/9 datasets |
| Graph, per query class | [GraphRAG-Bench](https://arxiv.org/abs/2506.05690) — graph loses on fact retrieval, wins +10.5 reasoning, +13.1 summarisation |
| Graph vs RAG | [Han et al., Meta/MSU](https://arxiv.org/abs/2502.11371) |
| Closest architecture | [HippoRAG 2](https://arxiv.org/abs/2502.14802) — passage pointers, no text duplication |
| Cost | [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) — 0.1% index cost, >700× lower query cost |
| Routing prior art | [RAGRouter-Bench](https://arxiv.org/abs/2602.00296) — oracle 60.83% vs best router 43.69% |
| Routing framing | [The Coverage Illusion](https://arxiv.org/abs/2605.27220), [LTRR (SIGIR 2025)](https://arxiv.org/abs/2506.13743) |
| Citations | [ALCE](https://aclanthology.org/2023.emnlp-main.398.pdf) |
| Corpus | [MultiHop-RAG](https://arxiv.org/abs/2401.15391) |
| Statistics | [ranx](https://github.com/AmenRa/ranx) |
