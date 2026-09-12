# What I need from you

Everything buildable without credentials, a cluster, or spending money is
built. This is the list of what is left, ordered by what unblocks the most.

Each item says **why** it is needed, so you can decide whether it is worth it
rather than just handing it over.

---

## 1. Decisions only you can make

### 1.1 May the bulk extractor and the judge run on a cheaper model?

**Currently:** everything is `claude-opus-5`, per the default.

Entity extraction runs once per chunk over the whole corpus, and the LLM judge
runs once per property per scored row. These are the two highest-volume model
calls in the project and neither is a reasoning-heavy task.

| | Opus 5 | Sonnet 5 | Haiku 4.5 |
|---|---|---|---|
| Input / output per MTok | $5 / $25 | $2 / $10 | $1 / $5 |

Moving extraction and judging to Sonnet is roughly a **2.5× cost reduction on
the bulk of the spend**. It is a quality/cost tradeoff on tasks where the
literature suggests the cheaper model is adequate — but it is your call, not
mine, so the default stays on Opus until you say otherwise.

**Note:** the judge must not be the same model as the generator. That is
enforced in config, so if you move the generator you must move the judge too,
or vice versa.

### 1.2 Corpus: MultiHop-RAG, or yours?

**Recommended:** MultiHop-RAG. 609 news articles, ODC-BY licensed, and it ships
unanswerable "null" queries as a first-class ~12% class — which is our
abstention slice for free, with a published baseline to compare against.

If you want a BrowserStack-internal corpus instead, say so: the reproducibility
claim changes (a third party cannot re-run it), and I would restructure the
pitch around "reproducible method" rather than "reproducible result".

### 1.3 Is a demo UI actually required?

The submission brief mentions a demo. If the Kibana scorecard satisfies it, I
would rather spend the time on arms than on a UI. If a UI is required, it is a
Phase 4 item and I need to know before then.

### 1.4 Does the graph have to be Neptune?

The graph has a complete in-memory implementation. A5 and A6 run on it today,
with no cloud resource.

Neptune Analytics is **~$0.48/hr provisioned — about $350/month whether or not
a single query uses it**. Since one of the benchmark's honest possible outcomes
is "the graph did not earn its cost", being able to answer that question
*before* provisioning it is worth something.

If the submission requires visible AWS usage, Neptune goes in and I will write
the adapter. If not, the in-memory graph is cheaper, faster and equally
defensible.

### 1.5 Same question for Bedrock AgentCore

With no LLM in the routing path and no tool loop, AgentCore is a managed
runtime for an agent loop we are not running. LangGraph still earns its place —
a router with a bounded retry is a small state machine. If the submission
requires AgentCore, that is a fine reason to keep it; otherwise it is ceremony.

---

## 2. Access I need

| What | Why | Blocks |
|---|---|---|
| **Elastic deployment + API key**, or a go-ahead to run locally | Somewhere to index | Everything past `ar validate` |
| **AWS role**: Bedrock (Claude enabled), S3 | Generation and extraction at corpus scale | Phase 3 onward |
| **Neptune Analytics**, only if 1.4 says yes | The graph backend | Nothing — in-memory works |
| **Cohere API key**, only if you prefer it to Elastic Rerank | A4's reranker | Nothing — Elastic Rerank is in-cluster |
| **Repo visibility decision** | "Reproducible by a third party" is half the pitch, and a private repo is not | The claim, not the code |

If you want the local path instead of a cloud Elastic deployment, I need one
thing: **permission to allocate ~14 GB to Docker**. That is not a nicety —
Elastic documents 8 GB of ML memory to host ELSER and the reranker together,
and under-provisioning produces silently wrong numbers rather than an error.

---

## 3. The one thing I cannot do alone

**Validating the golden set.**

`golden/v1.jsonl` currently holds 7 seed cases with placeholder chunk IDs. It
is a template, not data.

I can draft candidate questions from the corpus and pre-fill `gold_chunks`
automatically. **Someone has to check them.** This is not process for its own
sake: a wrong gold label corrupts every arm's score *equally and invisibly*.
Nothing in the harness can detect it, because from the code's point of view a
wrong label looks exactly like a correct one.

What I need:

- **~2 hours of someone's attention** on a stratified sample of drafted cases.
- A decision on **size**. 120 cases gives a noise floor of ±0.065 at 2 reps;
  one 30-case class in isolation is ±0.129. Comparable published work uses 824
  (FRAMES) or 2,556 (MultiHop-RAG). We should power the *overall* comparison
  and report per-class as descriptive.
- Confirmation that the split should be **stratified** across the four classes,
  or deliberately skewed toward whichever class carries the headline claim.

Per Anthropic's eval guidance, the right move is to start at 20–50 cases and
get the pipeline running end to end before scaling. I would rather have 30
verified cases than 150 unverified ones.

---

## 4. What I will do next without waiting

None of this needs anything above:

- Kibana dashboard definitions for the scorecard
- The oracle-router panel
- A `run` command wiring the executor to a live stack
- Golden-set drafting tooling, so your 2 hours is spent checking rather than writing
- The `A8` contextual-retrieval and `A9` agentic-ceiling arms

---

## 5. Housekeeping

**Rotate the GitHub token.** The one used to create this repo was shared in
plaintext and carries `admin:org`, `delete_repo` and `admin:enterprise` scopes
— far more than this work needs. A `repo`-scoped token is sufficient.
