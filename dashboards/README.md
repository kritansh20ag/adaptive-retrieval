# The Kibana scorecard

Four panels. Each answers one question, and together they are the deliverable —
results are *indexed*, not exported, so the scorecard is a live dashboard
someone can slice rather than a screenshot in a slide.

Import `scorecard.ndjson` via **Stack Management → Saved Objects → Import**
after at least one run has been indexed.

## The index

Rows are written to `benchmark-results-v1`, one per `(arm, query, rep)`. Load
them from a run directory with:

```bash
curl -s -H 'Content-Type: application/x-ndjson' \
  -XPOST 'http://localhost:9200/benchmark-results-v1/_bulk' \
  --data-binary @<(awk '{print "{\"index\":{}}"; print}' runs/<run-id>/results.jsonl)
```

## The panels

| Panel | Question | Built from |
|---|---|---|
| **Quality by arm × class** | Where does each strategy actually win? | avg `ndcg_at_k`, split by `query_class`. This is the panel that shows the crossover the whole project is about — an overall mean can hide a strategy that wins two classes and loses two. |
| **Cost per correct answer** | Is the best arm affordable? | `sum(cost_usd) / count(hit_rate_at_k: 1)`. Not cost per query: an arm that is cheap because it answers nothing should not look good. |
| **Latency by stage** | What is slow, and is it the model or us? | percentiles over `latency_ms.retrieve`, `.rerank`, `.graph`, `.generate`. Split by `retried` — a retry roughly doubles the work, so mixing the two distributions makes p95 unreadable. |
| **Router decision audit** | What did A6 choose, and was it right? | `route_taken` against the best-scoring arm for that `query_id`. This is the oracle gap, and it is the most interesting panel here. |

## Two things to filter on, always

- **`status: ok`.** Truncated rows are counted but must not be averaged into
  quality — a response cut off at `max_tokens` is not a wrong answer.
- **`ndcg_at_k: exists`.** Unanswerable questions have no retrieval score.
  Kibana's average would treat a missing value correctly, but an explicit
  filter makes the exclusion visible to whoever reads the dashboard.

## What is deliberately not a panel

**Cost without quality beside it, and quality without cost beside it.** Every
panel above pairs them, because the entire argument is that a strategy which is
two points better and six times more expensive should lose — and a dashboard
that lets you look at either number alone makes that easy to forget.

## Generating the saved objects

`scorecard.ndjson` is generated rather than hand-written, so it stays in step
with the row schema:

```bash
python -m adaptive_retrieval.dashboards > dashboards/scorecard.ndjson
```
