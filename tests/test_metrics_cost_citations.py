from __future__ import annotations

import pytest

from adaptive_retrieval.metrics.citations import Statement, score_citations
from adaptive_retrieval.metrics.cost import UnknownModelError, cost_usd

# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_opus_pricing() -> None:
    # 1M in at $5, 1M out at $25.
    assert cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0) == pytest.approx(5.0)
    assert cost_usd("claude-opus-5", input_tokens=0, output_tokens=1_000_000) == pytest.approx(25.0)


def test_worked_example_from_the_harness_doc() -> None:
    # 4,118 in + 317 out on Opus 5.
    cost = cost_usd("claude-opus-5", input_tokens=4118, output_tokens=317)
    assert cost == pytest.approx(0.0206 + 0.0079, abs=1e-4)


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    fresh = cost_usd("claude-opus-5", input_tokens=100_000, output_tokens=0)
    cached = cost_usd("claude-opus-5", input_tokens=0, output_tokens=0, cache_read_tokens=100_000)
    assert cached == pytest.approx(fresh * 0.1)


def test_cache_writes_cost_more_than_fresh_input() -> None:
    fresh = cost_usd("claude-opus-5", input_tokens=100_000, output_tokens=0)
    written = cost_usd("claude-opus-5", input_tokens=0, output_tokens=0, cache_write_tokens=100_000)
    assert written == pytest.approx(fresh * 1.25)


def test_bedrock_prefixed_model_resolves() -> None:
    assert cost_usd("anthropic.claude-opus-5", input_tokens=1_000_000, output_tokens=0) == (
        pytest.approx(5.0)
    )


def test_cheaper_models_are_actually_cheaper() -> None:
    args = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert cost_usd("claude-haiku-4-5", **args) < cost_usd("claude-sonnet-5", **args)
    assert cost_usd("claude-sonnet-5", **args) < cost_usd("claude-opus-5", **args)


def test_unknown_model_raises_rather_than_costing_zero() -> None:
    """A silent $0 would make that arm look free and win the cost comparison."""
    with pytest.raises(UnknownModelError, match="no published price"):
        cost_usd("some-other-llm", input_tokens=100, output_tokens=100)


def test_negative_token_counts_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        cost_usd("claude-opus-5", input_tokens=-1, output_tokens=0)


def test_zero_usage_costs_nothing() -> None:
    assert cost_usd("claude-opus-5", input_tokens=0, output_tokens=0) == 0.0


# --------------------------------------------------------------------------
# citations - ALCE definitions
# --------------------------------------------------------------------------

CHUNKS = {"c1": "Reuters covered the layoffs.", "c2": "The FT covered the merger.", "c3": "Noise."}


def _entails_if_contains(premise: str, hypothesis: str) -> bool:
    """Stub judge: entailment iff the premise mentions the hypothesis' keyword."""
    return hypothesis.lower() in premise.lower()


def test_all_citations_supported() -> None:
    statements = [Statement("Reuters", ("c1",)), Statement("FT", ("c2",))]
    scores = score_citations(statements, CHUNKS, _entails_if_contains)
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(1.0)


def test_irrelevant_citation_lowers_precision() -> None:
    """c3 neither entails alone nor is needed for the rest to entail."""
    statements = [Statement("Reuters", ("c1", "c3"))]
    scores = score_citations(statements, CHUNKS, _entails_if_contains)
    assert scores.precision == pytest.approx(0.5)
    assert scores.recall == pytest.approx(1.0)


def test_a_citation_that_fails_alone_but_is_load_bearing_counts_as_relevant() -> None:
    """ALCE clause (b): irrelevant only if the REST still entail without it.

    Neither chunk entails "Reuters and FT" alone, but removing either breaks
    the concatenated entailment, so both are contributing.
    """

    def entails(premise: str, hypothesis: str) -> bool:
        return all(part in premise for part in hypothesis.split(" and "))

    statements = [Statement("Reuters and The FT", ("c1", "c2"))]
    scores = score_citations(statements, CHUNKS, entails)
    assert scores.precision == pytest.approx(1.0)


def test_uncited_statement_scores_zero_recall() -> None:
    scores = score_citations([Statement("Reuters", ())], CHUNKS, _entails_if_contains)
    assert scores.recall == pytest.approx(0.0)
    # Nothing was cited, so precision has no denominator.
    assert scores.precision is None


def test_abstention_leaves_both_undefined_not_zero() -> None:
    """There is no answer to cite; scoring 0 would punish a correct refusal."""
    scores = score_citations([], CHUNKS, _entails_if_contains)
    assert scores.precision is None
    assert scores.recall is None
    assert scores.n_statements == 0


def test_hallucinated_chunk_id_does_not_raise() -> None:
    scores = score_citations([Statement("Reuters", ("c999",))], CHUNKS, _entails_if_contains)
    # Recall catches it; precision cannot - see the singleton test below.
    assert scores.recall == pytest.approx(0.0)
    assert scores.precision == pytest.approx(1.0)


def test_recall_is_averaged_per_statement() -> None:
    statements = [Statement("Reuters", ("c1",)), Statement("Reuters", ())]
    scores = score_citations(statements, CHUNKS, _entails_if_contains)
    assert scores.recall == pytest.approx(0.5)


def test_counts_are_reported() -> None:
    statements = [Statement("Reuters", ("c1", "c3")), Statement("FT", ("c2",))]
    scores = score_citations(statements, CHUNKS, _entails_if_contains)
    assert scores.n_statements == 2
    assert scores.n_citations == 3


def test_singleton_citation_is_never_irrelevant_by_alce_definition() -> None:
    """ALCE clause (b) makes precision 1.0 for any single citation, because
    removing it leaves an empty premise that entails nothing.

    This is what ALCE measures, not a bug - precision asks whether citations
    are padded, not whether the sentence is supported. It is exactly why
    precision must never be reported without recall.
    """
    scores = score_citations([Statement("Reuters", ("c3",))], CHUNKS, _entails_if_contains)
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(0.0)


def test_three_sentences_one_unsupported() -> None:
    """The example from the harness doc. The unsupported sentence shows up in
    RECALL at 2/3; precision stays 1.0 because every citation is a singleton."""
    statements = [
        Statement("Reuters", ("c1",)),
        Statement("FT", ("c2",)),
        Statement("Reuters", ("c3",)),
    ]
    scores = score_citations(statements, CHUNKS, _entails_if_contains)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.precision == pytest.approx(1.0)
