from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import pytest

from adaptive_retrieval.config import BenchmarkConfig, ConfigError, RetrievalArm, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL = """
corpus: test-corpus
golden_set: golden/v1.jsonl
judges:
  entailment: deberta-v3-large-mnli
defaults:
  index: chunks-v1
  k: 10
arms:
  - id: A0
    kind: closed_book
  - id: A1
    kind: retrieval
    retriever:
      type: bm25
"""

ROUTABLE = """
  - id: A2
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
      rank_window_size: 100
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the shipped config
# --------------------------------------------------------------------------


def test_shipped_config_is_valid() -> None:
    config = load_config(REPO_ROOT / "config" / "benchmark.yaml")
    assert isinstance(config, BenchmarkConfig)
    assert {arm.id for arm in config.arms} >= {"A0", "A6", "A7"}


def test_shipped_config_pins_the_things_that_must_not_vary() -> None:
    config = load_config(REPO_ROOT / "config" / "benchmark.yaml")
    # One generator for every arm, or an arm could win on model quality.
    assert config.generator.model == "claude-opus-5"
    # Chunking is a controlled variable.
    assert config.chunking.max_words == 250
    assert config.chunking.sentence_overlap == 0
    # More than one rep, or there are no error bars.
    assert config.run.reps >= 2
    # A model may not judge its own output.
    assert config.judges.second_opinion != config.generator.model


def test_shipped_a5_is_a4_plus_graph_expansion_only() -> None:
    """The arm-nesting claim, checked rather than assumed."""
    config = load_config(REPO_ROOT / "config" / "benchmark.yaml")
    a4 = config.resolved_arm("A4")
    a5 = config.resolved_arm("A5")
    assert isinstance(a4, RetrievalArm) and isinstance(a5, RetrievalArm)
    assert a5.retriever == a4.retriever
    assert a5.rerank == a4.rerank
    assert a4.graph_expansion is False
    assert a5.graph_expansion is True


def test_minimal_config_loads(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, MINIMAL))
    assert config.corpus == "test-corpus"
    assert config.defaults.k == 10


# --------------------------------------------------------------------------
# failing loudly
# --------------------------------------------------------------------------


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text("arms: [\n  - broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(path)


def test_typo_on_a_defaulted_field_is_rejected(tmp_path: Path) -> None:
    """The real risk. A typo on a *required* field is caught anyway by the
    missing-field error, so it proves nothing about extra="forbid"; a typo on a
    field that has a default is silently ignored without it."""
    with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
      rank_window_size: 100
      rank_constnat: 60
""",
            )
        )


def test_error_message_names_the_arm_not_its_index(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"arm 'A3'"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
""",
            )
        )


def test_unknown_arm_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"telepathy|discriminator|union_tag|tag"):
        load_config(_write(tmp_path, MINIMAL + "\n  - id: A9\n    kind: telepathy\n"))


def test_duplicate_arm_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duplicate arm ids"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A1
    kind: retrieval
    retriever:
      type: dense
""",
            )
        )


def test_missing_closed_book_arm_rejected(tmp_path: Path) -> None:
    """A0 is the denominator for every reported number."""
    without_a0 = MINIMAL.replace("  - id: A0\n    kind: closed_book\n", "")
    with pytest.raises(ConfigError, match="no closed_book arm defined"):
        load_config(_write(tmp_path, without_a0))


def test_model_may_not_judge_its_own_output(tmp_path: Path) -> None:
    body = MINIMAL.replace(
        "judges:\n  entailment: deberta-v3-large-mnli",
        "generator:\n  model: claude-opus-5\njudges:\n"
        "  entailment: deberta-v3-large-mnli\n  second_opinion: claude-opus-5",
    )
    with pytest.raises(ConfigError, match="may not judge its own output"):
        load_config(_write(tmp_path, body))


# --------------------------------------------------------------------------
# arm inheritance
# --------------------------------------------------------------------------


def test_extending_arm_may_not_redefine_the_retriever(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="may only add its one declared difference"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A5
    kind: retrieval
    extends: A1
    retriever:
      type: dense
""",
            )
        )


def test_extending_an_undefined_arm_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="extends undefined arm"):
        load_config(
            _write(tmp_path, MINIMAL + "\n  - id: A5\n    kind: retrieval\n    extends: A99\n")
        )


def test_extending_a_non_retrieval_arm_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a retrieval arm"):
        load_config(
            _write(tmp_path, MINIMAL + "\n  - id: A5\n    kind: retrieval\n    extends: A0\n")
        )


def test_arm_with_neither_retriever_nor_extends_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must define a retriever or extend"):
        load_config(_write(tmp_path, MINIMAL + "\n  - id: A5\n    kind: retrieval\n"))


def test_resolved_arm_inherits_parent_retriever(tmp_path: Path) -> None:
    config = load_config(
        _write(
            tmp_path,
            MINIMAL
            + "\n  - id: A5\n    kind: retrieval\n    extends: A1\n    graph_expansion: true\n",
        )
    )
    a1 = config.resolved_arm("A1")
    a5 = config.resolved_arm("A5")
    assert isinstance(a1, RetrievalArm) and isinstance(a5, RetrievalArm)
    assert a5.retriever == a1.retriever
    assert a5.graph_expansion is True


# --------------------------------------------------------------------------
# retriever / reranker bounds
# --------------------------------------------------------------------------


def test_rank_window_smaller_than_k_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"must be >= defaults\.k"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
      rank_window_size: 5
""",
            )
        )


def test_rerank_depth_smaller_than_k_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"rerank\.depth"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A4
    kind: retrieval
    retriever:
      type: bm25
    rerank:
      inference_id: .rerank-v1-elasticsearch
      depth: 3
""",
            )
        )


def test_rerank_depth_above_child_window_rejected(tmp_path: Path) -> None:
    """The reranker sees only what its child produced; publishing depth=100
    against a 20-candidate child reports a rerank that never happened."""
    with pytest.raises(ConfigError, match="exceeds the child retriever"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A4
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
      rank_window_size: 20
    rerank:
      inference_id: .rerank-v1-elasticsearch
      depth: 100
""",
            )
        )


def test_rank_window_size_has_no_default(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="rank_window_size"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
""",
            )
        )


def test_linear_normalizer_has_no_default(tmp_path: Path) -> None:
    """Elastic's default is `none`, which lets raw BM25 scores dwarf cosine
    similarities and degenerates this arm toward BM25."""
    with pytest.raises(ConfigError, match="normalizer"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3b
    kind: retrieval
    retriever:
      type: linear
      sources: [bm25, dense]
      rank_window_size: 100
""",
            )
        )


def test_fusion_with_one_source_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="at least 2 sources"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25]
      rank_window_size: 100
""",
            )
        )


def test_fusion_with_duplicate_sources_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be unique"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, bm25]
      rank_window_size: 100
""",
            )
        )


def test_negative_linear_weight_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="weights must be >= 0"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3b
    kind: retrieval
    retriever:
      type: linear
      sources: [bm25, dense]
      normalizer: minmax
      weights: {bm25: -1.0}
      rank_window_size: 100
""",
            )
        )


def test_weight_for_absent_source_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="weights reference sources not in this arm"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3b
    kind: retrieval
    retriever:
      type: linear
      sources: [bm25, dense]
      normalizer: minmax
      weights: {elser: 1.0}
      rank_window_size: 100
""",
            )
        )


def test_rank_constant_below_one_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="rank_constant"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A3
    kind: retrieval
    retriever:
      type: rrf
      sources: [bm25, dense]
      rank_constant: 0
      rank_window_size: 100
""",
            )
        )


# --------------------------------------------------------------------------
# routers
# --------------------------------------------------------------------------


def test_router_referencing_undefined_arm_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="routes to undefined arm"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A99]
    retry:
      max: 1
      widen_k: 50
""",
            )
        )


def test_router_routing_to_itself_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="routes to itself"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A6]
    retry:
      max: 1
      widen_k: 50
""",
            )
        )


def test_router_routing_to_another_router_rejected(tmp_path: Path) -> None:
    """Mutual recursion would blow the stack mid-run, after money was spent."""
    with pytest.raises(ConfigError, match="which is itself a router"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + ROUTABLE
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A7]
    retry:
      max: 1
      widen_k: 50
  - id: A7
    kind: router
    signal: query
    routes: [A1, A2]
    retry:
      max: 1
      widen_k: 50
""",
            )
        )


def test_retry_that_narrows_rejected(tmp_path: Path) -> None:
    """A "broader" retry with widen_k below k silently narrows it."""
    with pytest.raises(ConfigError, match="must broaden"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + ROUTABLE
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A2]
    retry:
      max: 1
      widen_k: 5
""",
            )
        )


def test_retry_above_route_window_rejected(tmp_path: Path) -> None:
    """Elasticsearch requires rank_window_size >= size, so this is a 400 at
    query time, mid-run, on the arm that is the contribution."""
    with pytest.raises(ConfigError, match="exceeds route"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + ROUTABLE
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A2]
    retry:
      max: 1
      widen_k: 500
""",
            )
        )


def test_retry_cannot_exceed_one(tmp_path: Path) -> None:
    """Routes are distinct here, so only the retry cap can fire."""
    with pytest.raises(ConfigError, match=r"less than or equal to 1"):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + ROUTABLE
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A2]
    retry:
      max: 5
      widen_k: 50
""",
            )
        )


# --------------------------------------------------------------------------
# immutability and warnings
# --------------------------------------------------------------------------


def test_sequences_are_tuples_so_validators_cannot_be_bypassed(tmp_path: Path) -> None:
    """frozen=True blocks attribute assignment but not list mutation."""
    config = load_config(_write(tmp_path, MINIMAL))
    assert isinstance(config.arms, tuple)
    with pytest.raises(AttributeError):
        config.arms.append("nonsense")  # type: ignore[attr-defined]


def test_single_rep_warns(tmp_path: Path) -> None:
    body = MINIMAL.replace("defaults:", "run:\n  reps: 1\ndefaults:")
    with pytest.warns(UserWarning, match="point estimate with no error bar"):
        load_config(_write(tmp_path, body))


def test_two_reps_does_not_warn(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_config(_write(tmp_path, MINIMAL))


def test_arm_lookup(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, MINIMAL))
    assert config.arm("A1").id == "A1"
    with pytest.raises(KeyError, match="no arm with id"):
        config.arm("A404")
