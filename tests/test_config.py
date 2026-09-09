from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from adaptive_retrieval.config import BenchmarkConfig, ConfigError, load_config

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
  - id: A1
    kind: retrieval
    retriever:
      type: bm25
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_shipped_config_is_valid() -> None:
    """The config we ship must actually load - otherwise nothing else matters."""
    config = load_config(REPO_ROOT / "config" / "benchmark.yaml")
    assert isinstance(config, BenchmarkConfig)
    assert [arm.id for arm in config.arms] == [
        "A0", "A1", "A2", "A3", "A3b", "A4", "A5", "A6", "A7",
    ]


def test_shipped_config_pins_the_things_that_must_not_vary() -> None:
    config = load_config(REPO_ROOT / "config" / "benchmark.yaml")
    # One generator for every arm, or an arm could win on model quality.
    assert config.generator.model == "claude-opus-5"
    # Chunking is a controlled variable.
    assert config.chunking.max_words == 250
    assert config.chunking.sentence_overlap == 0
    # More than one rep, or there are no error bars.
    assert config.run.reps >= 2


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


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo must fail, not be silently ignored."""
    with pytest.raises(ConfigError, match=r"rank_windwo_size|extra_forbidden|Extra inputs"):
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
      rank_windwo_size: 100
""",
            )
        )


def test_unknown_arm_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A9
    kind: telepathy
""",
            )
        )


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


def test_rank_window_smaller_than_k_rejected(tmp_path: Path) -> None:
    """Elasticsearch requires rank_window_size >= size."""
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


def test_rank_window_size_has_no_default(tmp_path: Path) -> None:
    """Omitting it must fail rather than silently inheriting Elastic's 10."""
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
    """Elasticsearch requires weights >= 0."""
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
      weights: {elser: 1.0}
      rank_window_size: 100
""",
            )
        )


def test_rank_constant_below_one_rejected(tmp_path: Path) -> None:
    """Elasticsearch requires rank_constant >= 1."""
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


def test_retry_cannot_exceed_one(tmp_path: Path) -> None:
    """An uncapped corrective loop destroys p95; the cap is structural."""
    with pytest.raises(ConfigError):
        load_config(
            _write(
                tmp_path,
                MINIMAL
                + """
  - id: A6
    kind: router
    signal: corpus
    routes: [A1, A1]
    retry:
      max: 5
      widen_k: 50
""",
            )
        )


def test_arm_lookup(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, MINIMAL))
    assert config.arm("A1").id == "A1"
    with pytest.raises(KeyError, match="no arm with id"):
        config.arm("A404")
