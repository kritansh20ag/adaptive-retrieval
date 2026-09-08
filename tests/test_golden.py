from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_retrieval.golden import (
    GoldenCase,
    GoldenSetError,
    QueryClass,
    check_provenance,
    class_distribution,
    load_golden_set,
)

ANSWERABLE = {
    "id": "Q17",
    "class": "multi_hop",
    "question": "Which outlets covered both the Acme layoffs and the Northwind merger?",
    "gold_chunks": ["c12", "c88"],
    "answer": "Reuters and the Financial Times.",
    "should_abstain": False,
}

UNANSWERABLE = {
    "id": "Q18",
    "class": "unanswerable",
    "question": "What was Acme's Q3 headcount after the layoffs?",
    "gold_chunks": [],
    "answer": None,
    "should_abstain": True,
}


def _write(tmp_path: Path, *cases: dict[str, object]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_loads_both_kinds_of_case(tmp_path: Path) -> None:
    cases = load_golden_set(_write(tmp_path, ANSWERABLE, UNANSWERABLE))
    assert [c.id for c in cases] == ["Q17", "Q18"]
    assert cases[0].query_class is QueryClass.MULTI_HOP
    assert cases[1].should_abstain


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(ANSWERABLE) + "\n\n\n", encoding="utf-8")
    assert len(load_golden_set(path)) == 1


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="cannot read golden set"):
        load_golden_set(tmp_path / "nope.jsonl")


def test_empty_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="contains no cases"):
        load_golden_set(path)


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(ANSWERABLE) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match=r"golden\.jsonl:2 is not valid JSON"):
        load_golden_set(path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="duplicate case ids"):
        load_golden_set(_write(tmp_path, ANSWERABLE, ANSWERABLE))


def test_unknown_field_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError):
        load_golden_set(_write(tmp_path, {**ANSWERABLE, "difficulty": "hard"}))


def test_unknown_class_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError):
        load_golden_set(_write(tmp_path, {**ANSWERABLE, "class": "trick_question"}))


# --------------------------------------------------------------------------
# answerability consistency - a labelling bug poisons two metrics at once
# --------------------------------------------------------------------------


def test_answerable_case_needs_gold_chunks(tmp_path: Path) -> None:
    """Without gold_chunks retrieval metrics are uncomputable for that case."""
    with pytest.raises(GoldenSetError, match="no gold_chunks"):
        load_golden_set(_write(tmp_path, {**ANSWERABLE, "gold_chunks": []}))


def test_answerable_case_needs_a_reference_answer(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="no reference answer"):
        load_golden_set(_write(tmp_path, {**ANSWERABLE, "answer": None}))


def test_unanswerable_case_may_not_have_gold_chunks(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="unanswerable but carries"):
        load_golden_set(_write(tmp_path, {**UNANSWERABLE, "gold_chunks": ["c1"]}))


def test_unanswerable_case_may_not_have_an_answer(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="unanswerable but carries a reference answer"):
        load_golden_set(_write(tmp_path, {**UNANSWERABLE, "answer": "42"}))


def test_unanswerable_class_must_set_should_abstain(tmp_path: Path) -> None:
    case = {**UNANSWERABLE, "should_abstain": False, "gold_chunks": [], "answer": None}
    with pytest.raises(GoldenSetError, match="should_abstain is False"):
        load_golden_set(_write(tmp_path, case))


def test_should_abstain_requires_the_unanswerable_class(tmp_path: Path) -> None:
    case = {**ANSWERABLE, "should_abstain": True, "gold_chunks": [], "answer": None}
    with pytest.raises(GoldenSetError, match="not 'unanswerable'"):
        load_golden_set(_write(tmp_path, case))


def test_duplicate_gold_chunk_rejected(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="more than once"):
        load_golden_set(_write(tmp_path, {**ANSWERABLE, "gold_chunks": ["c12", "c12"]}))


# --------------------------------------------------------------------------
# provenance - never score a model against its own output
# --------------------------------------------------------------------------


def test_gold_from_a_model_under_test_is_rejected() -> None:
    cases = [GoldenCase.model_validate({**ANSWERABLE, "gold_provenance": "model:claude-opus-5"})]
    with pytest.raises(GoldenSetError, match="itself under test"):
        check_provenance(cases, {"claude-opus-5"})


def test_gold_from_a_different_model_is_allowed() -> None:
    cases = [GoldenCase.model_validate({**ANSWERABLE, "gold_provenance": "model:some-other-llm"})]
    check_provenance(cases, {"claude-opus-5"})


def test_dataset_and_human_provenance_always_allowed() -> None:
    cases = [
        GoldenCase.model_validate({**ANSWERABLE, "gold_provenance": "dataset"}),
        GoldenCase.model_validate({**UNANSWERABLE, "gold_provenance": "human"}),
    ]
    check_provenance(cases, {"claude-opus-5"})


def test_provenance_error_names_offending_cases() -> None:
    cases = [
        GoldenCase.model_validate(
            {**ANSWERABLE, "id": f"Q{i}", "gold_provenance": "model:claude-opus-5"}
        )
        for i in range(3)
    ]
    with pytest.raises(GoldenSetError, match=r"Q0"):
        check_provenance(cases, {"claude-opus-5"})


def test_default_provenance_is_dataset() -> None:
    assert GoldenCase.model_validate(ANSWERABLE).gold_provenance == "dataset"


# --------------------------------------------------------------------------
# stratification
# --------------------------------------------------------------------------


def test_class_distribution_reports_every_class_including_empty_ones() -> None:
    cases = [GoldenCase.model_validate(ANSWERABLE), GoldenCase.model_validate(UNANSWERABLE)]
    assert class_distribution(cases) == {
        "single_hop": 0,
        "multi_hop": 1,
        "summarisation": 0,
        "unanswerable": 1,
    }
