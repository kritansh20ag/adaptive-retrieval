from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from adaptive_retrieval.generate import (
    AnswerPayload,
    CitedSentence,
    GenerationError,
    Generator,
    ModelMismatchError,
    build_prompt,
)
from adaptive_retrieval.retrieval.base import RetrievedChunk

CHUNKS = (
    RetrievedChunk(chunk_id="c12", text="Reuters covered the layoffs.", score=1.0),
    RetrievedChunk(chunk_id="c88", text="The FT covered the merger.", score=0.9),
)


class FakeMessages:
    """Stands in for client.messages, recording what it was called with."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeMessages(response)


def _response(
    payload: AnswerPayload | None,
    *,
    stop_reason: str | None = "end_turn",
    model: str = "claude-opus-5",
    usage: Any = None,
    stop_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=payload,
        stop_reason=stop_reason,
        model=model,
        usage=usage or SimpleNamespace(input_tokens=4118, output_tokens=317),
        stop_details=stop_details,
    )


def _answer() -> AnswerPayload:
    return AnswerPayload(
        abstained=False,
        sentences=[CitedSentence(text="Reuters and the FT.", cited_chunk_ids=["c12", "c88"])],
    )


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def test_prompt_labels_passages_by_chunk_id() -> None:
    """The model must cite the same identifiers the harness scores against."""
    prompt = build_prompt("Who covered both?", CHUNKS)
    assert "[c12]" in prompt
    assert "[c88]" in prompt


def test_question_comes_after_the_passages() -> None:
    """Caching is a prefix match: volatile content goes last."""
    prompt = build_prompt("Who covered both?", CHUNKS)
    assert prompt.index("Reuters covered") < prompt.index("Who covered both?")


def test_prompt_handles_no_retrieved_passages() -> None:
    """A0 is closed book, and a failed retrieval still has to produce a prompt."""
    assert "no passages" in build_prompt("Anything?", ())


# --------------------------------------------------------------------------
# the abstention contract
# --------------------------------------------------------------------------


def test_explicit_abstention_is_carried_through() -> None:
    client = FakeClient(_response(AnswerPayload(abstained=True, sentences=[])))
    result = Generator(client).answer("Unanswerable?", CHUNKS)
    assert result.payload.abstained is True
    assert result.statements == ()


def test_unparseable_response_is_an_error_not_an_abstention() -> None:
    """A crashed run and a correct refusal must never share a label."""
    client = FakeClient(_response(None))
    with pytest.raises(GenerationError, match="did not parse"):
        Generator(client).answer("Q?", CHUNKS)


def test_refusal_is_an_error_not_an_abstention() -> None:
    """A safety refusal is a plumbing outcome, not the model declining on
    evidential grounds."""
    client = FakeClient(
        _response(None, stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    )
    with pytest.raises(GenerationError, match="refused"):
        Generator(client).answer("Q?", CHUNKS)


def test_contradictory_payload_is_rejected() -> None:
    payload = AnswerPayload(
        abstained=True, sentences=[CitedSentence(text="But also this.", cited_chunk_ids=["c12"])]
    )
    client = FakeClient(_response(payload))
    with pytest.raises(GenerationError, match="abstained=true but also returned sentences"):
        Generator(client).answer("Q?", CHUNKS)


# --------------------------------------------------------------------------
# request shape - the Claude API contract
# --------------------------------------------------------------------------


def test_uses_adaptive_thinking_and_no_budget_tokens() -> None:
    """budget_tokens returns a 400 on Claude Opus 5."""
    client = FakeClient(_response(_answer()))
    Generator(client).answer("Q?", CHUNKS)
    call = client.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in call


def test_effort_goes_in_output_config() -> None:
    client = FakeClient(_response(_answer()))
    Generator(client, effort="max").answer("Q?", CHUNKS)
    assert client.messages.calls[0]["output_config"] == {"effort": "max"}


def test_requests_structured_output() -> None:
    client = FakeClient(_response(_answer()))
    Generator(client).answer("Q?", CHUNKS)
    assert client.messages.calls[0]["output_format"] is AnswerPayload


def test_no_assistant_prefill() -> None:
    """Prefill returns a 400 on Opus 5."""
    client = FakeClient(_response(_answer()))
    Generator(client).answer("Q?", CHUNKS)
    roles = [m["role"] for m in client.messages.calls[0]["messages"]]
    assert roles == ["user"]


def test_model_default_is_opus_5() -> None:
    client = FakeClient(_response(_answer()))
    Generator(client).answer("Q?", CHUNKS)
    assert client.messages.calls[0]["model"] == "claude-opus-5"


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------


def test_a_substituted_model_is_a_loud_error() -> None:
    """A provider fallback silently invalidates the run: the scores are no
    longer about the model the config names."""
    client = FakeClient(_response(_answer(), model="claude-sonnet-5"))
    with pytest.raises(ModelMismatchError, match="served by"):
        Generator(client, model="claude-opus-5").answer("Q?", CHUNKS)


def test_platform_qualified_ids_are_not_a_mismatch() -> None:
    """Bedrock serves "us.anthropic.claude-opus-5" for "claude-opus-5"; that is
    the same model, and it must price correctly rather than raise."""
    client = FakeClient(
        _response(
            _answer(),
            model="us.anthropic.claude-opus-5",
            usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0),
        )
    )
    result = Generator(client, model="claude-opus-5").answer("Q?", CHUNKS)
    assert result.served_model == "us.anthropic.claude-opus-5"
    assert result.cost_usd == pytest.approx(5.0)


def test_an_unpriceable_model_does_not_discard_the_answer() -> None:
    """Cost is metadata about the row, not a precondition for scoring it."""
    client = FakeClient(_response(_answer(), model="claude-future-9"))
    result = Generator(client, model="claude-future-9").answer("Q?", CHUNKS)
    assert result.cost_usd is None
    assert result.payload.abstained is False


def test_tokens_come_from_the_usage_block() -> None:
    client = FakeClient(_response(_answer()))
    result = Generator(client).answer("Q?", CHUNKS)
    assert result.input_tokens == 4118
    assert result.output_tokens == 317


def test_cache_tokens_are_recorded() -> None:
    client = FakeClient(
        _response(
            _answer(),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                cache_read_input_tokens=5000,
                cache_creation_input_tokens=200,
            ),
        )
    )
    result = Generator(client).answer("Q?", CHUNKS)
    assert result.cache_read_tokens == 5000
    assert result.cache_write_tokens == 200


def test_truncation_is_flagged_separately() -> None:
    client = FakeClient(_response(_answer(), stop_reason="max_tokens"))
    result = Generator(client).answer("Q?", CHUNKS)
    assert result.truncated is True
    assert result.payload.abstained is False


def test_statements_carry_their_citations() -> None:
    client = FakeClient(_response(_answer()))
    result = Generator(client).answer("Q?", CHUNKS)
    assert result.statements[0].cited_chunk_ids == ("c12", "c88")
