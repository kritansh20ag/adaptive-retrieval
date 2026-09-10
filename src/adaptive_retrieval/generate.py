"""Answer generation: retrieved chunks in, cited answer or explicit refusal out.

Everything here exists to make one distinction machine-checkable: **an
abstention is a decision, and an empty response is a failure.** They must never
share a label, because a whole class of the golden set is unanswerable - if
they collapsed, a runner that errored on every input would score identically to
one that correctly found nothing.

So the model is asked for structured output rather than prose. ``abstained`` is
a boolean the model sets deliberately; it is never inferred by pattern-matching
"I don't know" out of free text, which is exactly the conflation being avoided.
Assistant prefill would be the other way to force the shape, but it returns a
400 on Claude Opus 5.

Model defaults follow the Claude API skill: ``claude-opus-5``, adaptive
thinking (``budget_tokens`` is a 400 on this model), effort via
``output_config``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from adaptive_retrieval.metrics.citations import Statement
from adaptive_retrieval.metrics.cost import cost_usd
from adaptive_retrieval.retrieval.base import RetrievedChunk

__all__ = [
    "AnswerPayload",
    "CitedSentence",
    "GenerationError",
    "GenerationResult",
    "Generator",
    "build_prompt",
]

SYSTEM_PROMPT = """You answer questions strictly from the numbered passages provided.

Rules:
- Every sentence of your answer must cite the passage ids it came from.
- Use ONLY the passages. Do not use prior knowledge, even if you are confident.
- If the passages do not contain enough information to answer, set
  "abstained" to true and return an empty "sentences" list. Abstaining is the
  correct answer to an unanswerable question - it is not a failure.
- Never guess in order to produce an answer."""


class CitedSentence(BaseModel):
    """One sentence of the answer and the passages it came from."""

    text: str = Field(min_length=1)
    cited_chunk_ids: list[str] = Field(default_factory=list)


class AnswerPayload(BaseModel):
    """The structured output contract.

    ``abstained`` is the model's explicit decision. The harness treats an
    unparseable response as an error, never as an abstention.
    """

    abstained: bool
    sentences: list[CitedSentence] = Field(default_factory=list)


class GenerationError(RuntimeError):
    """Raised when a response cannot be scored - a plumbing failure, not a model one."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    payload: AnswerPayload
    served_model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: float
    cost_usd: float

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"

    @property
    def statements(self) -> tuple[Statement, ...]:
        return tuple(
            Statement(text=s.text, cited_chunk_ids=tuple(s.cited_chunk_ids))
            for s in self.payload.sentences
        )


class _MessagesParse(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    @property
    def messages(self) -> _MessagesParse: ...


def build_prompt(question: str, chunks: tuple[RetrievedChunk, ...]) -> str:
    """Render the passages and the question.

    Passages are numbered by chunk ID so the model cites the same identifiers
    the harness scores against, and the question comes last: prompt caching is
    a prefix match, so the stable part must precede the volatile part.
    """
    if not chunks:
        passages = "(no passages were retrieved)"
    else:
        passages = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks)
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


class Generator:
    """Wraps one Claude call and reports what it cost."""

    def __init__(
        self,
        client: _Client,
        *,
        model: str = "claude-opus-5",
        effort: str = "high",
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def answer(self, question: str, chunks: tuple[RetrievedChunk, ...]) -> GenerationResult:
        started = time.perf_counter()
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": build_prompt(question, chunks)}],
            output_format=AnswerPayload,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        stop_reason = getattr(response, "stop_reason", None)
        # A refusal is an HTTP 200 with no usable content. It is a plumbing
        # outcome for our purposes, not the model declining to answer the
        # question on evidential grounds - so it must not become an abstention.
        if stop_reason == "refusal":
            raise GenerationError(
                f"model refused (category="
                f"{getattr(getattr(response, 'stop_details', None), 'category', None)})"
            )

        payload = getattr(response, "parsed_output", None)
        if not isinstance(payload, AnswerPayload):
            raise GenerationError(
                "response did not parse into the answer schema; recording as an error "
                "rather than an abstention, which is a different outcome"
            )
        if payload.abstained and payload.sentences:
            raise GenerationError("model set abstained=true but also returned sentences")

        served_model = str(getattr(response, "model", self.model))
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

        return GenerationResult(
            payload=payload,
            served_model=served_model,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=latency_ms,
            # Priced against the model that ACTUALLY served the request: a
            # provider fallback would otherwise be priced as well as scored
            # wrongly.
            cost_usd=cost_usd(
                served_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
        )
