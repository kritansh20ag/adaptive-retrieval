"""Citation precision and recall, following ALCE verbatim.

Definitions from Gao et al. (EMNLP 2023), reproduced exactly rather than
reinvented, so our numbers are comparable to published ones:

* **Citation recall** of a statement is 1 iff its citation set is non-empty
  *and* the concatenation of all cited passages entails the statement.
* A citation is **irrelevant** iff (a) it alone does not entail the statement
  *and* (b) the remaining citations still entail it. **Citation precision** is
  the fraction of citations that are not irrelevant.

Note what (b) does: a citation that fails alone is still counted as relevant
when dropping it breaks the entailment - because then it was contributing.

**The consequence is surprising and must be reported alongside the number.**
For a statement with exactly ONE citation, removing it leaves an empty premise,
which entails nothing, so (b) is false and the citation is *never* irrelevant.
Citation precision is therefore always 1.0 for singleton citations, even when
the cited passage plainly does not support the sentence.

That is not a bug and not a deviation - it is what ALCE measures. Precision
asks "are these citations padded with unnecessary ones", not "is this sentence
supported". **The unsupported-sentence case is caught by citation recall.**
Reporting precision without recall would therefore hide exactly the failure
most people think precision detects.

The entailment judge is injected. The primary judge is an NLI model, which is
deterministic and free of the position, verbosity and self-preference biases an
LLM judge carries; an LLM can be passed as a second opinion and the
disagreement recorded. Injecting it also makes every rule below testable with
a stub, with no model in the loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["CitationScores", "EntailmentFn", "Statement", "score_citations"]

#: ``(premise, hypothesis) -> entails``. The premise is the concatenation of
#: cited passages; the hypothesis is the statement.
EntailmentFn = Callable[[str, str], bool]


@dataclass(frozen=True, slots=True)
class Statement:
    """One sentence of an answer, with the chunks it cites."""

    text: str
    cited_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationScores:
    precision: float | None
    recall: float | None
    n_statements: int
    n_citations: int

    @property
    def has_citations(self) -> bool:
        return self.n_citations > 0


def _concat(texts: Sequence[str]) -> str:
    return "\n\n".join(texts)


def score_citations(
    statements: Sequence[Statement],
    chunk_texts: dict[str, str],
    entails: EntailmentFn,
) -> CitationScores:
    """Score one answer's citations.

    Returns ``None`` for a metric that is undefined rather than 0.0:

    * With no statements at all - an abstention - both are undefined. There is
      no answer to cite, and scoring 0 would punish a correct refusal.
    * Precision is undefined when nothing was cited; recall is still defined
      and is 0, because failing to cite is a genuine recall failure.

    A cited chunk ID that is not in ``chunk_texts`` is treated as citing
    nothing, which cannot entail anything - a hallucinated citation is a real
    failure and must not raise.
    """
    if not statements:
        return CitationScores(precision=None, recall=None, n_statements=0, n_citations=0)

    total_citations = sum(len(s.cited_chunk_ids) for s in statements)

    recalls: list[float] = []
    relevant = 0
    for statement in statements:
        cited = [chunk_texts.get(cid, "") for cid in statement.cited_chunk_ids]
        supported = bool(cited) and entails(_concat(cited), statement.text)
        recalls.append(1.0 if supported else 0.0)

        for index in range(len(statement.cited_chunk_ids)):
            alone = entails(cited[index], statement.text)
            if alone:
                relevant += 1
                continue
            # ALCE: irrelevant only if the rest still entail without it. If
            # removing it breaks entailment, it was contributing.
            without = cited[:index] + cited[index + 1 :]
            rest_supports = bool(without) and entails(_concat(without), statement.text)
            if not rest_supports:
                relevant += 1

    recall = sum(recalls) / len(recalls)
    precision = (relevant / total_citations) if total_citations else None
    return CitationScores(
        precision=precision,
        recall=recall,
        n_statements=len(statements),
        n_citations=total_citations,
    )
