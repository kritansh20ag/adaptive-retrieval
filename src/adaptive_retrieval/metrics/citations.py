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

**The gate that is easy to miss** is that the per-citation test in ALCE's
reference implementation runs only when the citations *jointly* entail the
statement::

    if joint_entail and len(ref) > 1:
        ...per-citation A/B test...
    else:
        entail_prec += joint_entail

So a statement whose citations do not jointly support it contributes **zero**
to the precision numerator, for every one of its citations, while those
citations still count in the denominator. A fabricated answer therefore scores
precision 0, not 1. Omitting that gate - awarding credit on the A/B test alone
- inflates precision to 1.0 for exactly the answers the metric exists to catch.


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
        joint = bool(cited) and entails(_concat(cited), statement.text)
        recalls.append(1.0 if joint else 0.0)

        # The gate. No joint entailment means no precision credit for any of
        # this statement's citations - they still count in the denominator.
        if not joint:
            continue

        # A single citation that jointly entails is, trivially, the thing that
        # entails. ALCE takes the `else` branch here and adds joint_entail.
        if len(cited) == 1:
            relevant += 1
            continue

        for index in range(len(cited)):
            if entails(cited[index], statement.text):
                relevant += 1
                continue
            # Irrelevant only if the rest still entail without it. If removing
            # it breaks entailment, it was contributing.
            without = cited[:index] + cited[index + 1 :]
            if not (bool(without) and entails(_concat(without), statement.text)):
                relevant += 1

    recall = sum(recalls) / len(recalls)
    precision = (relevant / total_citations) if total_citations else None
    return CitationScores(
        precision=precision,
        recall=recall,
        n_statements=len(statements),
        n_citations=total_citations,
    )
