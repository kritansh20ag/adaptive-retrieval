"""Cost accounting from real token counts.

Two rules, both from the eval-health checklist:

* Token counts come from the API's ``usage`` block, never from string-length
  estimates. Estimates are off by enough to reverse a cost comparison.
* Cost is derived from the row's **actual** model, including cache rates -
  never a flat assumed rate, which would hide the very thing being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MODEL_PRICING", "ModelPricing", "UnknownModelError", "cost_usd"]


class UnknownModelError(KeyError):
    """Raised for a model with no published price.

    Deliberately fatal rather than defaulting to zero: a silent $0 would make
    an arm look free and win the cost comparison outright.
    """


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    #: Cache reads are ~0.1x input; cache writes ~1.25x input.
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


#: Anthropic first-party rates. Bedrock and Vertex are partner-operated and
#: priced separately - if the run goes through Bedrock these must be replaced
#: with that platform's rates, or the cost column is fiction.
MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-fable-5-1": ModelPricing(10.00, 50.00),
    "claude-fable-5": ModelPricing(10.00, 50.00),
    "claude-opus-5": ModelPricing(5.00, 25.00),
    "claude-opus-4-8": ModelPricing(5.00, 25.00),
    "claude-sonnet-5": ModelPricing(2.00, 10.00),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00),
}


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost of one call, from counts the API reported.

    ``model`` should be the model the response says served the request, not the
    one the request asked for - a provider fallback would otherwise be priced
    wrongly as well as scored wrongly.
    """
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cache_read_tokens", cache_read_tokens),
        ("cache_write_tokens", cache_write_tokens),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    # Bedrock model ids carry an "anthropic." prefix over the same model.
    key = model.removeprefix("anthropic.")
    try:
        pricing = MODEL_PRICING[key]
    except KeyError as exc:
        raise UnknownModelError(
            f"no published price for model {model!r}. Add it to MODEL_PRICING rather than "
            f"letting the run report a zero cost, which would make this arm look free."
        ) from exc

    per_token = pricing.input_per_mtok / 1_000_000
    return (
        input_tokens * per_token
        + output_tokens * (pricing.output_per_mtok / 1_000_000)
        + cache_read_tokens * per_token * pricing.cache_read_multiplier
        + cache_write_tokens * per_token * pricing.cache_write_multiplier
    )
