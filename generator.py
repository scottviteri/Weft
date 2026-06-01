"""Text generation using Together API endpoint."""

import os
import math
from together import Together
from dataclasses import dataclass, field

# Together dedicated endpoints get a fresh hash suffix each time they're
# recreated, so the model is read from $WEFT_MODEL when set. Update this
# fallback or export WEFT_MODEL to point at your current endpoint.
DEFAULT_MODEL = "sviteri/Qwen/Qwen3-30B-A3B-Base-65239313"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: str = field(default_factory=lambda: os.environ.get("WEFT_MODEL", DEFAULT_MODEL))
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    num_branches: int = 3


@dataclass
class Generation:
    """A single generated continuation plus its per-token logprobs.

    logprobs is the quantitative dual of the qualitative `analyze` readout:
    {"tokens": [...], "token_logprobs": [...]} when the endpoint returns them,
    else None. Kept as a plain dict so it round-trips through JSON unchanged.
    """
    text: str
    logprobs: dict | None = None


def mean_logprob(logprobs: dict | None) -> float | None:
    """Average per-token logprob of a generation, or None if unavailable.

    None entries (e.g. the very first token of a prompt, which has no
    preceding context) are skipped.
    """
    if not logprobs:
        return None
    lps = [x for x in (logprobs.get("token_logprobs") or []) if x is not None]
    if not lps:
        return None
    return sum(lps) / len(lps)


def slice_logprobs(logprobs: dict | None, start_char: int, end_char: int) -> dict | None:
    """Return the tokens/logprobs whose text falls within [start_char, end_char).

    Used to pull one node's tokens out of an echoed full-prompt scoring. A
    token straddling either boundary is dropped, so the returned tokens join
    back to exactly text[start_char:end_char] when alignment is clean.
    """
    if not logprobs:
        return None
    tokens = logprobs.get("tokens") or []
    tlps = logprobs.get("token_logprobs") or []
    if len(tokens) != len(tlps):
        return None

    out_tokens, out_lps, pos = [], [], 0
    for tok, lp in zip(tokens, tlps):
        tok_end = pos + len(tok)
        if pos >= start_char and tok_end <= end_char:
            out_tokens.append(tok)
            out_lps.append(lp)
        pos = tok_end
        if pos >= end_char:
            break
    if not out_tokens:
        return None
    return {"tokens": out_tokens, "token_logprobs": out_lps}


def perplexity(logprobs: dict | None) -> float | None:
    """Perplexity (exp of mean surprisal); lower = the model was more confident."""
    m = mean_logprob(logprobs)
    return math.exp(-m) if m is not None else None


def _extract(choice) -> Generation:
    """Build a Generation from a Together completion choice."""
    lp = getattr(choice, "logprobs", None)
    logprobs = None
    if lp is not None:
        tokens = getattr(lp, "tokens", None)
        token_logprobs = getattr(lp, "token_logprobs", None)
        if tokens is not None or token_logprobs is not None:
            logprobs = {"tokens": tokens, "token_logprobs": token_logprobs}
    return Generation(text=choice.text, logprobs=logprobs)


class Generator:
    """Generate text continuations using Together API."""

    def __init__(self, config: GenerationConfig | None = None):
        self.client = Together()
        self.config = config or GenerationConfig()

    def generate_continuations(
        self,
        prompt: str,
        num_branches: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Generation]:
        """Generate multiple possible continuations for a prompt.

        Uses a single request with `n=` so the branches cost one round-trip
        rather than one per branch.
        """
        n = num_branches or self.config.num_branches
        tokens = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature

        response = self.client.completions.create(
            model=self.config.model,
            prompt=prompt,
            max_tokens=tokens,
            temperature=temp,
            top_p=self.config.top_p,
            n=n,
            logprobs=1,
        )
        return [_extract(c) for c in response.choices]

    def generate_single(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generation:
        """Generate a single continuation."""
        tokens = max_tokens or self.config.max_tokens
        temp = temperature or self.config.temperature

        response = self.client.completions.create(
            model=self.config.model,
            prompt=prompt,
            max_tokens=tokens,
            temperature=temp,
            top_p=self.config.top_p,
            logprobs=1,
        )
        return _extract(response.choices[0])

    def score(self, text: str) -> Generation:
        """Score existing text: echo it back with per-token logprobs.

        No sampling — this asks the model how probable the given tokens are
        in context, used to (re)compute logprobs for text already in the tree.
        """
        response = self.client.completions.create(
            model=self.config.model,
            prompt=text,
            max_tokens=1,
            temperature=0,
            echo=True,
            logprobs=1,
        )
        return _extract(response.choices[0])
