"""Text generation using Together API endpoint."""

import os
import math
from concurrent.futures import ThreadPoolExecutor
from together import Together
from dataclasses import dataclass, field

# Together dedicated endpoints get a fresh hash suffix each time they're
# recreated, so the model is read from $WEFT_MODEL when set. Update this
# fallback or export WEFT_MODEL to point at your current endpoint.
DEFAULT_MODEL = "sviteri/Qwen/Qwen3-30B-A3B-Base-018467e9"

# How many candidate tokens to request per position. Dedicated endpoints bill
# per GPU-time (not per token), so capturing the top-k alternatives alongside
# the sampled token is essentially free and powers the hover "candidates" bar.
TOP_K = 5


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


def perplexity(logprobs: dict | None) -> float | None:
    """Perplexity (exp of mean surprisal); lower = the model was more confident."""
    m = mean_logprob(logprobs)
    return math.exp(-m) if m is not None else None


_LN2 = math.log(2)


def total_surprisal_bits(logprobs: dict | None) -> float | None:
    """Total surprisal of a generation in bits: -sum(logprob)/ln2.

    This is the cumulative "how unlikely was this exact text" measure — the
    number of bits the model needed to commit to every token along the way.
    None entries (the first prompt token) are skipped. Returns None when no
    token carries a logprob.
    """
    if not logprobs:
        return None
    lps = [x for x in (logprobs.get("token_logprobs") or []) if x is not None]
    if not lps:
        return None
    return -sum(lps) / _LN2


def _extract(choice, params: dict | None = None) -> Generation:
    """Build a Generation from a Together completion choice.

    Captures the sampled tokens and their logprobs, plus the top-k candidate
    tokens per position (`top_logprobs`) when the endpoint returns them. Each
    top_logprobs entry is a {token: logprob} dict; we normalize to plain dicts
    so the whole structure round-trips through JSON unchanged. `params` (the
    sampling settings this text was produced under) is stored alongside so the
    UI can show what temperature a node was generated at — surprisal is only
    comparable across nodes sampled at the same temperature.
    """
    lp = getattr(choice, "logprobs", None)
    logprobs = None
    if lp is not None:
        tokens = getattr(lp, "tokens", None)
        token_logprobs = getattr(lp, "token_logprobs", None)
        top = getattr(lp, "top_logprobs", None)
        if tokens is not None or token_logprobs is not None:
            logprobs = {"tokens": tokens, "token_logprobs": token_logprobs}
            if top is not None:
                logprobs["top_logprobs"] = [dict(d) if d is not None else None for d in top]
    if logprobs is not None and params:
        logprobs["params"] = params
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
            logprobs=TOP_K,
        )
        params = {"temperature": temp, "top_p": self.config.top_p, "model": self.config.model}
        return [_extract(c, params) for c in response.choices]

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
            logprobs=TOP_K,
        )
        params = {"temperature": temp, "top_p": self.config.top_p, "model": self.config.model}
        return _extract(response.choices[0], params)

    def score(self, prefix: str, text: str,
              with_candidates: bool = False, candidate_cap: int = 120) -> Generation:
        """Score existing `text` (conditioned on `prefix`) without generating.

        Uses an `echo=True`, `max_tokens=1` pass so the endpoint returns
        per-token logprobs for the *prompt* itself, then slices off the prefix
        tokens so the result aligns exactly with `text`. This colors
        human-written or imported text the same way generated text is colored.

        The echo pass returns the chosen token's logprob but NOT the top-k
        alternatives (a vLLM limitation — only *generated* tokens get top_logprobs).
        So when `with_candidates` is set, we additionally ask the model to predict
        each position (see `candidates_for_tokens`) to recover the hover candidates
        bar for human-written text, up to `candidate_cap` tokens.

        Returns a Generation whose logprobs cover `text`, or logprobs=None when
        the tokenization can't be aligned to the prefix boundary.

        Scoring runs at temperature 1.0 so the logprobs are the model's true
        distribution — a clean, comparable surprisal measurement, independent of
        whatever temperature generated nodes were sampled at.
        """
        response = self.client.completions.create(
            model=self.config.model,
            prompt=prefix + text,
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=1.0,
        )
        pr = getattr(response, "prompt", None)
        if not pr:
            return Generation(text=text, logprobs=None)
        plp = pr[0].logprobs
        tokens = list(getattr(plp, "tokens", None) or [])
        tlps = list(getattr(plp, "token_logprobs", None) or [])
        logprobs = _slice_to_text(tokens, tlps, prefix, text)
        if logprobs is not None:
            logprobs["params"] = {"temperature": 1.0, "model": self.config.model, "scored": True}
            if with_candidates:
                tops = self.candidates_for_tokens(
                    prefix, logprobs["tokens"], logprobs["token_logprobs"], cap=candidate_cap)
                if any(t is not None for t in tops):
                    logprobs["top_logprobs"] = tops
        return Generation(text=text, logprobs=logprobs)

    def candidates_for_tokens(self, prefix, tokens, token_logprobs=None,
                              cap: int = 120, max_workers: int = 12):
        """Top-k next-token candidates at each position of `tokens`.

        The echo pass gives the chosen token's logprob but no alternatives, so to
        show "what else could go here" for human-written text we ask the model to
        predict each position: a 1-token generation whose prompt is everything up
        to that token. That generated token's `top_logprobs` IS the candidate
        distribution at that position. Positions are fetched concurrently.

        Returns a list aligned to `tokens`; each entry is a {token: logprob} dict
        (with the actual token merged in so the bar always shows what was written
        and can mark it chosen), or None where unavailable — no preceding context
        (the global root's first token), beyond `cap`, or on error.
        """
        n = len(tokens)
        limit = min(n, max(0, cap))

        def fetch(i):
            p = prefix + "".join(tokens[:i])
            if not p:                       # no context to condition on
                return i, None
            try:
                resp = self.client.completions.create(
                    model=self.config.model, prompt=p, max_tokens=1,
                    logprobs=TOP_K, temperature=1.0,
                )
                lp = getattr(resp.choices[0], "logprobs", None)
                top = getattr(lp, "top_logprobs", None) if lp is not None else None
                d = dict(top[0]) if top and top[0] is not None else {}
            except Exception:
                return i, None
            actual = tokens[i]
            if actual not in d:             # always include what was actually written
                la = token_logprobs[i] if token_logprobs and i < len(token_logprobs) else None
                if la is not None:
                    d[actual] = la
            return i, (d or None)

        out = [None] * n
        if limit:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i, d in ex.map(fetch, range(limit)):
                    out[i] = d
        return out


def _slice_to_text(tokens, token_logprobs, prefix, text, top_logprobs=None):
    """Keep only the (token, logprob) entries that make up `text`.

    The echo pass tokenizes prefix+text; we drop the leading tokens that spell
    out the prefix so the logprobs line up with `text`. `top_logprobs` (the
    per-position candidate dicts, when requested) is sliced the same way so the
    candidates bar lines up too. Returns None if the prefix boundary doesn't
    fall on a token boundary (so callers can skip rather
    than mis-color).
    """
    if len(tokens) != len(token_logprobs):
        return None
    have_top = top_logprobs is not None and len(top_logprobs) == len(tokens)
    acc = ""
    k = 0
    while k < len(tokens) and len(acc) < len(prefix):
        acc += tokens[k]
        k += 1
    if acc == prefix and "".join(tokens[k:]) == text:
        out = {"tokens": tokens[k:], "token_logprobs": token_logprobs[k:]}
        if have_top:
            out["top_logprobs"] = top_logprobs[k:]
        return out
    if "".join(tokens) == text:  # prefix empty (e.g. root)
        out = {"tokens": tokens, "token_logprobs": token_logprobs}
        if have_top:
            out["top_logprobs"] = top_logprobs
        return out
    return None
