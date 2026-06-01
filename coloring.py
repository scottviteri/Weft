"""Map per-token logprobs to colors for surprisal visualization.

Tokens the model was confident about render dim grey; surprising tokens (low
probability, where the model "took a turn") render bright red. This is the
qualitative dual of the `analyze` readout: it shows *where* along the path the
model committed to something non-obvious.
"""

# Surprisal in nats at which a token is treated as maximally surprising.
# ~6 nats ≈ a token the model gave < 0.25% probability.
DEFAULT_MAX_SURPRISAL = 6.0

# Endpoints of the colour ramp: confident -> surprising.
_CONFIDENT_RGB = (128, 128, 128)
_SURPRISING_RGB = (255, 60, 60)

LEGEND = "dim grey = expected · bright red = surprising"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def surprisal_fraction(logprob: float, max_surprisal: float = DEFAULT_MAX_SURPRISAL) -> float:
    """0.0 (confident) .. 1.0 (surprising) for a single token logprob."""
    return _clamp((-logprob) / max_surprisal, 0.0, 1.0)


def rgb_for_logprob(logprob: float, max_surprisal: float = DEFAULT_MAX_SURPRISAL) -> tuple[int, int, int]:
    """Interpolate the confident->surprising ramp for a token logprob."""
    f = surprisal_fraction(logprob, max_surprisal)
    return tuple(
        int(c0 + f * (c1 - c0))
        for c0, c1 in zip(_CONFIDENT_RGB, _SURPRISING_RGB)
    )


def hex_for_logprob(logprob: float, max_surprisal: float = DEFAULT_MAX_SURPRISAL) -> str:
    """'#rrggbb' for a token logprob (works for both Rich styles and HTML)."""
    r, g, b = rgb_for_logprob(logprob, max_surprisal)
    return f"#{r:02x}{g:02x}{b:02x}"


def token_segments(text: str, logprobs: dict | None) -> list[tuple[str, float | None]]:
    """Split text into (token, logprob) segments for per-token coloring.

    Falls back to a single (text, None) segment when logprobs are missing,
    malformed, or don't reconstruct the text exactly (e.g. after a manual
    edit) — so callers can always render, just without colour.
    """
    if not logprobs:
        return [(text, None)]
    tokens = logprobs.get("tokens")
    lps = logprobs.get("token_logprobs")
    if not tokens or not lps or len(tokens) != len(lps):
        return [(text, None)]
    if "".join(tokens) != text:
        return [(text, None)]
    return list(zip(tokens, lps))
