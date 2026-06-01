"""Map per-token logprobs to colors for surprisal visualization.

Tokens the model was confident about render dim grey; surprising tokens (low
probability, where the model "took a turn") render bright red. This is the
qualitative dual of the `analyze` readout: it shows *where* along the path the
model committed to something non-obvious.
"""

import html as _html
import math

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


_LN2 = math.log(2)


def surprisal_bits(logprob: float) -> float:
    """Surprisal of a single token in bits: -logprob/ln2."""
    return -logprob / _LN2


def token_title(logprob: float) -> str:
    """Hover-tooltip text for a token: exact logprob, probability, surprisal."""
    bits = surprisal_bits(logprob) + 0.0  # normalize -0.0 -> 0.0 for display
    return f"logprob {logprob:.2f} · p={math.exp(logprob):.1%} · {bits:.1f} bits"


def gen_params_label(logprobs: dict | None) -> str:
    """Short human label for how a node's text was produced, or '' if unknown.

    Reads the `params` recorded on the node's logprobs (temperature, and whether
    it was echo-scored rather than sampled). Surfacing this makes clear that
    surprisal is only comparable across nodes produced at the same temperature.
    """
    if not logprobs:
        return ""
    p = logprobs.get("params")
    if not p:
        return ""
    if p.get("scored"):
        return f"scored · T={p.get('temperature', 1.0):g}"
    bits = []
    if "temperature" in p:
        bits.append(f"T={p['temperature']:g}")
    if "top_p" in p:
        bits.append(f"top_p={p['top_p']:g}")
    return " · ".join(bits)


def color_bar_html(width_px: int = 160) -> str:
    """A horizontal gradient legend matching the surprisal ramp (for the GUI)."""
    stops = ", ".join(hex_for_logprob(-(i / 4) * DEFAULT_MAX_SURPRISAL) for i in range(5))
    return (
        '<div style="display:flex; align-items:center; gap:8px; '
        'font-size:0.8em; color:#888; margin-top:4px;">'
        '<span>expected</span>'
        f'<div style="flex:0 0 {width_px}px; height:12px; border-radius:3px; '
        f'background:linear-gradient(to right, {stops});"></div>'
        '<span>surprising</span></div>'
    )


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


def alt_bar_items(top_at_pos: dict | None, limit: int | None = None):
    """Candidate next-tokens at one position, as (token, probability) pairs.

    Takes one entry of the endpoint's `top_logprobs` (a {token: logprob} dict),
    drops None logprobs, and returns the candidates sorted most-probable first.
    Powers the hover "candidates" bar; returns [] when nothing is available.
    """
    if not top_at_pos:
        return []
    items = [(tok, math.exp(lp)) for tok, lp in top_at_pos.items() if lp is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:limit] if limit else items


def logprob_plot_svg(points, width: int = 340, height: int = 170,
                     max_surprisal: float = DEFAULT_MAX_SURPRISAL,
                     mark_idx: int | None = None) -> str:
    """An SVG line plot of token logprob vs. token index.

    `points` is an iterable of (id_index, logprob) pairs in reading order;
    entries with logprob None are skipped. The y-axis runs from 0 (top,
    confident) down to ``-max_surprisal`` (bottom, surprising); each point is
    colored by the same ramp as the text and carries ``id=f"p{id_index}"`` /
    ``data-idx`` so the matching token span can cross-highlight it. The
    id_index need not be contiguous (the spans use a single global index across
    prefix + current text). When ``mark_idx`` matches a plotted point, a dashed
    "fork" divider is drawn there. Returns "" when no point carries a logprob.
    """
    pts = [(idx, lp) for idx, lp in points if lp is not None]
    if not pts:
        return ""
    left, top, right, bottom = 30, 12, 12, 24
    pw = max(1, width - left - right)
    ph = max(1, height - top - bottom)
    m = len(pts)

    def x(k):
        return left + (pw * k / (m - 1) if m > 1 else pw / 2)

    def y(lp):
        return top + ph * surprisal_fraction(lp, max_surprisal)

    coords = [(x(k), y(lp), idx, lp) for k, (idx, lp) in enumerate(pts)]
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py, _, _ in coords)
    circles = "".join(
        f'<circle class="pt" id="p{idx}" data-idx="{idx}" cx="{px:.1f}" cy="{py:.1f}" '
        f'r="3" fill="{hex_for_logprob(lp, max_surprisal)}" stroke="#fff" '
        f'stroke-width="0"><title>{_html.escape(token_title(lp))}</title></circle>'
        for px, py, idx, lp in coords
    )
    mark = ""
    if mark_idx is not None:
        mx = next((px for px, _, idx, _ in coords if idx == mark_idx), None)
        if mx is not None:
            mark = (
                f'<line x1="{mx:.1f}" y1="{top}" x2="{mx:.1f}" y2="{top + ph}" '
                f'stroke="#6aa3ff" stroke-width="1" stroke-dasharray="3,2" opacity="0.85"/>'
                f'<text x="{mx:.1f}" y="{top - 2}" font-size="8" fill="#6aa3ff" text-anchor="middle">fork</text>'
            )
    axis = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="#bbb"/>'
        f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="#bbb"/>'
        f'<text x="{left - 4}" y="{top + 4}" font-size="9" fill="#888" text-anchor="end">0</text>'
        f'<text x="{left - 4}" y="{top + ph}" font-size="9" fill="#888" text-anchor="end">-{max_surprisal:g}</text>'
        f'<text x="{left + pw / 2:.0f}" y="{top + ph + 18}" font-size="9" fill="#888" text-anchor="middle">token index</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px; height:auto; overflow:visible;">'
        f'{axis}'
        f'{mark}'
        f'<polyline points="{poly}" fill="none" stroke="#888" stroke-width="1" opacity="0.5"/>'
        f'{circles}</svg>'
    )
