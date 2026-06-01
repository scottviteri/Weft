"""Tests for logprob coloring helpers and summary metrics."""

import math

import pytest

from coloring import (
    token_segments,
    surprisal_fraction,
    rgb_for_logprob,
    hex_for_logprob,
    token_title,
    color_bar_html,
    alt_bar_items,
    logprob_plot_svg,
    DEFAULT_MAX_SURPRISAL,
)
from generator import mean_logprob, perplexity


# --- token_segments -----------------------------------------------------

def test_token_segments_splits_when_reconstructable():
    lp = {"tokens": ["he", "llo"], "token_logprobs": [-0.1, -2.0]}
    assert token_segments("hello", lp) == [("he", -0.1), ("llo", -2.0)]


def test_token_segments_fallback_when_no_logprobs():
    assert token_segments("hello", None) == [("hello", None)]


def test_token_segments_fallback_on_text_mismatch():
    # Text was edited after generation; tokens no longer reconstruct it.
    lp = {"tokens": ["he", "llo"], "token_logprobs": [-0.1, -2.0]}
    assert token_segments("HELLO", lp) == [("HELLO", None)]


def test_token_segments_fallback_on_length_mismatch():
    lp = {"tokens": ["he", "llo"], "token_logprobs": [-0.1]}
    assert token_segments("hello", lp) == [("hello", None)]


# --- colour ramp --------------------------------------------------------

def test_surprisal_fraction_bounds():
    assert surprisal_fraction(0.0) == 0.0           # certain token
    assert surprisal_fraction(-100.0) == 1.0        # clamped at the top
    assert 0.0 < surprisal_fraction(-DEFAULT_MAX_SURPRISAL / 2) < 1.0


def test_rgb_endpoints():
    assert rgb_for_logprob(0.0) == (128, 128, 128)              # confident grey
    assert rgb_for_logprob(-DEFAULT_MAX_SURPRISAL) == (255, 60, 60)  # surprising red


def test_hex_format():
    assert hex_for_logprob(0.0) == "#808080"
    assert hex_for_logprob(-DEFAULT_MAX_SURPRISAL) == "#ff3c3c"


def test_token_title_shows_logprob_and_probability():
    assert token_title(0.0) == "logprob 0.00 · p=100.0%"
    assert "logprob -2.00" in token_title(-2.0)


def test_color_bar_spans_the_ramp():
    bar = color_bar_html()
    assert "linear-gradient" in bar
    assert "#808080" in bar  # confident endpoint
    assert "#ff3c3c" in bar  # surprising endpoint


# --- alt_bar_items ------------------------------------------------------

def test_alt_bar_items_sorted_desc_and_skips_none():
    out = alt_bar_items({" a": -1.0, " b": -0.1, " c": None})
    assert [t for t, _ in out] == [" b", " a"]  # most probable first, None dropped
    assert out[0][1] == pytest.approx(math.exp(-0.1))


def test_alt_bar_items_empty_inputs():
    assert alt_bar_items(None) == []
    assert alt_bar_items({}) == []


def test_alt_bar_items_respects_limit():
    out = alt_bar_items({"a": -1.0, "b": -2.0, "c": -3.0}, limit=2)
    assert len(out) == 2
    assert [t for t, _ in out] == ["a", "b"]


# --- logprob_plot_svg ---------------------------------------------------

def test_logprob_plot_svg_has_a_point_per_token():
    # points are (global_index, logprob) pairs in reading order
    svg = logprob_plot_svg([(0, -0.1), (1, -2.0), (2, -5.0)])
    assert svg.count("<circle") == 3
    assert "polyline" in svg
    assert 'data-idx="2"' in svg


def test_logprob_plot_svg_uses_supplied_indices():
    # The span index is global (prefix + current), so it need not start at 0.
    svg = logprob_plot_svg([(7, -0.1), (8, -2.0)])
    assert 'id="p7"' in svg and 'id="p8"' in svg


def test_logprob_plot_svg_empty_without_logprobs():
    assert logprob_plot_svg([(0, None)]) == ""


def test_logprob_plot_svg_marks_the_fork():
    svg = logprob_plot_svg([(5, -0.1), (6, -2.0), (7, -1.0)], mark_idx=6)
    assert "fork" in svg
    assert "stroke-dasharray" in svg


def test_logprob_plot_svg_no_mark_when_idx_absent():
    svg = logprob_plot_svg([(5, -0.1), (6, -2.0)], mark_idx=99)
    assert "fork" not in svg


# --- metrics ------------------------------------------------------------

def test_mean_logprob_and_perplexity():
    lp = {"tokens": ["a", "b"], "token_logprobs": [-1.0, -1.0]}
    assert mean_logprob(lp) == -1.0
    assert perplexity(lp) == pytest.approx(math.e)


def test_metrics_none_when_missing():
    assert mean_logprob(None) is None
    assert perplexity(None) is None
    assert mean_logprob({"tokens": [], "token_logprobs": []}) is None


def test_mean_logprob_skips_none_entries():
    # The first prompt token has no context, so its logprob is None.
    lp = {"tokens": ["a", "b"], "token_logprobs": [None, -2.0]}
    assert mean_logprob(lp) == -2.0
