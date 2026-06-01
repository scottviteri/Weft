"""Tests for logprob coloring helpers and summary metrics."""

import math

import pytest

from coloring import (
    token_segments,
    surprisal_fraction,
    rgb_for_logprob,
    hex_for_logprob,
    DEFAULT_MAX_SURPRISAL,
)
from generator import mean_logprob, perplexity, slice_logprobs


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


# --- slice_logprobs -----------------------------------------------------

def test_slice_logprobs_extracts_node_region():
    lp = {"tokens": list("seed more"), "token_logprobs": [-0.5] * 9}
    out = slice_logprobs(lp, 4, 9)  # the " more" suffix
    assert "".join(out["tokens"]) == " more"
    assert len(out["token_logprobs"]) == 5


def test_slice_logprobs_none_when_region_empty():
    lp = {"tokens": list("ab"), "token_logprobs": [-0.1, -0.2]}
    assert slice_logprobs(lp, 5, 9) is None


def test_slice_logprobs_none_on_missing_or_mismatched():
    assert slice_logprobs(None, 0, 3) is None
    assert slice_logprobs({"tokens": ["a"], "token_logprobs": []}, 0, 1) is None
