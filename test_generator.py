"""Tests for generation helpers that don't need the network.

_slice_to_text is the alignment step behind echo-based scoring: it drops the
prompt tokens belonging to the prefix so the logprobs line up with the node's
own text. The Together call itself is exercised manually, not here.
"""

from generator import _slice_to_text


def test_slice_drops_prefix_tokens():
    tokens = ["The", " machine", " began"]
    lps = [None, -2.0, -1.0]
    out = _slice_to_text(tokens, lps, "The machine", " began")
    assert out == {"tokens": [" began"], "token_logprobs": [-1.0]}


def test_slice_whole_text_when_prefix_empty():
    tokens = ["hel", "lo"]
    lps = [None, -1.5]
    out = _slice_to_text(tokens, lps, "", "hello")
    assert out == {"tokens": ["hel", "lo"], "token_logprobs": [None, -1.5]}


def test_slice_returns_none_on_misaligned_boundary():
    # The prefix boundary falls inside a token (" machinebegan"), so it can't
    # be split cleanly -> None, so the caller skips rather than mis-color.
    tokens = ["The", " machinebegan"]
    lps = [None, -2.0]
    assert _slice_to_text(tokens, lps, "The machine", "began") is None


def test_slice_returns_none_on_length_mismatch():
    assert _slice_to_text(["a", "b"], [-1.0], "", "ab") is None
