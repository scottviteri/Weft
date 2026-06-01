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


def test_slice_carries_top_logprobs_through_prefix_drop():
    tokens = ["The", " machine", " began"]
    lps = [None, -2.0, -1.0]
    top = [None, {" machine": -2.0, " car": -3.0}, {" began": -1.0, " was": -2.0}]
    out = _slice_to_text(tokens, lps, "The machine", " began", top_logprobs=top)
    assert out == {
        "tokens": [" began"],
        "token_logprobs": [-1.0],
        "top_logprobs": [{" began": -1.0, " was": -2.0}],
    }


def test_slice_carries_top_logprobs_when_prefix_empty():
    tokens = ["hel", "lo"]
    lps = [None, -1.5]
    top = [None, {"lo": -1.5, "p": -2.0}]
    out = _slice_to_text(tokens, lps, "", "hello", top_logprobs=top)
    assert out["top_logprobs"] == top


def test_slice_omits_top_logprobs_on_length_mismatch():
    # Defensive: a malformed top list shorter than tokens is ignored, not crashed.
    tokens = ["a", "b"]
    lps = [None, -1.0]
    out = _slice_to_text(tokens, lps, "", "ab", top_logprobs=[None])
    assert "top_logprobs" not in out
