"""Tests for generation helpers that don't need the network.

_slice_to_text is the alignment step behind echo-based scoring: it drops the
prompt tokens belonging to the prefix so the logprobs line up with the node's
own text. The Together call itself is exercised manually, not here.
"""

from generator import _slice_to_text, _provider_for, Generator, GenerationConfig


class _Obj:
    def __init__(self, **kw): self.__dict__.update(kw)


class _EchoClient:
    """Fake completions client for echo passes. Shapes the response like OpenAI
    (logprobs on choices[0]) or Together (logprobs on a `prompt` field)."""
    def __init__(self, lp, shape):
        self.completions = self
        self._lp = lp
        self._shape = shape

    def create(self, **kw):
        if self._shape == "openai":
            return _Obj(choices=[_Obj(logprobs=self._lp)])
        return _Obj(prompt=[_Obj(logprobs=self._lp)])


def _gen(model, provider, client):
    g = Generator.__new__(Generator)
    g.config = GenerationConfig(model=model, provider=provider)
    g._client_cache = {g.provider: client}
    return g


def test_provider_for_routes_openai_base_models_else_together():
    assert _provider_for("davinci-002") == "openai"
    assert _provider_for("babbage-002") == "openai"
    assert _provider_for("sviteri/Qwen/Qwen3-30B-A3B-Base-x") == "together"
    assert _provider_for("davinci-002", "together") == "together"  # explicit wins


def test_echo_logprobs_openai_reads_choices_with_top():
    lp = _Obj(tokens=["A", "B"], token_logprobs=[None, -1.0],
              top_logprobs=[None, {"B": -1.0, "C": -2.0}])
    g = _gen("davinci-002", "", _EchoClient(lp, "openai"))
    toks, tlps, top = g._echo_logprobs("AB")
    assert toks == ["A", "B"] and tlps == [None, -1.0]
    assert top == [None, {"B": -1.0, "C": -2.0}]


def test_echo_logprobs_together_reads_prompt_without_top():
    lp = _Obj(tokens=["A", "B"], token_logprobs=[None, -1.0], top_logprobs=None)
    g = _gen("sviteri/Qwen/x", "together", _EchoClient(lp, "together"))
    toks, tlps, top = g._echo_logprobs("AB")
    assert toks == ["A", "B"] and top is None


class _FakeClient:
    """Stands in for the Together client: returns canned top_logprobs per prompt.

    The completions API is reached as `client.completions.create(...)`, so this
    object serves as both the client and its `.completions` attribute.
    """
    def __init__(self, by_prompt):
        self.completions = self
        self.by_prompt = by_prompt

    def create(self, model, prompt, max_tokens, logprobs, temperature):
        top = self.by_prompt.get(prompt)

        class _LP:
            top_logprobs = [top]

        class _Choice:
            logprobs = _LP()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _gen_with_client(by_prompt):
    g = Generator.__new__(Generator)            # skip __init__ (no network/key)
    g.config = GenerationConfig(model="m")
    g._client_cache = {g.provider: _FakeClient(by_prompt)}
    return g


def test_candidates_for_tokens_aligns_and_merges_actual():
    g = _gen_with_client({"A": {"B": -1.0}, "AB": {"X": -2.0}})
    out = g.candidates_for_tokens("", ["A", "B", "C"], token_logprobs=[None, -1.0, -0.5])
    assert out[0] is None                     # no context -> skipped
    assert out[1] == {"B": -1.0}              # actual token already a candidate
    assert out[2] == {"X": -2.0, "C": -0.5}   # actual merged in from its echo logprob


def test_candidates_for_tokens_respects_cap():
    g = _gen_with_client({"pre": {"Z": -1.0}})
    out = g.candidates_for_tokens("pre", ["A", "B", "C"], cap=1)
    assert out[0] == {"Z": -1.0}
    assert out[1] is None and out[2] is None  # beyond the cap


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
