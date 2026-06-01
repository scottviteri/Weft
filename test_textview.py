"""Tests for the interactive Current Text component builder.

build_text_component only needs a `loom`-shaped object (a tree + current_node),
so these tests drive the real api.Loom with a faked Generator and inspect the
HTML/JS it emits. The browser behavior itself isn't exercised — only that the
right hooks (colors, branch markers, candidates payload, plot links) are wired.
"""

import json
import re

import pytest

import api
from api import Loom
from generator import GenerationConfig, Generation


def _lp(text, top=None):
    """A one-token logprobs dict, optionally with top-k candidates."""
    d = {"tokens": [text], "token_logprobs": [-0.5]}
    if top is not None:
        d["top_logprobs"] = [top]
    return d


class FakeGenerator:
    def __init__(self, config=None):
        self.config = config or GenerationConfig()

    def generate_single(self, prompt, max_tokens=None, temperature=None):
        return Generation(" <cont>", logprobs=_lp(" <cont>"))

    def generate_continuations(self, prompt, num_branches=None, max_tokens=None, temperature=None):
        n = num_branches or self.config.num_branches
        return [Generation(f" branch{i}", logprobs=_lp(f" branch{i}")) for i in range(n)]


@pytest.fixture
def loom(monkeypatch):
    monkeypatch.setattr(api, "Generator", FakeGenerator)
    return Loom()


def _build(loom):
    from textview import build_text_component
    return build_text_component(loom)


# --- basic structure ----------------------------------------------------

def test_empty_root_has_no_logprobs(loom):
    html, has_lp = _build(loom)
    assert has_lp is False
    assert "<div class=\"text\">" in html


def test_current_tokens_are_colored_and_linked_to_plot(loom):
    loom.write("seed")
    loom.continue_branch()  # current node carries logprobs
    html, has_lp = _build(loom)
    assert has_lp is True
    # current-node token gets a plot point + matching circle id
    assert 'class="cur"' in html
    assert "<circle" in html
    # the first current token carries a split offset of 0
    assert 'data-split="0"' in html


def test_human_text_has_no_split_offsets_in_prefix(loom):
    # Root is human-written (no logprobs); after a generated child, the root
    # text is prefix and should not be marked current/splittable.
    loom.write("seed")
    loom.continue_branch()
    html, _ = _build(loom)
    # exactly the current node's tokens carry data-split; prefix tokens do not
    assert html.count('data-split="0"') == 1


# --- branch points ------------------------------------------------------

def test_branch_point_marked_when_siblings_exist(loom):
    loom.write("seed")
    loom.generate(n=3)
    loom.select_all()       # three sibling children under the seed
    loom.child(2)           # move onto the middle sibling
    html, _ = _build(loom)
    assert "branch" in html
    # the branch span lists all three sibling ids and the current position
    m = re.search(r'data-sibs="([^"]+)" data-pos="(\d+)"', html)
    assert m is not None
    assert len(m.group(1).split(",")) == 3
    assert m.group(2) == "1"   # middle sibling is index 1


def test_no_branch_marker_for_only_child(loom):
    loom.write("seed")
    loom.continue_branch()  # single child, no siblings
    html, _ = _build(loom)
    assert "data-sibs" not in html


def test_plot_extends_through_recent_branch_point(loom):
    # seed -> three siblings; descend one, then continue. The plot should cover
    # the fork region (a "fork" divider) plus tokens before the current node.
    loom.write("seed")
    loom.generate(n=3)
    loom.select_all()
    loom.child(2)            # onto a sibling (the branch point)
    loom.continue_branch()   # current node is now past the fork
    html, _ = _build(loom)
    assert "fork" in html            # divider drawn at the branch point
    # more plot circles than just the current node's single token
    assert html.count("<circle") >= 2


def test_plot_only_current_node_when_no_branch(loom):
    loom.write("seed")
    loom.continue_branch()   # linear path, no fork
    html, _ = _build(loom)
    assert "fork" not in html


# --- candidates payload -------------------------------------------------

def test_top_logprobs_become_candidates_payload(monkeypatch, loom):
    loom.write("seed")
    # craft a node whose logprobs include top-k candidates
    node = loom.tree.add_branch(loom.current_node.id, " X",
                                logprobs=_lp(" X", top={" X": -0.5, " Y": -1.5, " Z": None}))
    loom.current_node = node
    html, _ = _build(loom)
    payload = json.loads(re.search(r"var ALTS=(\{.*?\});", html).group(1))
    # one token, candidates sorted by probability, None dropped, chosen flagged
    rows = next(iter(payload.values()))
    toks = [r[0] for r in rows]
    assert toks == [" X", " Y"]
    assert rows[0][2] is True   # " X" is the sampled token
    assert all(len(r) == 3 for r in rows)


def test_candidates_absent_without_top_logprobs(loom):
    loom.write("seed")
    loom.continue_branch()  # logprobs but no top_logprobs
    html, _ = _build(loom)
    assert re.search(r"var ALTS=(\{.*?\});", html).group(1) == "{}"
