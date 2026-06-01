"""Tests for the Loom programmatic API.

The real Generator needs a Together API key and makes network calls, so these
tests patch it with a fake that returns canned text. analyze() is exercised
only on its offline (error) paths.
"""

import pytest

import api
from api import Loom, analyze_continuation, model_label
from generator import GenerationConfig, Generation


def _lp(token):
    """A minimal logprobs dict shaped like Together's output."""
    return {"tokens": [token], "token_logprobs": [-0.5]}


class FakeGenerator:
    """Stand-in for Generator: records calls, returns deterministic Generations."""

    def __init__(self, config=None):
        self.config = config or GenerationConfig()
        self.calls = []

    def generate_single(self, prompt, max_tokens=None, temperature=None):
        self.calls.append(("single", prompt))
        return Generation(" <cont>", logprobs=_lp(" <cont>"))

    def generate_continuations(self, prompt, num_branches=None, max_tokens=None, temperature=None):
        n = num_branches or self.config.num_branches
        self.calls.append(("multi", prompt, n))
        return [Generation(f" branch{i}", logprobs=_lp(f" branch{i}")) for i in range(n)]

    def score(self, text):
        # Echo-style: one token per char so slicing reconstructs any substring.
        self.calls.append(("score", text))
        return Generation(text, logprobs={"tokens": list(text),
                                          "token_logprobs": [-0.5] * len(text)})


@pytest.fixture
def loom(monkeypatch):
    monkeypatch.setattr(api, "Generator", FakeGenerator)
    return Loom()


# --- construction -------------------------------------------------------

def test_init_with_prompt_seeds_root(monkeypatch):
    monkeypatch.setattr(api, "Generator", FakeGenerator)
    loom = Loom(prompt="hello root")
    assert loom.tree.root.text == "hello root"


def test_init_with_config_sets_generator(monkeypatch):
    monkeypatch.setattr(api, "Generator", FakeGenerator)
    loom = Loom(config=GenerationConfig(max_tokens=42, temperature=0.3))
    assert loom.generator.config.max_tokens == 42
    assert loom.generator.config.temperature == 0.3


# --- write / navigation -------------------------------------------------

def test_write_fills_empty_root(loom):
    loom.write("hello")
    assert loom.current_node is loom.tree.root
    assert loom.tree.root.text == "hello"
    assert loom.depth == 0


def test_write_after_root_creates_child(loom):
    loom.write("hello")
    loom.write(" world")
    assert loom.current_node.parent_id == loom.tree.root.id
    assert loom.full_text() == "hello world"
    assert loom.depth == 1


def test_up_root_and_child_navigation(loom):
    loom.write("a")
    loom.write("b")
    leaf = loom.current_node
    loom.up()
    assert loom.current_node is loom.tree.root
    loom.child(1)
    assert loom.current_node is leaf
    loom.root()
    assert loom.current_node is loom.tree.root


def test_up_at_root_reports_already_at_root(loom):
    loom.write("a")
    assert "Already at root" in loom.up()


def test_goto_known_and_unknown(loom):
    loom.write("a")
    loom.write("b")
    target = loom.current_node.id
    loom.root()
    loom.goto(target)
    assert loom.current_node.id == target
    assert "not found" in loom.goto("missing")


def test_child_out_of_range(loom):
    loom.write("a")
    assert "No children" in loom.child(1)
    loom.write("b")
    loom.up()
    assert "Invalid child" in loom.child(5)


# --- generate / select --------------------------------------------------

def test_continue_branch_advances(loom):
    loom.write("seed")
    loom.continue_branch()
    assert loom.current_node.text == " <cont>"
    assert loom.depth == 1


def test_continue_with_no_text_errors(loom):
    assert "Error" in loom.continue_branch()


def test_generate_then_select(loom):
    loom.write("seed")
    loom.generate(n=3)
    assert len(loom._last_branches) == 3
    loom.select(2)
    assert loom.current_node.text == " branch1"
    assert loom._last_branches == []  # cleared after select


def test_select_without_generate_errors(loom):
    loom.write("seed")
    assert "No branches" in loom.select(1)


def test_select_out_of_range_errors(loom):
    loom.write("seed")
    loom.generate(n=2)
    assert "Invalid branch" in loom.select(9)


def test_select_all_adds_children_without_moving(loom):
    loom.write("seed")
    loom.generate(n=3)
    before = loom.current_node
    loom.select_all()
    assert loom.current_node is before
    assert len(before.children) == 3
    assert loom._last_branches == []


def test_add_branches_subset(loom):
    loom.write("seed")
    loom.generate(n=3)
    before = loom.current_node
    loom.add_branches([1, 3])
    assert loom.current_node is before  # subset add does not move the cursor
    assert [c.text for c in before.children] == [" branch0", " branch2"]
    assert loom._last_branches == []


def test_add_branches_ignores_out_of_range(loom):
    loom.write("seed")
    loom.generate(n=2)
    loom.add_branches([1, 99])
    assert len(loom.current_node.children) == 1


# --- logprobs -----------------------------------------------------------

def test_continue_stores_logprobs(loom):
    loom.write("seed")
    loom.continue_branch()
    assert loom.current_node.logprobs == _lp(" <cont>")


def test_select_stores_logprobs(loom):
    loom.write("seed")
    loom.generate(n=2)
    loom.select(1)
    assert loom.current_node.logprobs == _lp(" branch0")


def test_generate_is_a_single_batched_call(loom):
    loom.write("seed")
    loom.generate(n=4)
    multi_calls = [c for c in loom.generator.calls if c[0] == "multi"]
    assert len(multi_calls) == 1
    assert multi_calls[0][2] == 4  # asked for 4 branches in one request


def test_recompute_logprobs_scores_each_node(loom):
    loom.write("seed")
    loom.write(" more")  # child node, prefix "seed"
    result = loom.recompute_logprobs()
    assert "2/2" in result
    child = loom.current_node
    assert "".join(child.logprobs["tokens"]) == " more"  # only the node's own text
    root = loom.tree.root
    assert "".join(root.logprobs["tokens"]) == "seed"


# --- split / trim -------------------------------------------------------

def test_trim_keeps_prefix(loom):
    loom.write("hello world")
    loom.trim(5)
    assert loom.current_node.text == "hello"


def test_split_keep_remainder_creates_child(loom):
    loom.write("hello world")
    node = loom.current_node
    loom.split(5, keep_remainder=True)
    assert node.text == "hello"
    assert node.children[0].text == " world"


def test_split_clears_stale_analysis_and_logprobs(loom):
    loom.write("hello world")
    loom.current_node.analysis = "stale"
    loom.current_node.logprobs = _lp("hello world")
    loom.trim(5)
    assert loom.current_node.analysis is None
    assert loom.current_node.logprobs is None


@pytest.mark.parametrize("index", [0, -1, 999])
def test_split_invalid_index_errors(loom, index):
    loom.write("hello")
    assert "Error" in loom.split(index)


# --- analyze (offline paths) -------------------------------------------

def test_analyze_at_root_errors(loom):
    loom.write("only root")
    assert "Error" in loom.analyze()  # no prefix


def test_analyze_continuation_validates_inputs():
    with pytest.raises(ValueError):
        analyze_continuation("prefix", "")
    with pytest.raises(ValueError):
        analyze_continuation("", "continuation")


@pytest.mark.parametrize("model, expected", [
    ("sviteri/Qwen/Qwen3-30B-A3B-Base-3737eb6e", "Qwen3-30B-A3B-Base"),
    ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base"),
    ("meta-llama/Llama-3-8B", "Llama-3-8B"),
    ("bare-model", "bare-model"),  # short suffix is not a deploy hash
])
def test_model_label(model, expected):
    assert model_label(model) == expected


# --- save / load --------------------------------------------------------

def test_save_and_load_round_trip(loom, tmp_path, monkeypatch):
    loom.write("seed")
    loom.continue_branch()  # adds a node carrying logprobs
    target_id = loom.current_node.id
    path = tmp_path / "saved.json"
    loom.save(str(path))

    monkeypatch.setattr(api, "Generator", FakeGenerator)
    fresh = Loom()
    fresh.load(str(path))
    node = fresh.tree.get_node(target_id)
    assert node is not None
    assert fresh.tree.get_full_text(target_id) == "seed <cont>"
    assert node.logprobs == _lp(" <cont>")  # logprobs survive the round-trip


def test_load_missing_file_errors(loom):
    assert "not found" in loom.load("definitely_missing_12345.json")


# --- views --------------------------------------------------------------

def test_tree_view_marks_current(loom):
    loom.write("seed")
    loom.continue_branch()
    view = loom.tree_view()
    assert ">>>" in view
    assert loom.current_node.id in view


def test_state_includes_prefix_and_current(loom):
    loom.write("seed")
    loom.write(" tail")
    state = loom.state()
    assert "PREFIX" in state
    assert "CURRENT" in state
