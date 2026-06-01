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

    def score(self, prefix, text):
        self.calls.append(("score", prefix, text))
        return Generation(text, logprobs={"tokens": [text], "token_logprobs": [-0.7]})


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


# --- siblings (shared by GUI branch points + TUI cycling) ---------------

def test_siblings_empty_at_root(loom):
    loom.write("seed")
    assert loom.siblings() == []  # root has no parent


def test_siblings_lists_all_children_of_parent(loom):
    loom.write("seed")
    loom.generate(n=3)
    loom.select_all()
    loom.child(1)
    sibs = loom.siblings()
    assert len(sibs) == 3
    assert loom.current_node in sibs


def test_select_sibling_cycles_with_wraparound(loom):
    loom.write("seed")
    loom.generate(n=3)
    loom.select_all()
    loom.child(1)                       # on first of three siblings
    sibs = loom.siblings()
    assert loom.current_node is sibs[0]
    loom.select_sibling(1)
    assert loom.current_node is sibs[1]
    loom.select_sibling(-1)             # back to the first
    assert loom.current_node is sibs[0]
    loom.select_sibling(-1)             # wrap to the last
    assert loom.current_node is sibs[-1]


def test_select_sibling_noop_without_siblings(loom):
    loom.write("seed")
    loom.continue_branch()              # single child, no siblings
    assert "No siblings" in loom.select_sibling(1)


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


def test_split_and_branch_opens_empty_sibling(loom):
    loom.write("hello world")
    node = loom.current_node
    loom.split_and_branch(5)
    # original node keeps the prefix; the remainder and a fresh empty branch
    # are now siblings under it, and the cursor sits on the empty one.
    assert node.text == "hello"
    assert [c.text for c in node.children] == [" world", ""]
    assert loom.current_node is node.children[1]
    assert loom.current_node.text == ""


def test_split_and_branch_invalid_index_errors(loom):
    loom.write("hello")
    before = loom.current_node
    assert "Error" in loom.split_and_branch(0)
    assert before.children == []  # nothing created on failure


# --- score (echo-based coloring of existing text) ----------------------

def test_score_node_sets_logprobs(loom):
    loom.write("hello")            # human-written, no logprobs
    assert loom.current_node.logprobs is None
    result = loom.score_node()
    assert loom.current_node.logprobs is not None
    assert "Scored" in result


def test_score_node_empty_is_noop(loom):
    assert "Nothing" in loom.score_node()  # empty root


def test_score_node_conditions_on_prefix(loom):
    loom.write("seed")
    loom.write(" tail")            # child node
    loom.score_node()
    # the echo prefix passed to the generator is the path before this node
    score_calls = [c for c in loom.generator.calls if c[0] == "score"]
    assert score_calls[-1][1] == "seed"      # prefix
    assert score_calls[-1][2] == " tail"     # text


def test_score_tree_fills_unscored_nodes(loom):
    loom.write("seed")
    loom.continue_branch()         # generated child already has logprobs
    generated = loom.current_node
    result = loom.score_tree()
    assert loom.tree.root.logprobs is not None       # root got scored
    assert "Scored" in result


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
