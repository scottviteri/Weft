"""Tests for the Tree / Node data structure."""

import json
from pathlib import Path

import pytest

from tree import Tree, Node


# --- Node ---------------------------------------------------------------

def test_node_ids_are_unique_and_short():
    a, b = Node(text="a"), Node(text="b")
    assert a.id != b.id
    assert len(a.id) == 8


def test_add_child_sets_parent():
    parent = Node(text="root")
    child = parent.add_child("hello")
    assert child in parent.children
    assert child.parent_id == parent.id
    assert child.text == "hello"


def test_node_dict_round_trip():
    root = Node(text="root")
    child = root.add_child("child", logprobs={"tokens": ["child"], "token_logprobs": [-0.3]})
    child.analysis = "some analysis"
    child.add_child("grandchild")

    restored = Node.from_dict(root.to_dict())

    assert restored.id == root.id
    assert restored.text == "root"
    assert restored.children[0].text == "child"
    assert restored.children[0].analysis == "some analysis"
    assert restored.children[0].logprobs == {"tokens": ["child"], "token_logprobs": [-0.3]}
    assert restored.children[0].children[0].text == "grandchild"
    assert restored.children[0].parent_id == root.id


# --- Tree ---------------------------------------------------------------

@pytest.fixture
def tree():
    return Tree(root_text="root")


def test_root_is_indexed(tree):
    assert tree.get_node(tree.root.id) is tree.root


def test_add_branch_indexes_child(tree):
    child = tree.add_branch(tree.root.id, "child")
    assert child is not None
    assert tree.get_node(child.id) is child
    assert child.parent_id == tree.root.id


def test_add_branch_to_missing_parent_returns_none(tree):
    assert tree.add_branch("nonexistent", "x") is None


def test_get_missing_node_returns_none(tree):
    assert tree.get_node("nope") is None


def test_path_and_full_text(tree):
    a = tree.add_branch(tree.root.id, " a")
    b = tree.add_branch(a.id, " b")
    path = tree.get_path_to_node(b.id)
    assert [n.id for n in path] == [tree.root.id, a.id, b.id]
    assert tree.get_full_text(b.id) == "root a b"


def test_path_only_follows_selected_branch(tree):
    a = tree.add_branch(tree.root.id, " a")
    tree.add_branch(tree.root.id, " sibling")
    assert tree.get_full_text(a.id) == "root a"


def test_save_load_round_trip(tree, tmp_path):
    a = tree.add_branch(tree.root.id, " a")
    tree.add_branch(a.id, " b")
    path = tmp_path / "t.json"
    tree.save(path)

    # File is valid JSON with the expected shape.
    assert json.loads(path.read_text())["text"] == "root"

    loaded = Tree.load(path)
    assert loaded.root.id == tree.root.id
    assert loaded.get_full_text(a.id) == "root a"
    # Loaded nodes are reachable via the rebuilt index.
    assert loaded.get_node(a.id) is loaded.root.children[0]
