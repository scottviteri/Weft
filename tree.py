"""Tree data structure for branching text exploration."""

from dataclasses import dataclass, field
from typing import Optional
import uuid
import json
from pathlib import Path


@dataclass
class Node:
    """A node in the text tree."""
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    children: list["Node"] = field(default_factory=list)
    parent_id: Optional[str] = None
    analysis: Optional[str] = None  # Claude's analysis of this node

    def add_child(self, text: str) -> "Node":
        """Add a child node with the given text."""
        child = Node(text=text, parent_id=self.id)
        self.children.append(child)
        return child

    def to_dict(self) -> dict:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "parent_id": self.parent_id,
            "analysis": self.analysis,
            "children": [c.to_dict() for c in self.children]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Deserialize node from dictionary."""
        node = cls(
            text=data["text"],
            id=data["id"],
            parent_id=data.get("parent_id"),
            analysis=data.get("analysis")
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node


class Tree:
    """A tree of branching text."""

    def __init__(self, root_text: str = ""):
        self.root = Node(text=root_text)
        self._node_index: dict[str, Node] = {self.root.id: self.root}
        self._build_index(self.root)

    def _build_index(self, node: Node):
        """Build index of all nodes."""
        self._node_index[node.id] = node
        for child in node.children:
            self._build_index(child)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self._node_index.get(node_id)

    def add_branch(self, parent_id: str, text: str) -> Optional[Node]:
        """Add a new branch to a node."""
        parent = self.get_node(parent_id)
        if parent is None:
            return None
        child = parent.add_child(text)
        self._node_index[child.id] = child
        return child

    def get_path_to_node(self, node_id: str) -> list[Node]:
        """Get the path from root to a node."""
        path = []
        node = self.get_node(node_id)
        while node:
            path.append(node)
            node = self.get_node(node.parent_id) if node.parent_id else None
        return list(reversed(path))

    def get_full_text(self, node_id: str) -> str:
        """Get the concatenated text from root to a node."""
        path = self.get_path_to_node(node_id)
        return "".join(n.text for n in path)

    def save(self, filepath: Path):
        """Save tree to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.root.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "Tree":
        """Load tree from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        tree = cls.__new__(cls)
        tree.root = Node.from_dict(data)
        tree._node_index = {}
        tree._build_index(tree.root)
        return tree
