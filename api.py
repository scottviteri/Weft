"""Programmatic API for Loom - usable by Claude or scripts."""

from pathlib import Path
from datetime import datetime
from tree import Tree, Node
from generator import Generator, GenerationConfig

TREES_DIR = Path(__file__).parent / "trees"


class Loom:
    """Non-interactive API for tree-based text exploration."""

    def __init__(self, model: str | None = None, load_path: Path | None = None):
        """Initialize Loom with optional model override or existing tree."""
        config = GenerationConfig()
        if model:
            config.model = model
        self.generator = Generator(config)

        if load_path and load_path.exists():
            self.tree = Tree.load(load_path)
        else:
            self.tree = Tree()

        self.current_node = self.tree.root
        self._last_branches: list[str] = []

    def write(self, text: str) -> str:
        """Write text at current position."""
        if not self.current_node.text and self.current_node == self.tree.root:
            self.current_node.text = text
        else:
            new_node = self.tree.add_branch(self.current_node.id, text)
            self.current_node = new_node
        return f"Written. Now at node {self.current_node.id}, depth {self.depth}"

    def continue_branch(self, max_tokens: int | None = None) -> str:
        """Generate a single continuation and follow it."""
        full_text = self.tree.get_full_text(self.current_node.id)
        if not full_text:
            return "Error: No text to continue from. Write something first."

        text = self.generator.generate_single(full_text, max_tokens=max_tokens)
        new_node = self.tree.add_branch(self.current_node.id, text)
        self.current_node = new_node
        return f"Generated:\n{text}\n\nNow at node {self.current_node.id}, depth {self.depth}"

    def generate(self, n: int = 3, max_tokens: int | None = None) -> str:
        """Generate multiple branches and store them for selection."""
        full_text = self.tree.get_full_text(self.current_node.id)
        if not full_text:
            return "Error: No text to continue from. Write something first."

        self._last_branches = self.generator.generate_continuations(
            full_text, num_branches=n, max_tokens=max_tokens
        )

        result = f"Generated {n} branches:\n\n"
        for i, text in enumerate(self._last_branches):
            preview = text[:200].replace('\n', ' ')
            if len(text) > 200:
                preview += "..."
            result += f"[{i + 1}] {preview}\n\n"
        result += "Use select(n) to choose one, or select_all() to add all."
        return result

    def select(self, branch_num: int) -> str:
        """Select a generated branch by number (1-indexed)."""
        idx = branch_num - 1
        if not self._last_branches:
            return "Error: No branches to select. Run generate() first."
        if idx < 0 or idx >= len(self._last_branches):
            return f"Error: Invalid branch number. Choose 1-{len(self._last_branches)}."

        text = self._last_branches[idx]
        new_node = self.tree.add_branch(self.current_node.id, text)
        self.current_node = new_node
        self._last_branches = []
        return f"Selected branch {branch_num}. Now at node {self.current_node.id}, depth {self.depth}"

    def select_all(self) -> str:
        """Add all generated branches as children."""
        if not self._last_branches:
            return "Error: No branches to select. Run generate() first."

        for text in self._last_branches:
            self.tree.add_branch(self.current_node.id, text)

        count = len(self._last_branches)
        self._last_branches = []
        return f"Added {count} branches as children of node {self.current_node.id}"

    def split(self, char_index: int, keep_remainder: bool = False) -> str:
        """Split current node's text at character index.

        Trims text to first char_index characters. If keep_remainder=True,
        creates a child node with the remainder. Otherwise discards it.
        """
        text = self.current_node.text
        if not text:
            return "Error: No text to split."
        if char_index <= 0:
            return "Error: char_index must be positive."
        if char_index >= len(text):
            return f"Error: char_index {char_index} >= text length {len(text)}."

        before = text[:char_index]
        after = text[char_index:]

        self.current_node.text = before
        self.current_node.analysis = None  # Clear stale analysis

        if keep_remainder and after.strip():
            child = self.tree.add_branch(self.current_node.id, after)
            return f"Split at {char_index}. Kept {len(before)} chars, remainder ({len(after)} chars) moved to child {child.id}."
        else:
            return f"Split at {char_index}. Kept {len(before)} chars, discarded {len(after)} chars. Ready to regenerate."

    def trim(self, keep_chars: int) -> str:
        """Shorthand: keep only the first N characters of current node."""
        return self.split(keep_chars, keep_remainder=False)

    def up(self) -> str:
        """Go to parent node."""
        if self.current_node.parent_id:
            parent = self.tree.get_node(self.current_node.parent_id)
            if parent:
                self.current_node = parent
                return f"Moved to parent node {self.current_node.id}, depth {self.depth}"
        return "Already at root."

    def root(self) -> str:
        """Go to root node."""
        self.current_node = self.tree.root
        return f"At root node {self.current_node.id}"

    def goto(self, node_id: str) -> str:
        """Go to a specific node by ID."""
        node = self.tree.get_node(node_id)
        if node:
            self.current_node = node
            return f"Moved to node {node_id}, depth {self.depth}"
        return f"Error: Node {node_id} not found."

    def child(self, n: int) -> str:
        """Go to nth child (1-indexed)."""
        idx = n - 1
        if idx < 0 or idx >= len(self.current_node.children):
            if not self.current_node.children:
                return "No children at this node."
            return f"Error: Invalid child number. Choose 1-{len(self.current_node.children)}."
        self.current_node = self.current_node.children[idx]
        return f"Moved to child {n}, node {self.current_node.id}, depth {self.depth}"

    def analyze(self) -> str:
        """Analyze current node with Claude."""
        import anthropic

        path = self.tree.get_path_to_node(self.current_node.id)
        prefix_nodes = path[:-1]
        prefix_text = "".join(n.text for n in prefix_nodes)
        continuation_text = self.current_node.text

        if not continuation_text:
            return "Error: No text to analyze at this node."
        if not prefix_text:
            return "Error: No prefix text - this is the root node."

        prompt = f"""Analyze this text generation. A language model (Qwen) was given a prefix and generated a continuation.

PREFIX:
{prefix_text}

CONTINUATION (generated by Qwen):
{continuation_text}

Provide a brief analysis (2-3 sentences each):
1. **Prefix**: What is happening/being discussed in the prefix?
2. **Continuation**: What does the continuation add or where does it take the text?
3. **Qwen's interpretation**: Based on how the continuation follows the prefix, what does Qwen seem to think the text is about or where it should go?"""

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = response.content[0].text
        self.current_node.analysis = analysis
        return analysis

    def state(self) -> str:
        """Get current state as readable text."""
        path = self.tree.get_path_to_node(self.current_node.id)
        prefix_text = "".join(n.text for n in path[:-1])
        current_text = self.current_node.text or ""

        result = f"=== LOOM STATE ===\n"
        result += f"Node: {self.current_node.id} | Depth: {self.depth} | Children: {len(self.current_node.children)}\n\n"

        if prefix_text:
            result += f"--- PREFIX ---\n{prefix_text}\n\n"
        if current_text:
            result += f"--- CURRENT (this node) ---\n{current_text}\n\n"

        if self.current_node.analysis:
            result += f"--- ANALYSIS ---\n{self.current_node.analysis}\n\n"

        if self.current_node.children:
            result += f"--- CHILDREN ---\n"
            for i, child in enumerate(self.current_node.children):
                preview = child.text[:80].replace('\n', ' ')
                if len(child.text) > 80:
                    preview += "..."
                result += f"[{i + 1}] {child.id}: {preview}\n"

        return result

    def full_text(self) -> str:
        """Get the full text from root to current node."""
        return self.tree.get_full_text(self.current_node.id)

    def save(self, filepath: str | None = None) -> str:
        """Save tree to file. Uses trees/ folder with datetime name by default."""
        if filepath is None:
            TREES_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = TREES_DIR / f"loom_{timestamp}.json"
        else:
            filepath = Path(filepath)
        self.tree.save(filepath)
        return f"Saved to {filepath}"

    def load(self, filepath: str) -> str:
        """Load tree from file. Checks trees/ folder if not found directly."""
        path = Path(filepath)
        if not path.exists():
            # Try in trees folder
            trees_path = TREES_DIR / filepath
            if trees_path.exists():
                path = trees_path
            else:
                return f"Error: File {filepath} not found."
        self.tree = Tree.load(path)
        self.current_node = self.tree.root
        self.tree._build_index(self.tree.root)
        return f"Loaded from {path}. At root node."

    def list_trees(self) -> str:
        """List saved trees in the trees folder."""
        if not TREES_DIR.exists():
            return "No trees folder found."
        files = sorted(TREES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return "No saved trees found."
        result = "=== SAVED TREES ===\n"
        for f in files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            result += f"{f.name} ({mtime})\n"
        return result

    @property
    def depth(self) -> int:
        """Current depth in tree."""
        return len(self.tree.get_path_to_node(self.current_node.id)) - 1

    def tree_view(self) -> str:
        """Get a text representation of the tree structure."""
        def _render(node: Node, indent: int = 0) -> str:
            marker = ">>> " if node.id == self.current_node.id else ""
            preview = node.text[:50].replace('\n', ' ') if node.text else "[empty]"
            if node.text and len(node.text) > 50:
                preview += "..."
            result = "  " * indent + f"{marker}[{node.id}] {preview}\n"
            for child in node.children:
                result += _render(child, indent + 1)
            return result

        return f"=== TREE ===\n{_render(self.tree.root)}"
