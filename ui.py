"""Terminal UI for navigating the text tree."""

import sys
import importlib
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, IntPrompt
from rich.tree import Tree as RichTree

from tree import Tree, Node
from generator import Generator, GenerationConfig


class LoomUI:
    """Interactive terminal UI for exploring branching text."""

    def __init__(self, tree: Tree | None = None, generator: Generator | None = None):
        self.console = Console()
        self.tree = tree or Tree()
        self.generator = generator or Generator()
        self.current_node = self.tree.root

    def display_current_state(self):
        """Display the current text and available branches."""
        self.console.clear()

        # Show full text path with current node highlighted
        path = self.tree.get_path_to_node(self.current_node.id)
        if path:
            # Build styled text: prefix in normal, current node in cyan
            styled_text = Text()
            prefix_nodes = path[:-1]
            current_node = path[-1]

            prefix_text = "".join(n.text for n in prefix_nodes)
            if prefix_text:
                styled_text.append(prefix_text, style="dim")
            if current_node.text:
                styled_text.append(current_node.text, style="bold cyan")

            if styled_text:
                self.console.print(Panel(
                    styled_text,
                    title="[bold blue]Current Text[/bold blue] [dim](latest in cyan)[/dim]",
                    border_style="blue"
                ))
            else:
                self.console.print(Panel(
                    "[dim]Empty - start by writing or generating text[/dim]",
                    title="[bold blue]Current Text[/bold blue]",
                    border_style="blue"
                ))

            # Show analysis if present
            if self.current_node.analysis:
                self.console.print(Panel(
                    self.current_node.analysis,
                    title="[bold yellow]Analysis[/bold yellow]",
                    border_style="yellow"
                ))

        # Show current node info
        depth = len(path) - 1
        self.console.print(f"\n[dim]Depth: {depth} | Node: {self.current_node.id} | Children: {len(self.current_node.children)}[/dim]")

        # Show children if any
        if self.current_node.children:
            self.console.print("\n[bold green]Branches:[/bold green]")
            for i, child in enumerate(self.current_node.children):
                preview = child.text[:60].replace("\n", " ")
                if len(child.text) > 60:
                    preview += "..."
                self.console.print(f"  [{i + 1}] {preview}")

    def display_help(self):
        """Display available commands."""
        help_text = """
[bold]Commands:[/bold]
  [cyan]c[/cyan] / [cyan]continue[/cyan]  - Continue current branch (single generation)
  [cyan]g[/cyan] / [cyan]generate[/cyan]  - Generate multiple branches to choose from
  [cyan]w[/cyan] / [cyan]write[/cyan]     - Write text manually
  [cyan]a[/cyan] / [cyan]analyze[/cyan]   - Analyze current node with Claude
  [cyan]1-9[/cyan]           - Select a branch by number
  [cyan]u[/cyan] / [cyan]up[/cyan]        - Go back to parent node
  [cyan]r[/cyan] / [cyan]root[/cyan]      - Go back to root
  [cyan]t[/cyan] / [cyan]tree[/cyan]      - Show full tree structure
  [cyan]s[/cyan] / [cyan]save[/cyan]      - Save tree to file
  [cyan]o[/cyan] / [cyan]options[/cyan]   - Configure generation settings
  [cyan]R[/cyan] / [cyan]reload[/cyan]    - Hot-reload code (after editing)
  [cyan]h[/cyan] / [cyan]help[/cyan]      - Show this help
  [cyan]q[/cyan] / [cyan]quit[/cyan]      - Exit
"""
        self.console.print(Panel(help_text, title="Help", border_style="yellow"))

    def display_tree_structure(self, node: Node | None = None, rich_tree: RichTree | None = None, is_current: bool = False):
        """Display the full tree structure."""
        if node is None:
            node = self.tree.root
            preview = node.text[:40].replace("\n", " ") if node.text else "[root]"
            label = f"[bold]{preview}[/bold]" if node.id == self.current_node.id else preview
            rich_tree = RichTree(label)

        for child in node.children:
            preview = child.text[:40].replace("\n", " ")
            if len(child.text) > 40:
                preview += "..."
            label = f"[bold cyan]{preview}[/bold cyan]" if child.id == self.current_node.id else preview
            branch = rich_tree.add(label)
            self.display_tree_structure(child, branch)

        if node == self.tree.root:
            self.console.print(Panel(rich_tree, title="Tree Structure", border_style="magenta"))

    def continue_branch(self):
        """Generate a single continuation and follow it."""
        full_text = self.tree.get_full_text(self.current_node.id)
        if not full_text:
            self.console.print("[yellow]No text to continue from. Write something first.[/yellow]")
            return

        self.console.print("[dim]Generating continuation...[/dim]")
        try:
            text = self.generator.generate_single(full_text)
            new_node = self.tree.add_branch(self.current_node.id, text)
            self.current_node = new_node
            self.console.print("[green]Continued[/green]")
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def generate_branches(self):
        """Generate new branches from current position."""
        full_text = self.tree.get_full_text(self.current_node.id)
        if not full_text:
            self.console.print("[yellow]No text to continue from. Write something first.[/yellow]")
            return

        num = IntPrompt.ask("How many branches?", default=3)
        self.console.print(f"[dim]Generating {num} continuations...[/dim]")

        try:
            continuations = self.generator.generate_continuations(full_text, num_branches=num)

            self.console.print("\n[bold green]Generated branches:[/bold green]")
            for i, text in enumerate(continuations):
                preview = text[:100].replace("\n", " ")
                if len(text) > 100:
                    preview += "..."
                self.console.print(f"\n[cyan][{i + 1}][/cyan] {preview}")

            choice = Prompt.ask(
                "\nAdd which branches? (e.g., '1,3' or 'all' or 'none')",
                default="all"
            )

            if choice.lower() == "none":
                return
            elif choice.lower() == "all":
                indices = list(range(len(continuations)))
            else:
                indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]

            for i in indices:
                if 0 <= i < len(continuations):
                    self.tree.add_branch(self.current_node.id, continuations[i])

            self.console.print(f"[green]Added {len(indices)} branch(es)[/green]")

        except Exception as e:
            self.console.print(f"[red]Error generating: {e}[/red]")

    def write_text(self):
        """Manually write text."""
        self.console.print("[dim]Enter text (type '.' on its own line to finish):[/dim]")
        lines = []
        while True:
            line = input()
            if line == ".":
                break
            lines.append(line)

        if lines:
            text = "\n".join(lines)
            if not self.current_node.text and self.current_node == self.tree.root:
                # Set root text if empty
                self.current_node.text = text
            else:
                # Add as child
                new_node = self.tree.add_branch(self.current_node.id, text)
                self.current_node = new_node
            self.console.print("[green]Text added[/green]")

    def configure(self):
        """Configure generation settings."""
        self.console.print(f"\n[bold]Current config:[/bold]")
        self.console.print(f"  Model: {self.generator.config.model}")
        self.console.print(f"  Max tokens: {self.generator.config.max_tokens}")
        self.console.print(f"  Temperature: {self.generator.config.temperature}")
        self.console.print(f"  Top-p: {self.generator.config.top_p}")

        if Prompt.ask("\nChange settings?", choices=["y", "n"], default="n") == "y":
            self.generator.config.max_tokens = IntPrompt.ask(
                "Max tokens", default=self.generator.config.max_tokens
            )
            temp = Prompt.ask(
                "Temperature (0.0-2.0)", default=str(self.generator.config.temperature)
            )
            self.generator.config.temperature = float(temp)

    def save_tree(self):
        """Save tree to file."""
        from pathlib import Path
        from datetime import datetime
        trees_dir = Path(__file__).parent / "trees"
        trees_dir.mkdir(exist_ok=True)
        default_name = f"loom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename = Prompt.ask("Filename", default=default_name)
        if not filename.startswith("/"):
            filepath = trees_dir / filename
        else:
            filepath = Path(filename)
        self.tree.save(filepath)
        self.console.print(f"[green]Saved to {filepath}[/green]")

    def analyze_node(self):
        """Use Claude to analyze the current node's text."""
        import anthropic

        path = self.tree.get_path_to_node(self.current_node.id)
        prefix_nodes = path[:-1]
        prefix_text = "".join(n.text for n in prefix_nodes)
        continuation_text = self.current_node.text

        if not continuation_text:
            self.console.print("[yellow]No text to analyze at this node.[/yellow]")
            return

        if not prefix_text:
            self.console.print("[yellow]No prefix text - this is the root node.[/yellow]")
            return

        self.console.print("[dim]Analyzing with Claude...[/dim]")

        prompt = f"""Analyze this text generation. A language model (Qwen) was given a prefix and generated a continuation.

PREFIX:
{prefix_text}

CONTINUATION (generated by Qwen):
{continuation_text}

Provide a brief analysis (2-3 sentences each):
1. **Prefix**: What is happening/being discussed in the prefix?
2. **Continuation**: What does the continuation add or where does it take the text?
3. **Qwen's interpretation**: Based on how the continuation follows the prefix, what does Qwen seem to think the text is about or where it should go?"""

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = response.content[0].text

            # Store in node
            self.current_node.analysis = analysis
            self.console.print(Panel(
                analysis,
                title="[bold yellow]Analysis[/bold yellow]",
                border_style="yellow"
            ))
        except Exception as e:
            self.console.print(f"[red]Error analyzing: {e}[/red]")

    def reload_modules(self):
        """Hot-reload the code modules."""
        import tree
        import generator
        importlib.reload(tree)
        importlib.reload(generator)
        # Recreate generator with reloaded module
        from generator import Generator, GenerationConfig
        self.generator = Generator(GenerationConfig(
            model=self.generator.config.model,
            max_tokens=self.generator.config.max_tokens,
            temperature=self.generator.config.temperature,
            top_p=self.generator.config.top_p,
        ))
        self.console.print("[green]Modules reloaded[/green]")

    def run(self):
        """Main UI loop."""
        self.console.print("[bold magenta]== Loom: Tree-based Writing ==[/bold magenta]\n")
        self.display_help()
        input("\nPress Enter to start...")

        while True:
            self.display_current_state()
            cmd = Prompt.ask("\n[bold]Command[/bold]", default="h").strip().lower()

            if cmd in ("q", "quit", "exit"):
                if Prompt.ask("Save before quitting?", choices=["y", "n"], default="y") == "y":
                    self.save_tree()
                break

            elif cmd in ("h", "help", "?"):
                self.display_help()
                input("\nPress Enter to continue...")

            elif cmd in ("c", "continue", "cont"):
                self.continue_branch()

            elif cmd in ("g", "generate", "gen"):
                self.generate_branches()
                input("\nPress Enter to continue...")

            elif cmd in ("w", "write"):
                self.write_text()

            elif cmd in ("a", "analyze"):
                self.analyze_node()
                input("\nPress Enter to continue...")

            elif cmd in ("u", "up", "back"):
                if self.current_node.parent_id:
                    parent = self.tree.get_node(self.current_node.parent_id)
                    if parent:
                        self.current_node = parent
                else:
                    self.console.print("[yellow]Already at root[/yellow]")

            elif cmd in ("r", "root"):
                self.current_node = self.tree.root

            elif cmd in ("t", "tree"):
                self.display_tree_structure()
                input("\nPress Enter to continue...")

            elif cmd in ("s", "save"):
                self.save_tree()
                input("\nPress Enter to continue...")

            elif cmd in ("o", "options", "config"):
                self.configure()
                input("\nPress Enter to continue...")

            elif cmd in ("R", "reload"):
                self.reload_modules()
                input("\nPress Enter to continue...")

            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(self.current_node.children):
                    self.current_node = self.current_node.children[idx]
                else:
                    self.console.print(f"[red]Invalid branch number[/red]")

            else:
                self.console.print(f"[red]Unknown command: {cmd}[/red]")
