"""Terminal UI for navigating the text tree.

Presentation only: all state lives in a `Loom` (api.py) and every mutation is
delegated to it, so the TUI, the Streamlit GUI, and scripts share one code path.
"""

from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, IntPrompt
from rich.tree import Tree as RichTree

from tree import Node
from generator import perplexity
from api import Loom, analyze_continuation, model_label
from coloring import token_segments, hex_for_logprob, alt_bar_items, LEGEND


def _preview(text: str, n: int) -> str:
    """First n chars on one line, with an ellipsis if truncated."""
    out = text[:n].replace("\n", " ")
    return out + "..." if len(text) > n else out


def _ppl_tag(node_or_gen) -> str:
    """' (ppl X.X)' confidence tag, or '' when no logprobs are available."""
    ppl = perplexity(node_or_gen.logprobs)
    return f" (ppl {ppl:.1f})" if ppl is not None else ""


class LoomUI:
    """Interactive terminal UI for exploring branching text."""

    def __init__(self, loom: Loom | None = None):
        self.console = Console()
        self.loom = loom or Loom()

    # State lives on the Loom; expose read-only views for the display code.
    @property
    def tree(self):
        return self.loom.tree

    @property
    def generator(self):
        return self.loom.generator

    @property
    def current_node(self):
        return self.loom.current_node

    def _append_text(self, styled: Text, text: str, logprobs: dict | None,
                     base_style: str, bold: bool = True):
        """Append text to a Rich Text, coloring per-token by surprisal if possible.

        Tokens with no logprob fall back to base_style (e.g. "dim" for the
        human-written prefix); colored tokens use the surprisal ramp.
        """
        for tok, lp in token_segments(text, logprobs):
            if lp is None:
                style = base_style
            else:
                style = f"bold {hex_for_logprob(lp)}" if bold else hex_for_logprob(lp)
            styled.append(tok, style=style)

    def display_current_state(self):
        """Display the current text and available branches."""
        self.console.clear()

        # Show full text path with current node highlighted
        path = self.tree.get_path_to_node(self.current_node.id)
        if path:
            # Build styled text: prefix dim, current node colored by logprob
            styled_text = Text()
            current_node = path[-1]

            prefix_text = "".join(n.text for n in path[:-1])
            if prefix_text:
                for n in path[:-1]:
                    self._append_text(styled_text, n.text, n.logprobs, "dim", bold=False)
            if current_node.text:
                self._append_text(styled_text, current_node.text, current_node.logprobs, "bold cyan")

            if styled_text:
                title = "[bold blue]Current Text[/bold blue]"
                if current_node.logprobs:
                    title += f" [dim]({LEGEND})[/dim]"
                self.console.print(Panel(styled_text, title=title, border_style="blue"))
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
        self.console.print(f"\n[dim]Depth: {self.loom.depth} | Node: {self.current_node.id} | Children: {len(self.current_node.children)}[/dim]")

        # Show siblings (this node is a branch point) so they can be cycled
        sibs = self.loom.siblings()
        if len(sibs) > 1:
            idx = next((i for i, c in enumerate(sibs) if c.id == self.current_node.id), 0)
            self.console.print(f"[dim]Branch point: sibling {idx + 1}/{len(sibs)} — [n]ext / [p]rev to cycle[/dim]")

        # Show children if any
        if self.current_node.children:
            self.console.print("\n[bold green]Branches:[/bold green]")
            for i, child in enumerate(self.current_node.children):
                self.console.print(f"  [{i + 1}]{_ppl_tag(child)} {_preview(child.text, 60)}")

    def display_help(self):
        """Display available commands."""
        help_text = """
[bold]Commands:[/bold]
  [cyan]c[/cyan] / [cyan]continue[/cyan]  - Continue current branch (single generation)
  [cyan]g[/cyan] / [cyan]generate[/cyan]  - Generate multiple branches to choose from
  [cyan]w[/cyan] / [cyan]write[/cyan]     - Write text manually
  [cyan]a[/cyan] / [cyan]analyze[/cyan]   - Analyze current node with Claude
  [cyan]k[/cyan] / [cyan]candidates[/cyan] - Show top-k next-token candidates per token
  [cyan]S[/cyan] / [cyan]score[/cyan]     - Score current node's text (color by surprisal)
  [cyan]1-9[/cyan]           - Select a branch by number
  [cyan]u[/cyan] / [cyan]up[/cyan]        - Go back to parent node
  [cyan]n[/cyan] / [cyan]p[/cyan]         - Next / previous sibling (cycle a branch point)
  [cyan]b[/cyan] / [cyan]branch[/cyan]    - Split current node here and open a new sibling
  [cyan]r[/cyan] / [cyan]root[/cyan]      - Go back to root
  [cyan]z[/cyan] / [cyan]deepest[/cyan]   - Zoom to the tip of the longest branch below here
  [cyan]t[/cyan] / [cyan]tree[/cyan]      - Show full tree structure
  [cyan]s[/cyan] / [cyan]save[/cyan]      - Save tree to file
  [cyan]o[/cyan] / [cyan]options[/cyan]   - Configure generation settings
  [cyan]h[/cyan] / [cyan]help[/cyan]      - Show this help
  [cyan]q[/cyan] / [cyan]quit[/cyan]      - Exit
"""
        self.console.print(Panel(help_text, title="Help", border_style="yellow"))

    def display_tree_structure(self, node: Node | None = None, rich_tree: RichTree | None = None):
        """Display the full tree structure."""
        if node is None:
            node = self.tree.root
            preview = _preview(node.text, 40) if node.text else "[root]"
            label = f"[bold]{preview}[/bold]" if node.id == self.current_node.id else preview
            rich_tree = RichTree(label)

        for child in node.children:
            preview = _preview(child.text, 40)
            label = f"[bold cyan]{preview}[/bold cyan]" if child.id == self.current_node.id else preview
            branch = rich_tree.add(label)
            self.display_tree_structure(child, branch)

        if node == self.tree.root:
            self.console.print(Panel(rich_tree, title="Tree Structure", border_style="magenta"))

    def continue_branch(self):
        """Generate a single continuation and follow it."""
        self.console.print("[dim]Generating continuation...[/dim]")
        try:
            result = self.loom.continue_branch()
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return
        if result.startswith("Error"):
            self.console.print(f"[yellow]{result}[/yellow]")
        else:
            self.console.print("[green]Continued[/green]")

    def generate_branches(self):
        """Generate new branches from current position."""
        num = IntPrompt.ask("How many branches?", default=3)
        self.console.print(f"[dim]Generating {num} continuations...[/dim]")

        try:
            result = self.loom.generate(n=num)
        except Exception as e:
            self.console.print(f"[red]Error generating: {e}[/red]")
            return
        if result.startswith("Error"):
            self.console.print(f"[yellow]{result}[/yellow]")
            return

        self.console.print("\n[bold green]Generated branches:[/bold green]")
        for i, gen in enumerate(self.loom._last_branches):
            self.console.print(f"\n[cyan][{i + 1}][/cyan]{_ppl_tag(gen)} {_preview(gen.text, 100)}")

        choice = Prompt.ask(
            "\nAdd which branches? (e.g., '1,3' or 'all' or 'none')",
            default="all"
        )

        if choice.lower() == "none":
            self.loom._last_branches = []
        elif choice.lower() == "all":
            self.console.print(f"[green]{self.loom.select_all()}[/green]")
        else:
            nums = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
            self.console.print(f"[green]{self.loom.add_branches(nums)}[/green]")

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
            self.loom.write("\n".join(lines))
            self.console.print("[green]Text added[/green]")

    def configure(self):
        """Configure generation settings."""
        config = self.generator.config
        self.console.print(f"\n[bold]Current config:[/bold]")
        self.console.print(f"  Model: {config.model}")
        self.console.print(f"  Max tokens: {config.max_tokens}")
        self.console.print(f"  Temperature: {config.temperature}")
        self.console.print(f"  Top-p: {config.top_p}")

        if Prompt.ask("\nChange settings?", choices=["y", "n"], default="n") == "y":
            config.max_tokens = IntPrompt.ask("Max tokens", default=config.max_tokens)
            temp = Prompt.ask("Temperature (0.0-2.0)", default=str(config.temperature))
            config.temperature = float(temp)

    def save_tree(self):
        """Save tree to file."""
        trees_dir = Path(__file__).parent / "trees"
        trees_dir.mkdir(exist_ok=True)
        default_name = f"loom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename = Prompt.ask("Filename", default=default_name)
        filepath = Path(filename) if filename.startswith("/") else trees_dir / filename
        self.console.print(f"[green]{self.loom.save(str(filepath))}[/green]")

    def analyze_node(self):
        """Use Claude to analyze the current node's text."""
        self.console.print("[dim]Analyzing with Claude...[/dim]")
        try:
            path = self.tree.get_path_to_node(self.current_node.id)
            prefix_text = "".join(n.text for n in path[:-1])
            analysis = analyze_continuation(
                prefix_text, self.current_node.text, model_label(self.generator.config.model)
            )
        except ValueError as e:
            self.console.print(f"[yellow]{e}[/yellow]")
            return
        except Exception as e:
            self.console.print(f"[red]Error analyzing: {e}[/red]")
            return

        self.current_node.analysis = analysis
        self.console.print(Panel(
            analysis,
            title="[bold yellow]Analysis[/bold yellow]",
            border_style="yellow"
        ))

    def show_candidates(self):
        """Show the top-k next-token candidates the model considered per token."""
        node = self.current_node
        segs = token_segments(node.text or "", node.logprobs)
        top = (node.logprobs or {}).get("top_logprobs")
        if not top or len(top) != len(segs):
            self.console.print("[yellow]No top-k candidates stored for this node "
                               "(generate with the current code to capture them).[/yellow]")
            return
        body = Text()
        for (tok, lp), cand in zip(segs, top):
            tok_style = hex_for_logprob(lp) if lp is not None else "dim"
            body.append(f"{tok!r:>16}", style=f"bold {tok_style}")
            body.append("  ")
            items = alt_bar_items(cand, limit=5)
            body.append("  ".join(f"{t!r}={p * 100:.0f}%" for t, p in items), style="dim")
            body.append("\n")
        self.console.print(Panel(body, title="[bold]Next-token candidates[/bold]", border_style="cyan"))

    def branch_here(self):
        """Split the current node at a character offset and open a new sibling."""
        text = self.current_node.text
        if not text:
            self.console.print("[yellow]No text to split.[/yellow]")
            return
        self.console.print(f"[dim]Current text is {len(text)} chars.[/dim]")
        idx = IntPrompt.ask("Split at character offset", default=len(text) // 2)
        result = self.loom.split_and_branch(idx)
        style = "yellow" if result.startswith("Error") else "green"
        self.console.print(f"[{style}]{result}[/{style}]")

    def score_node(self):
        """Compute per-token logprobs for the current node (echo pass)."""
        self.console.print("[dim]Scoring current node...[/dim]")
        try:
            result = self.loom.score_node()
        except Exception as e:
            self.console.print(f"[red]Error scoring: {e}[/red]")
            return
        style = "yellow" if result.startswith(("Error", "Nothing")) else "green"
        self.console.print(f"[{style}]{result}[/{style}]")

    def run(self):
        """Main UI loop."""
        self.console.print("[bold magenta]== Loom: Tree-based Writing ==[/bold magenta]\n")
        self.display_help()
        input("\nPress Enter to start...")

        while True:
            self.display_current_state()
            raw = Prompt.ask("\n[bold]Command[/bold]", default="h").strip()
            cmd = raw.lower()

            if raw == "S" or cmd == "score":
                self.score_node()
                input("\nPress Enter to continue...")

            elif cmd in ("q", "quit", "exit"):
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
                self.loom.up()

            elif cmd in ("n", "next"):
                result = self.loom.select_sibling(1)
                if result.startswith("No siblings"):
                    self.console.print(f"[yellow]{result}[/yellow]")

            elif cmd in ("p", "prev", "previous"):
                result = self.loom.select_sibling(-1)
                if result.startswith("No siblings"):
                    self.console.print(f"[yellow]{result}[/yellow]")

            elif cmd in ("b", "branch"):
                self.branch_here()

            elif cmd in ("k", "candidates", "cand"):
                self.show_candidates()
                input("\nPress Enter to continue...")

            elif cmd in ("r", "root"):
                self.loom.root()

            elif cmd in ("z", "deepest", "deep"):
                self.console.print(f"[cyan]{self.loom.deepest()}[/cyan]")

            elif cmd in ("t", "tree"):
                self.display_tree_structure()
                input("\nPress Enter to continue...")

            elif cmd in ("s", "save"):
                self.save_tree()
                input("\nPress Enter to continue...")

            elif cmd in ("o", "options", "config"):
                self.configure()
                input("\nPress Enter to continue...")

            elif cmd.isdigit():
                result = self.loom.child(int(cmd))
                if result.startswith(("Error", "No children")):
                    self.console.print(f"[red]{result}[/red]")

            else:
                self.console.print(f"[red]Unknown command: {cmd}[/red]")
