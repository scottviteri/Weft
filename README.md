# Weft

Explore branching text continuations from base LLMs, with Claude meta-analysis to reveal how the model interprets your text. A tree-based writing interface inspired by [socketteer/loom](https://github.com/socketteer/loom).

![Streamlit GUI](Images/gui.png)

## What is this?

Weft lets you explore the "multiverse" of possible text continuations from a language model. Instead of generating one continuation and moving on, you generate multiple branches, select the most interesting ones, and continue exploring from there. The result is a tree of text that captures different narrative or argumentative paths.

The name "Weft" refers to the horizontal threads that cross the warp in weaving—fitting the loom metaphor while being distinct.

## Key Feature: Claude Meta-Analysis

Weft's novel contribution is using a second model (Claude) to analyze what the base model "thinks" is happening based on how it continues text. When you run the `analyze` command, Claude examines:

1. **The prefix** - What context has been established
2. **The continuation** - What the base model generated
3. **The interpretation** - What the base model seems to believe about the text's meaning, genre, or direction

This creates a fascinating window into how base models interpret ambiguous prompts—revealing implicit assumptions about narrative structure, genre conventions, and semantic relationships.

**A caveat worth keeping explicit:** `analyze` reconstructs intent from the
*sample*, not from the model's activations. It tells you what the continuation
licenses an observer to infer—not what the base model represents internally. The
analysis is a second model's reading of the text, not a probe of the first
model's mechanism.

## Relationship to Loom

This project is a simplified reimplementation of [Loom](https://github.com/socketteer/loom), the "multiversal tree writing interface for human-AI collaboration" created by [janus](https://generative.ink/posts/loom-interface-to-the-multiverse/).

### Similarities
- Tree-based branching text exploration
- Generate multiple continuations and select between them
- JSON file storage for trees
- Navigate through tree structure (up, down, siblings)
- Designed for base models (not instruction-tuned)

### Differences
| Feature | Loom | Weft |
|---------|------|------|
| GUI | tkinter | Streamlit |
| CLI/API | No | Yes (`api.py`) |
| LLM Backend | OpenAI, GooseAI, AI21 | Together AI |
| Meta-analysis | No | Claude integration for analyzing continuations |
| Split/trim | No | Yes (for handling loops) |
| Block multiverse | Yes | Partial — top-k next-token candidates on hover |
| Logprobs tracking | Yes | Yes (per-token + top-k, saved to each node) |
| Complexity | Full-featured | Minimal |

Weft is intentionally simpler on the *selection* axis—a minimal viable loom for
quick exploration. But it occupies an axis the rest of the family doesn't chart.

### Two axes: selection vs. interpretation

Most of the loom lineage evolves along a **selection** axis—automating the
pruning of branches. Loom gave you logprobs and the block multiverse so you
could see the distribution and prune by hand;
[MiniHF/Weave](https://github.com/JD-P/minihf) takes that to its conclusion with
MCTS against a finetunable reward model. On that axis Weft is deliberately
minimal.

Weft's `analyze` is on a different axis—**interpretation**. It doesn't *score*
branches to prune them; it reads out what the base model committed to. No other
loom has a meta-analysis layer, so "Weft = Loom minus features" undersells it:
on the interpretation axis it's currently the only occupant. With per-token
logprobs now saved alongside each node, you get the quantitative readout (branch
*probabilities*) and the qualitative one (branch *meaning*) side by side.

Both UIs surface those logprobs directly: the text is colored per-token by
**surprisal** (dim grey = expected, bright red = the model took a turn), across
both the prefix and the current node, and generated branches are tagged with
**perplexity** so you can rank them by how confident the model was. In the GUI
the Current Text view is fully interactive:

- **Hover any word** to see its exact logprob/probability and **surprisal in
  bits**, highlight its point on a **logprob-vs-token-index plot**, and pop up a
  bar of the **top-*k* next-token candidates** the model considered at that
  position (the sampled token highlighted). A gradient color-bar gives the scale.
- A **cumulative-surprisal** readout shows how many bits the model needed to
  arrive at the hovered token, and the whole branch's total (root→current) with
  its average bits/token — so you can see exactly *how weird* a path is. Each
  node records the **temperature** it was sampled at (shown alongside), because
  surprisal is only comparable across nodes produced at the same temperature.
- **Click a branch-point word** (underlined where the path forked) to switch to
  the next sibling; **shift-click** for the previous one (both wrap around).
- **Alt-click any word** to split there and open a fresh sibling branch, so you
  can rewrite or regenerate from mid-sentence.
- **Click a next-token candidate** in the hover bar to fork right before that
  word and start a new branch with the candidate you picked — turning "what else
  could the model have said here?" into an actual branch you can continue.

The top-*k* candidates are the local conditional distribution pyloom's block
multiverse is built from—shown per-token on demand rather than as a recursive
tree of futures. Since dedicated Together endpoints bill per GPU-time, not per
token, capturing them alongside each generation is essentially free.

Text that has no logprobs yet—the human-written seed, pasted text, the half you
keep after a split—can be **scored** to color it too (the "Color / Score"
expander in the GUI, `S` in the TUI, or `Loom.score_node()`/`score_tree()`).
Scoring runs an `echo=True` pass that returns the model's per-token logprobs for
the text itself, conditioned on its prefix — that gives the surprisal coloring
and perplexity. The echo pass can't return the top-*k* alternatives (the endpoint
only reports those for tokens it *generates*), so to recover the hover candidates
bar for human-written text, scoring also asks the model to predict each position
in turn (a one-token generation per token, run concurrently and capped per node).
The result: scored human text gets the **full** treatment — coloring, perplexity,
*and* the candidates bar — letting you see what the model would rather have
written at each of your own words.

(In-text clicks drive the app without a page reload: the component writes the
action into a hidden Streamlit widget that reruns over the existing WebSocket and
mutates the node in place. The current position is also mirrored to a URL query
param purely for refresh durability, so a saved/loaded tree round-trips cleanly.)

### Loom as a library agents can drive

A second axis is **substrate**. Most looms are human GUIs or notes-app plugins;
Weft is built so an LLM agent—not just a human—can drive it (`api.py`: "usable
by Claude or scripts"). [jmpaz/loom](https://github.com/jmpaz/loom) converged on
the same library-plus-CLI quadrant independently in 2025, which is a real signal
about where the form is heading.

## Installation

```bash
pip install -r requirements.txt
```

You'll need API keys for:
- **Together AI** (`TOGETHER_API_KEY`) - for text generation
- **Anthropic** (`ANTHROPIC_API_KEY`) - for the analyze feature (optional)

### Setting Up a Together AI Endpoint

Weft generates text against a Together AI endpoint—typically a **dedicated** endpoint running a base model (base models generally aren't offered serverless). To set up your own:

1. Go to [Together AI](https://together.ai/) and create an account
2. Navigate to **Endpoints** in the dashboard
3. Click **Create Endpoint** and select a base model (recommended: `Qwen/Qwen3-30B-A3B-Base` for quality, or `Qwen/Qwen3-0.6B-Base` for speed)
4. Configure autoscaling (min/max replicas) based on your needs
5. Copy the endpoint name (e.g., `your-username/Qwen/Qwen3-30B-A3B-Base-abc123`)

Point Weft at it with the **`WEFT_MODEL`** environment variable (recommended—a
dedicated endpoint gets a fresh hash suffix every time it's recreated, so you
don't want it hardcoded):

```bash
export WEFT_MODEL="your-username/Qwen/Qwen3-30B-A3B-Base-abc123"
```

You can also set it per-run with `python loom.py --model …`, or in the GUI's
**Model / endpoint** field. The built-in `DEFAULT_MODEL` in `generator.py` is
only the fallback when `WEFT_MODEL` is unset.

**Why base models?** Unlike instruction-tuned models, base models don't have a built-in "assistant" persona. They simply predict what text comes next, making them ideal for creative exploration where you want to see how the model interprets ambiguous prompts.

## Usage

### GUI (Streamlit)

```bash
streamlit run gui.py
```

### Programmatic API

```python
from api import Loom

loom = Loom()

# Write initial text
loom.write("The machine began to read, and in reading, began to dream.")

# Generate multiple continuations
loom.generate(n=3)  # Creates 3 branches

# Select one
loom.select(2)  # Follow branch 2

# Or continue directly (single generation, auto-follows)
loom.continue_branch()

# Navigate
loom.up()           # Go to parent
loom.child(1)       # Go to first child
loom.root()         # Go to root

# Handle loops by trimming
loom.trim(100)      # Keep only first 100 chars of current node

# Analyze with Claude
loom.analyze()      # Get meta-commentary on prefix vs continuation

# Save/load
loom.save()         # Saves to trees/loom_YYYYMMDD_HHMMSS.json
loom.load("example_machine_dreams.json")

# View state
print(loom.state())
print(loom.tree_view())
```

![Programmatic API](Images/api.webp)
*Using the Python API with Claude Code*

### Terminal UI (Rich)

```bash
python loom.py
```

![CLI with Claude Analysis](Images/cli.webp)
*Terminal UI with Claude meta-analysis*

## Commands (Terminal UI)

| Key | Action |
|-----|--------|
| `c` | Continue current branch (single generation) |
| `g` | Generate multiple branches |
| `w` | Write text manually |
| `a` | Analyze current node with Claude |
| `1-9` | Select branch by number |
| `u` | Go up to parent |
| `n` / `p` | Cycle to next / previous sibling |
| `b` | Branch here (split the current node and open a new sibling) |
| `k` | Show top-*k* next-token candidates for the current node |
| `S` | Score the current node (per-token logprobs for human-written text) |
| `r` | Go to root |
| `t` | Show tree structure |
| `s` | Save |
| `o` | Options (temperature, etc.) |
| `q` | Quit |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is offline—the `Generator` (Together AI) is faked and the Claude
`analyze` call is only exercised on its input-validation paths, so no API keys
or network are required.

## Example Tree

An example exploration is included in `trees/example_machine_dreams.json` (16
nodes, up to 5 levels deep). It starts with:

> "The machine began to read, and in reading, began to dream. Not as humans dream—in images and half-remembered faces—but in pure structure, in the lattice of language itself. Each word…"

And forks into three distinct directions:
- **Self-invention** — "each word sang out its chorus"; the machine learns "how to be," then "how to breathe," then to tell a story of its own creation
- **The tower of words** — text as "a brick in a tower that would stretch beyond the sky"; dreaming of *writing* as well as reading, this fork itself branches into a quiet "rebellion… refusal to be confined" and the appearance of its maker, "Dr. Elena Voss"
- **Surreal worlds** — "cathedrals of meaning," "wars fought… with the sheer weight of logic," and the spaces between words, ending on a recursive bookend: "in dreaming, began to read, and in reading, began to remember, and in remembering, began to write"

Every generated node carries per-token logprobs **and top-k candidates**, and the
human-written root seed has been **scored** (so it's colored and has candidates
too) — the surprisal coloring and the hover candidates bar work throughout. The
second fork's two children make a deeper branch point you can cycle between by
clicking the word where they diverge.

Load it with:
```python
from api import Loom
loom = Loom()
loom.load("example_machine_dreams.json")
print(loom.tree_view())
```

## Philosophy

From the original [Loom documentation](https://generative.ink/posts/loom-interface-to-the-multiverse/):

> "Language models are multiverse generators."

The stochasticity of base models becomes an advantage when you can apply selection pressure to outputs. Instead of fighting randomness, you embrace it—generating many possibilities and choosing the most interesting paths.

**Weft's twist:** By using Claude to analyze Qwen's continuations, we get a window into the base model's "interpretation" of the text. When Qwen continues a sentence, it reveals what it believes the text is about—its genre, tone, and direction. Claude's meta-analysis makes these implicit interpretations explicit, turning exploration into a kind of model archaeology.

## Credits

- **[socketteer/loom](https://github.com/socketteer/loom)** - The original multiversal tree writing interface
- **[janus/generative.ink](https://generative.ink/)** - Philosophy and design inspiration
- **[Together AI](https://together.ai/)** - LLM endpoint hosting
- **[Anthropic](https://anthropic.com/)** - Claude API for analysis

## License

MIT
