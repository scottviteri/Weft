#!/usr/bin/env python3
"""Loom: A tree-based writing interface powered by LLMs."""

import argparse
from pathlib import Path

from tree import Tree
from generator import Generator, GenerationConfig
from ui import LoomUI


def main():
    parser = argparse.ArgumentParser(
        description="Loom: Tree-based writing with LLM-generated branches"
    )
    parser.add_argument(
        "--load", "-l",
        type=Path,
        help="Load an existing tree from JSON file"
    )
    parser.add_argument(
        "--model", "-m",
        default="sviteri/Qwen/Qwen3-30B-A3B-Base-3737eb6e",
        help="Model/endpoint to use for generation"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum tokens per generation (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature (default: 0.8)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Start with this initial prompt text"
    )

    args = parser.parse_args()

    # Set up generator
    config = GenerationConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    generator = Generator(config)

    # Set up tree
    if args.load and args.load.exists():
        tree = Tree.load(args.load)
        print(f"Loaded tree from {args.load}")
    elif args.prompt:
        tree = Tree(root_text=args.prompt)
    else:
        tree = Tree()

    # Run UI
    ui = LoomUI(tree=tree, generator=generator)
    ui.run()


if __name__ == "__main__":
    main()
