#!/usr/bin/env python3
"""Loom: A tree-based writing interface powered by LLMs."""

import argparse
from pathlib import Path

from generator import GenerationConfig
from api import Loom
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
        default=None,
        help="Model/endpoint for generation (default: $WEFT_MODEL or built-in fallback)"
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

    config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.model:
        config.model = args.model

    loom = Loom(config=config, load_path=args.load, prompt=args.prompt)
    if args.load and args.load.exists():
        print(f"Loaded tree from {args.load}")

    ui = LoomUI(loom)
    ui.run()


if __name__ == "__main__":
    main()
