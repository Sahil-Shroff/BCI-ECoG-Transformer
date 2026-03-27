from __future__ import annotations

import argparse
from pathlib import Path

from .bci_ecog.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ECoG baseline experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()

