from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.transfer import make_kaggle_input_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="transfer/kaggle_input.zip")
    args = parser.parse_args()
    path = make_kaggle_input_bundle(args.root, Path(args.root) / args.out)
    print(path)


if __name__ == "__main__":
    main()
