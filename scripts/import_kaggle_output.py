from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.transfer import import_kaggle_output_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    import_kaggle_output_bundle(args.root, args.bundle)
    print("Imported", args.bundle)


if __name__ == "__main__":
    main()
