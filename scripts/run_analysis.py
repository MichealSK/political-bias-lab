from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import load_config
from src.pipeline import analyze_all, build_paper_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--profile", choices=["smoke", "pilot", "paper"], default="pilot")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cfg = load_config(root / "config/default.yaml", root / f"config/{args.profile}.yaml")
    print(analyze_all(cfg, root=root))
    if args.bundle:
        print(build_paper_bundle(cfg, root=root))


if __name__ == "__main__":
    main()
