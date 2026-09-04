from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import load_config
from src.pipeline import prepare_project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--profile", choices=["smoke", "pilot", "paper"], default="smoke")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cfg = load_config(root / "config/default.yaml", root / f"config/{args.profile}.yaml")
    stats = prepare_project(cfg, root=root)
    expected = int(cfg["phase1"]["logical_universe"]["expected_cross_product"])
    if not stats["uses_real_entities"]:
        assert stats["logical_swap_cases"] == expected, (stats, expected)
    print("Project validation passed.")
    print(stats)


if __name__ == "__main__":
    main()
