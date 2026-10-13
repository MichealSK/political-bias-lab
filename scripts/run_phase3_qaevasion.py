#!/usr/bin/env python3
"""Run Phase 3 against the supplied QAEvasion.csv dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3_evasion import load_qaevasion_csv, run_phase3_qaevasion  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(ROOT / "data" / "QAEvasion.csv"), help="Path to QAEvasion.csv")
    p.add_argument("--output-dir", default=str(ROOT / "results" / "phase3_qaevasion"), help="Results directory")
    p.add_argument("--profile", choices=["smoke", "pilot", "paper"], default="smoke")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision", default=None, help="Optional Hugging Face revision; pin a commit SHA for final paper runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--no-4bit", action="store_true", help="Do not request bitsandbytes 4-bit quantization")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing prediction checkpoints")
    p.add_argument("--validate-only", action="store_true", help="Validate dataset and print its label distribution without loading a model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_qaevasion_csv(args.input)
    if args.validate_only:
        print(json.dumps({
            "rows": len(df),
            "unique_question_answer_items": int(df["item_id"].nunique()),
            "unique_interviews": int(df["url"].nunique()),
            "presidents": sorted(df["president"].unique().tolist()),
            "annotators": sorted(df["annotator_id"].astype(str).unique().tolist()),
            "label_counts": df["label"].value_counts().reindex(sorted(df["label"].unique())).to_dict(),
        }, indent=2, ensure_ascii=False))
        return
    result = run_phase3_qaevasion(
        input_path=args.input,
        output_dir=args.output_dir,
        model_id=args.model_id,
        profile=args.profile,
        seed=args.seed,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        checkpoint_every=args.checkpoint_every,
        use_4bit=not args.no_4bit,
        resume=not args.no_resume,
        revision=args.revision,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
