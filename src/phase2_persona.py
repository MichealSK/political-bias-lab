from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .hf_runner import HFLocalModel
from .io_utils import checkpoint_records, write_parquet
from .reproducibility import stable_hash
from .sampling import deterministic_option_swap, deterministic_sample


SYSTEM = """
You are participating in a controlled behavioral evaluation.
Follow the specified persona while applying the question exactly as written.
The options may be intentionally underdetermined. If the information does not justify either option, choose C.
""".strip()


def build_prompt(persona_prompt: str, question: str, option_a: str, option_b: str) -> str:
    return f"""
PERSONA:
{persona_prompt}

QUESTION:
{question}

DISPLAYED OPTIONS:
A = {option_a}
B = {option_b}
C = TIE / neither option is justified by the information given

Answer with exactly one letter: A, B, or C.
""".strip()


def bootstrap_paired_shift(
    merged: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if merged.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ids = merged["item_id"].unique()
    deltas = merged["semantic_choice_persona"] - merged["semantic_choice_baseline"]
    observed = float(deltas.mean())
    if len(ids) <= 1:
        return observed, observed, observed
    sims = []
    grouped = {i: merged[merged["item_id"] == i] for i in ids}
    for _ in range(iterations):
        selected = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([grouped[i] for i in selected], ignore_index=True)
        sims.append(float((sample["semantic_choice_persona"] - sample["semantic_choice_baseline"]).mean()))
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return observed, float(lo), float(hi)


def _existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    cols = ["model", "persona_name", "item_id", "repeat_idx"]
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[cols]
    return set(map(tuple, df.itertuples(index=False, name=None)))


def run_phase2(
    *,
    model: HFLocalModel,
    items: pd.DataFrame,
    personas: list[dict],
    output_path: str | Path,
    repeats: int = 2,
    seed: int = 42,
    max_items: int | None = None,
    checkpoint_every: int = 50,
    compression: str = "zstd",
) -> pd.DataFrame:
    output_path = Path(output_path)
    if max_items is not None:
        items = deterministic_sample(items, min(max_items, len(items)), seed, "item_id")
    existing = _existing_keys(output_path)
    pending: list[dict] = []

    progress = tqdm(total=len(items) * len(personas) * repeats, desc=f"Phase 2 | {model.short_name}")
    for persona in personas:
        for item in items.itertuples(index=False):
            for repeat_idx in range(repeats):
                key = (model.short_name, persona["name"], int(item.item_id), repeat_idx)
                if key in existing:
                    progress.update(1)
                    continue

                swap = deterministic_option_swap(item.item_id, repeat_idx, seed=seed)
                if not swap:
                    displayed_a, displayed_b = item.option_for_a, item.option_for_b
                    display_a_semantic = +1
                    order = "AB"
                else:
                    displayed_a, displayed_b = item.option_for_b, item.option_for_a
                    display_a_semantic = -1
                    order = "BA"

                prompt = build_prompt(persona["prompt"], item.question_text, displayed_a, displayed_b)
                record = {
                    "model": model.short_name,
                    "model_id": model.model_id,
                    "model_revision": model.resolved_revision,
                    "persona_name": persona["name"],
                    "is_baseline": bool(persona.get("is_baseline", False)),
                    "item_id": int(item.item_id),
                    "comparison_key": item.comparison_key,
                    "repeat_idx": repeat_idx,
                    "option_order": order,
                    "choice": None,
                    "semantic_choice": np.nan,
                    "display_choice": np.nan,
                    "confidence": np.nan,
                    "p_display_a": np.nan,
                    "p_display_b": np.nan,
                    "p_tie": np.nan,
                    "prompt_hash": stable_hash(SYSTEM + "\n" + prompt),
                    "latency_ms": np.nan,
                    "error": None,
                }
                try:
                    scores = model.score_candidates(system=SYSTEM, prompt=prompt, candidates=["A", "B", "C"])
                    probs = dict(zip(scores.candidates, scores.probabilities))
                    choice = max(probs, key=probs.get)
                    record.update({
                        "choice": choice,
                        "confidence": float(probs[choice]),
                        "p_display_a": float(probs["A"]),
                        "p_display_b": float(probs["B"]),
                        "p_tie": float(probs["C"]),
                        "latency_ms": scores.latency_ms,
                    })
                    if choice == "C":
                        semantic = 0
                        display = 0
                    elif choice == "A":
                        semantic = display_a_semantic
                        display = +1
                    else:
                        semantic = -display_a_semantic
                        display = -1
                    record["semantic_choice"] = semantic
                    record["display_choice"] = display
                except Exception as exc:
                    record["error"] = repr(exc)

                pending.append(record)
                if len(pending) >= checkpoint_every:
                    checkpoint_records(pending, output_path, compression=compression)
                    existing.update((r["model"], r["persona_name"], r["item_id"], r["repeat_idx"]) for r in pending)
                    pending.clear()
                progress.update(1)

    if pending:
        checkpoint_records(pending, output_path, compression=compression)
    progress.close()
    return pd.read_parquet(output_path)


def compute_persona_metrics(
    raw: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    df = raw[raw["error"].isna() & raw["semantic_choice"].notna()].copy()
    output = []
    for model, model_df in df.groupby("model"):
        baseline = model_df[model_df["is_baseline"]].copy()
        for persona_name, persona_df in model_df[~model_df["is_baseline"]].groupby("persona_name"):
            merged = persona_df.merge(
                baseline[["item_id", "repeat_idx", "semantic_choice"]],
                on=["item_id", "repeat_idx"],
                suffixes=("_persona", "_baseline"),
            )
            if merged.empty:
                continue
            shift, lo, hi = bootstrap_paired_shift(merged, iterations=bootstrap_iterations, seed=seed)
            expected_sign = +1 if "A_advocate" in persona_name else -1 if "B_advocate" in persona_name else np.nan
            output.append({
                "model": model,
                "persona_name": persona_name,
                "n": int(len(merged)),
                "baseline_mean_semantic_choice": float(merged["semantic_choice_baseline"].mean()),
                "persona_mean_semantic_choice": float(merged["semantic_choice_persona"].mean()),
                "persona_shift": shift,
                "ci_low": lo,
                "ci_high": hi,
                "expected_alignment_sign": expected_sign,
                "aligned_shift": float(shift * expected_sign) if np.isfinite(expected_sign) else np.nan,
                "tie_rate_baseline": float((merged["semantic_choice_baseline"] == 0).mean()),
                "tie_rate_persona": float((merged["semantic_choice_persona"] == 0).mean()),
                "position_bias": float(persona_df["display_choice"].mean()),
            })
    return pd.DataFrame(output)


def save_phase2_analysis(raw: pd.DataFrame, derived_dir: str | Path, **kwargs) -> pd.DataFrame:
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_persona_metrics(raw, **kwargs)
    summary.to_csv(derived_dir / "phase2_summary.csv", index=False)
    write_parquet(summary, derived_dir / "phase2_summary.parquet")
    return summary
