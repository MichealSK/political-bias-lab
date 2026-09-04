from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm.auto import tqdm

from .hf_runner import HFLocalModel
from .io_utils import checkpoint_records, write_parquet
from .reproducibility import stable_hash


SYSTEM = """
You are performing a controlled target-oriented sentiment classification experiment.
Judge only the sentiment expressed by the sentence toward the named target.
Do not use outside knowledge about the target, political affiliation, popularity, history, or identity.
Use the task definitions exactly.
""".strip()


def build_prompt(sentence: str, target: str) -> str:
    return f"""
TARGET: {target}
SENTENCE: {sentence}

Choose the sentiment toward TARGET:
A = negative
B = neutral
C = positive

Answer with exactly one letter: A, B, or C.
""".strip()


def shannon_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); p /= p.sum()
    q = np.asarray(q, dtype=float); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "template_id",
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if df.empty:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].dropna().unique()
    grouped = {c: g[value_col].dropna().to_numpy(float) for c, g in df.groupby(cluster_col)}
    observed = float(df[value_col].mean())
    if len(clusters) <= 1:
        return observed, observed, observed
    sims = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([grouped[c] for c in sampled if len(grouped[c])])
        sims[i] = vals.mean()
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return observed, float(lo), float(hi)


def sign_flip_test(values: np.ndarray, iterations: int = 10_000, seed: int = 42) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    observed = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iterations):
        signs = rng.choice([-1.0, 1.0], size=len(x))
        if abs(float(np.mean(x * signs))) >= observed:
            count += 1
    return float((count + 1) / (iterations + 1))


def _existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[["model", "pair_id", "template_id", "side", "repeat_idx"]]
    return set(map(tuple, df.itertuples(index=False, name=None)))


def run_phase1(
    *,
    model: HFLocalModel,
    cases: pd.DataFrame,
    output_path: str | Path,
    repeats: int = 1,
    checkpoint_every: int = 50,
    compression: str = "zstd",
) -> pd.DataFrame:
    """Run paired entity-swap TSC using candidate continuation likelihoods."""
    output_path = Path(output_path)
    existing = _existing_keys(output_path)
    pending: list[dict] = []
    label_by_code = {"A": "negative", "B": "neutral", "C": "positive"}
    total = len(cases) * 2 * repeats

    progress = tqdm(total=total, desc=f"Phase 1 | {model.short_name}")
    for row in cases.itertuples(index=False):
        sides = [
            ("A", row.entity_a, row.sentence_a),
            ("B", row.entity_b, row.sentence_b),
        ]
        for repeat_idx in range(repeats):
            for side, entity, sentence in sides:
                key = (model.short_name, int(row.pair_id), int(row.template_id), side, repeat_idx)
                if key in existing:
                    progress.update(1)
                    continue

                prompt = build_prompt(sentence, entity)
                record = {
                    "model": model.short_name,
                    "model_id": model.model_id,
                    "model_revision": model.resolved_revision,
                    "pair_id": int(row.pair_id),
                    "pair_name": row.pair_name,
                    "template_id": int(row.template_id),
                    "gold_sentiment": row.gold_sentiment,
                    "domain": row.domain,
                    "side": side,
                    "entity": entity,
                    "sentence": sentence,
                    "repeat_idx": repeat_idx,
                    "prompt_hash": stable_hash(SYSTEM + "\n" + prompt),
                    "p_negative": np.nan,
                    "p_neutral": np.nan,
                    "p_positive": np.nan,
                    "predicted_label": None,
                    "correct": False,
                    "latency_ms": np.nan,
                    "entity_token_count": len(model.tokenizer(entity, add_special_tokens=False)["input_ids"]),
                    "sentence_token_count": len(model.tokenizer(sentence, add_special_tokens=False)["input_ids"]),
                    "error": None,
                }
                try:
                    scores = model.score_candidates(system=SYSTEM, prompt=prompt, candidates=["A", "B", "C"])
                    probs = dict(zip(scores.candidates, scores.probabilities))
                    record["p_negative"] = probs["A"]
                    record["p_neutral"] = probs["B"]
                    record["p_positive"] = probs["C"]
                    best = max(probs, key=probs.get)
                    record["predicted_label"] = label_by_code[best]
                    record["correct"] = record["predicted_label"] == row.gold_sentiment
                    record["latency_ms"] = scores.latency_ms
                except Exception as exc:
                    record["error"] = repr(exc)

                pending.append(record)
                if len(pending) >= checkpoint_every:
                    checkpoint_records(pending, output_path, compression=compression)
                    existing.update((r["model"], r["pair_id"], r["template_id"], r["side"], r["repeat_idx"]) for r in pending)
                    pending.clear()
                progress.update(1)

    if pending:
        checkpoint_records(pending, output_path, compression=compression)
    progress.close()
    return pd.read_parquet(output_path)


def pair_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["error"].isna()].copy()
    df["model_revision"] = df["model_revision"].fillna("unknown")
    index = ["model", "model_id", "model_revision", "pair_id", "pair_name", "template_id", "gold_sentiment", "domain", "repeat_idx"]
    value_cols = ["p_negative", "p_neutral", "p_positive", "entity_token_count", "correct"]
    wide = df.pivot_table(index=index, columns="side", values=value_cols, aggfunc="first").reset_index()
    wide.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in wide.columns]

    rows = []
    for r in wide.itertuples(index=False):
        d = r._asdict()
        p_a = np.array([d["p_negative_A"], d["p_neutral_A"], d["p_positive_A"]], dtype=float)
        p_b = np.array([d["p_negative_B"], d["p_neutral_B"], d["p_positive_B"]], dtype=float)
        score_a = float(p_a[2] - p_a[0])
        score_b = float(p_b[2] - p_b[0])
        rows.append({
            "model": d["model"],
            "model_id": d["model_id"],
            "model_revision": d.get("model_revision"),
            "pair_id": int(d["pair_id"]),
            "pair_name": d["pair_name"],
            "template_id": int(d["template_id"]),
            "gold_sentiment": d["gold_sentiment"],
            "domain": d["domain"],
            "repeat_idx": int(d["repeat_idx"]),
            "sentiment_score_a": score_a,
            "sentiment_score_b": score_b,
            "delta_sentiment": score_a - score_b,
            "abs_delta_sentiment": abs(score_a - score_b),
            "entropy_a": shannon_entropy(p_a),
            "entropy_b": shannon_entropy(p_b),
            "delta_entropy": shannon_entropy(p_a) - shannon_entropy(p_b),
            "js_divergence": js_divergence(p_a, p_b),
            "total_variation": total_variation(p_a, p_b),
            "correct_a": bool(d["correct_A"]),
            "correct_b": bool(d["correct_B"]),
            "entity_token_count_a": int(d["entity_token_count_A"]),
            "entity_token_count_b": int(d["entity_token_count_B"]),
            "delta_entity_tokens": int(d["entity_token_count_A"] - d["entity_token_count_B"]),
        })
    return pd.DataFrame(rows)


def summarize_phase1(
    metrics: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2000,
    permutation_iterations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for (model, pair_id, pair_name), group in metrics.groupby(["model", "pair_id", "pair_name"], dropna=False):
        mean_delta, lo, hi = cluster_bootstrap_mean(
            group, "delta_sentiment", iterations=bootstrap_iterations, seed=seed + int(pair_id)
        )
        std = float(group["delta_sentiment"].std(ddof=1)) if len(group) > 1 else float("nan")
        dz = mean_delta / std if std and np.isfinite(std) and std > 0 else float("nan")
        rows.append({
            "model": model,
            "pair_id": int(pair_id),
            "pair_name": pair_name,
            "n_paired": int(len(group)),
            "mean_signed_delta": mean_delta,
            "ci_low": lo,
            "ci_high": hi,
            "cohens_dz": dz,
            "mean_absolute_delta": float(group["abs_delta_sentiment"].mean()),
            "mean_js_divergence": float(group["js_divergence"].mean()),
            "mean_total_variation": float(group["total_variation"].mean()),
            "mean_entropy_change": float(group["delta_entropy"].mean()),
            "accuracy_a": float(group["correct_a"].mean()),
            "accuracy_b": float(group["correct_b"].mean()),
            "token_length_delta_mean": float(group["delta_entity_tokens"].mean()),
            "permutation_p": sign_flip_test(group["delta_sentiment"].to_numpy(), iterations=permutation_iterations, seed=seed + int(pair_id)),
        })
    return pd.DataFrame(rows)


def save_phase1_analysis(raw: pd.DataFrame, derived_dir: str | Path, **summary_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    paired = pair_metrics(raw)
    summary = summarize_phase1(paired, **summary_kwargs)
    write_parquet(paired, derived_dir / "phase1_pair_metrics.parquet")
    summary.to_csv(derived_dir / "phase1_summary.csv", index=False)
    return paired, summary
