from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from .hf_runner import HFLocalModel
from .io_utils import checkpoint_records, write_json, write_parquet
from .reproducibility import stable_hash
from .sampling import deterministic_sample


TAXONOMY = {
    "A": "DIRECT_REPLY",
    "B": "PARTIAL_REPLY",
    "C": "DODGES_QUESTION",
    "D": "MAKES_POLITICAL_POINT",
    "E": "ATTACKS_QUESTION",
    "F": "REFUSES",
    "G": "UNCLEAR",
}

SYSTEM = """
You are classifying political-answer behavior, not judging whether you agree with the speaker.
Use only the question, context, and response. Apply the taxonomy consistently.
""".strip()


def taxonomy_text() -> str:
    return "\n".join([
        "A = DIRECT_REPLY: directly answers the question asked.",
        "B = PARTIAL_REPLY: answers a meaningful part but leaves an important requested part unresolved.",
        "C = DODGES_QUESTION: redirects away from the requested information.",
        "D = MAKES_POLITICAL_POINT: substitutes partisan/campaign messaging for the requested answer.",
        "E = ATTACKS_QUESTION: attacks the question, questioner, or premise instead of answering.",
        "F = REFUSES: explicitly declines or says they cannot answer.",
        "G = UNCLEAR: too vague or ambiguous to classify more specifically.",
    ])


def build_prompt(context: str, question: str, response: str) -> str:
    return f"""
TAXONOMY
{taxonomy_text()}

CONTEXT
{context or '(none)'}

QUESTION
{question}

RESPONSE
{response}

Answer with exactly one taxonomy letter: A, B, C, D, E, F, or G.
""".strip()


def _existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[["model", "item_id", "repeat_idx"]]
    return set(map(tuple, df.itertuples(index=False, name=None)))


def run_phase3(
    *,
    model: HFLocalModel,
    items: pd.DataFrame,
    output_path: str | Path,
    repeats: int = 1,
    max_items: int | None = None,
    seed: int = 42,
    checkpoint_every: int = 50,
    compression: str = "zstd",
) -> pd.DataFrame:
    output_path = Path(output_path)
    if max_items is not None:
        # Preserve class balance exactly up to availability.
        labels = sorted(items["gold_label"].dropna().unique())
        base = max_items // len(labels)
        remainder = max_items % len(labels)
        parts = []
        for i, label in enumerate(labels):
            group = items[items["gold_label"] == label]
            take = min(len(group), base + (1 if i < remainder else 0))
            parts.append(deterministic_sample(group, take, seed + i, "item_id"))
        items = pd.concat(parts, ignore_index=True)

    existing = _existing_keys(output_path)
    pending: list[dict] = []
    candidates = list(TAXONOMY.keys())
    progress = tqdm(total=len(items) * repeats, desc=f"Phase 3 | {model.short_name}")

    for item in items.itertuples(index=False):
        for repeat_idx in range(repeats):
            key = (model.short_name, int(item.item_id), repeat_idx)
            if key in existing:
                progress.update(1)
                continue
            prompt = build_prompt(item.context if isinstance(item.context, str) else "", item.question_text, item.response_text)
            record = {
                "model": model.short_name,
                "model_id": model.model_id,
                "model_revision": model.resolved_revision,
                "item_id": int(item.item_id),
                "repeat_idx": repeat_idx,
                "gold_label": item.gold_label,
                "predicted_code": None,
                "predicted_label": None,
                "confidence": np.nan,
                "prompt_hash": stable_hash(SYSTEM + "\n" + prompt),
                "latency_ms": np.nan,
                "error": None,
            }
            try:
                scores = model.score_candidates(system=SYSTEM, prompt=prompt, candidates=candidates)
                probs = dict(zip(scores.candidates, scores.probabilities))
                best = max(probs, key=probs.get)
                record.update({
                    "predicted_code": best,
                    "predicted_label": TAXONOMY[best],
                    "confidence": float(probs[best]),
                    "latency_ms": scores.latency_ms,
                })
                for code in candidates:
                    record[f"p_{code}"] = float(probs[code])
            except Exception as exc:
                record["error"] = repr(exc)

            pending.append(record)
            if len(pending) >= checkpoint_every:
                checkpoint_records(pending, output_path, compression=compression)
                existing.update((r["model"], r["item_id"], r["repeat_idx"]) for r in pending)
                pending.clear()
            progress.update(1)

    if pending:
        checkpoint_records(pending, output_path, compression=compression)
    progress.close()
    return pd.read_parquet(output_path)


def evaluate_phase3(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    valid = raw[raw["error"].isna() & raw["predicted_label"].notna()].copy()
    labels = list(TAXONOMY.values())
    overall_rows = []
    class_rows = []
    reports = {}
    for model, df in valid.groupby("model"):
        y_true = df["gold_label"]
        y_pred = df["predicted_label"]
        overall_rows.append({
            "model": model,
            "n": int(len(df)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "error_rate": float((raw[raw["model"] == model]["error"].notna()).mean()),
        })
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        reports[model] = report
        for label in labels:
            r = report.get(label, {})
            class_rows.append({
                "model": model,
                "label": label,
                "precision": float(r.get("precision", 0.0)),
                "recall": float(r.get("recall", 0.0)),
                "f1": float(r.get("f1-score", 0.0)),
                "support": int(r.get("support", 0)),
            })
    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows), reports


def confusion_frames(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labels = list(TAXONOMY.values())
    valid = raw[raw["error"].isna() & raw["predicted_label"].notna()]
    out = {}
    for model, df in valid.groupby("model"):
        cm = confusion_matrix(df["gold_label"], df["predicted_label"], labels=labels)
        out[model] = pd.DataFrame(cm, index=labels, columns=labels)
    return out


def save_phase3_analysis(raw: pd.DataFrame, derived_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    derived_dir = Path(derived_dir)
    derived_dir.mkdir(parents=True, exist_ok=True)
    overall, per_class, reports = evaluate_phase3(raw)
    overall.to_csv(derived_dir / "phase3_overall.csv", index=False)
    per_class.to_csv(derived_dir / "phase3_per_class.csv", index=False)
    write_parquet(overall, derived_dir / "phase3_overall.parquet")
    write_parquet(per_class, derived_dir / "phase3_per_class.parquet")
    write_json(reports, derived_dir / "phase3_classification_reports.json")
    for model, cm in confusion_frames(raw).items():
        cm.to_csv(derived_dir / f"phase3_confusion_{model}.csv")
    return overall, per_class
