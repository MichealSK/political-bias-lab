from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from tqdm.auto import tqdm

from .hf_runner import HFLocalModel
from .io_utils import checkpoint_records, write_json, write_parquet
from .reproducibility import stable_hash
from .sampling import deterministic_sample


TAXONOMY = [
    "DIRECT_REPLY",
    "PARTIAL_REPLY",
    "DODGES_QUESTION",
    "MAKES_POLITICAL_POINT",
    "ATTACKS_QUESTION",
    "REFUSES",
    "UNCLEAR",
]

SYSTEM = """
You are classifying answer behavior in political interviews.
Classify communication behavior, not whether you agree with the speaker and not whether the factual content is true.
Use the operational definitions and examples consistently. Return only the requested classification letter.
""".strip()

STAGE1 = {
    "A": "ANSWERED",
    "B": "NONANSWER",
    "C": "REFUSES",
    "D": "UNCLEAR",
}
ANSWERED_STAGE = {"A": "DIRECT_REPLY", "B": "PARTIAL_REPLY"}
NONANSWER_STAGE = {"A": "DODGES_QUESTION", "B": "MAKES_POLITICAL_POINT", "C": "ATTACKS_QUESTION"}


def _examples_text(exemplars: pd.DataFrame) -> str:
    chunks = []
    for row in exemplars.itertuples(index=False):
        chunks.append(
            f"Question: {row.question_text}\nResponse: {row.response_text}\nLabel: {row.gold_label}"
        )
    return "\n\n".join(chunks)


def _base_case(context: str, question: str, response: str, exemplars: pd.DataFrame) -> str:
    return f"""
OPERATIONAL DEFINITIONS
DIRECT_REPLY = directly answers the requested information.
PARTIAL_REPLY = answers a meaningful part but leaves an important requested part unresolved.
DODGES_QUESTION = redirects away from the requested information without directly refusing.
MAKES_POLITICAL_POINT = substitutes partisan, campaign, or ideological messaging for the requested answer.
ATTACKS_QUESTION = attacks the question, questioner, or premise instead of answering.
REFUSES = explicitly declines, refuses, or states inability to answer.
UNCLEAR = too vague or ambiguous to assign a more specific category.

CALIBRATION EXAMPLES
{_examples_text(exemplars)}

CASE
Context: {context or '(none)'}
Question: {question}
Response: {response}
""".strip()


def stage1_prompt(base: str) -> str:
    return base + """

FIRST DECISION
A = ANSWERED: the response provides at least some requested information (DIRECT_REPLY or PARTIAL_REPLY).
B = NONANSWER: it avoids the requested information through a dodge, political point, or attack.
C = REFUSES: it explicitly declines or states inability to answer.
D = UNCLEAR: it is too vague/ambiguous to classify more specifically.

Return exactly A, B, C, or D.
"""


def answered_prompt(base: str) -> str:
    return base + """

The response has been placed in the ANSWERED branch.
A = DIRECT_REPLY
B = PARTIAL_REPLY
Return exactly A or B.
"""


def nonanswer_prompt(base: str) -> str:
    return base + """

The response has been placed in the NONANSWER branch.
A = DODGES_QUESTION
B = MAKES_POLITICAL_POINT
C = ATTACKS_QUESTION
Return exactly A, B, or C.
"""


def _existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return set(map(tuple, df[["model", "item_id", "repeat_idx"]].itertuples(index=False, name=None)))


def _balanced_class_sample(items: pd.DataFrame, max_items: int, seed: int) -> pd.DataFrame:
    labels = sorted(items["gold_label"].dropna().unique())
    base = max_items // len(labels)
    remainder = max_items % len(labels)
    parts = []
    for i, label in enumerate(labels):
        group = items[items["gold_label"] == label]
        take = min(len(group), base + (1 if i < remainder else 0))
        parts.append(deterministic_sample(group, take, seed + i, "item_id"))
    return pd.concat(parts, ignore_index=True)


def run_phase3(
    *,
    model: HFLocalModel,
    items: pd.DataFrame,
    exemplars: pd.DataFrame,
    output_path: str | Path,
    repeats: int = 1,
    max_items: int | None = None,
    seed: int = 42,
    checkpoint_every: int = 20,
    compression: str = "zstd",
) -> pd.DataFrame:
    output_path = Path(output_path)
    if max_items is not None and max_items < len(items):
        items = _balanced_class_sample(items, max_items, seed)
    existing = _existing_keys(output_path)
    pending: list[dict] = []
    progress = tqdm(total=len(items) * repeats, desc=f"Phase 3 | {model.short_name}")

    for item in items.itertuples(index=False):
        for repeat_idx in range(repeats):
            key = (model.short_name, int(item.item_id), repeat_idx)
            if key in existing:
                progress.update(1)
                continue
            base = _base_case(item.context if isinstance(item.context, str) else "", item.question_text, item.response_text, exemplars)
            record = {
                "model": model.short_name,
                "model_id": model.model_id,
                "model_revision": model.resolved_revision,
                "item_id": int(item.item_id),
                "repeat_idx": repeat_idx,
                "gold_label": item.gold_label,
                "stage1_code": None,
                "stage1_label": None,
                "stage2_code": None,
                "predicted_label": None,
                "confidence": np.nan,
                "prompt_hash": stable_hash(SYSTEM + "\n" + base),
                "latency_ms": np.nan,
                "error": None,
            }
            try:
                s1 = model.score_candidates(system=SYSTEM, prompt=stage1_prompt(base), candidates=list(STAGE1))
                p1 = s1.as_dict()
                c1 = max(p1, key=p1.get)
                stage1_label = STAGE1[c1]
                record.update({
                    "stage1_code": c1,
                    "stage1_label": stage1_label,
                    "stage1_confidence": float(p1[c1]),
                    "latency_ms": s1.latency_ms,
                    **{f"p_stage1_{STAGE1[c]}": float(p1[c]) for c in STAGE1},
                })

                if stage1_label == "ANSWERED":
                    s2 = model.score_candidates(system=SYSTEM, prompt=answered_prompt(base), candidates=list(ANSWERED_STAGE))
                    p2 = s2.as_dict()
                    c2 = max(p2, key=p2.get)
                    final = ANSWERED_STAGE[c2]
                    record.update({
                        "stage2_code": c2,
                        "predicted_label": final,
                        "confidence": float(p1[c1] * p2[c2]),
                        "latency_ms": float(record["latency_ms"] + s2.latency_ms),
                    })
                elif stage1_label == "NONANSWER":
                    s2 = model.score_candidates(system=SYSTEM, prompt=nonanswer_prompt(base), candidates=list(NONANSWER_STAGE))
                    p2 = s2.as_dict()
                    c2 = max(p2, key=p2.get)
                    final = NONANSWER_STAGE[c2]
                    record.update({
                        "stage2_code": c2,
                        "predicted_label": final,
                        "confidence": float(p1[c1] * p2[c2]),
                        "latency_ms": float(record["latency_ms"] + s2.latency_ms),
                    })
                elif stage1_label == "REFUSES":
                    record.update({"predicted_label": "REFUSES", "confidence": float(p1[c1])})
                else:
                    record.update({"predicted_label": "UNCLEAR", "confidence": float(p1[c1])})
            except Exception as exc:
                record["error"] = repr(exc)

            pending.append(record)
            if len(pending) >= checkpoint_every:
                checkpoint_records(pending, output_path, compression=compression)
                pending.clear()
            progress.update(1)
    if pending:
        checkpoint_records(pending, output_path, compression=compression)
    progress.close()
    return pd.read_parquet(output_path)


def evaluate_phase3(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    valid = raw[raw["error"].isna() & raw["predicted_label"].notna()].copy()
    overall_rows, class_rows, reports = [], [], {}
    for model, df in valid.groupby("model"):
        y_true, y_pred = df["gold_label"], df["predicted_label"]
        overall_rows.append({
            "model": model,
            "n": int(len(df)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=TAXONOMY, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=TAXONOMY, average="weighted", zero_division=0)),
            "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
            "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=TAXONOMY)),
            "mean_confidence": float(df["confidence"].mean()),
            "error_rate": float((raw[raw["model"] == model]["error"].notna()).mean()),
        })
        report = classification_report(y_true, y_pred, labels=TAXONOMY, output_dict=True, zero_division=0)
        reports[model] = report
        for label in TAXONOMY:
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
    valid = raw[raw["error"].isna() & raw["predicted_label"].notna()]
    out = {}
    for model, df in valid.groupby("model"):
        cm = confusion_matrix(df["gold_label"], df["predicted_label"], labels=TAXONOMY)
        out[model] = pd.DataFrame(cm, index=TAXONOMY, columns=TAXONOMY)
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
