"""Phase 3: QEvasion fine-grained response-evasion and clarity evaluation.

This module keeps the original annotation rows intact, performs inference once per
unique question/answer input, and scores predictions against the provided human
annotations. The GPT-3.5 summary/prediction columns are deliberately never used as
model inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROMPT_VERSION = "qaevasion_zero_shot_v1"

LABELS: tuple[str, ...] = (
    "1.1 Explicit",
    "1.2 Implicit",
    "2.1 Dodging",
    "2.2 Deflection",
    "2.3 Partial/half-answer",
    "2.4 General",
    "2.6 Declining to answer",
    "2.7 Claims ignorance",
    "2.8 Clarification",
)

LABEL_DESCRIPTIONS: dict[str, str] = {
    "1.1 Explicit": "The requested information is explicitly stated in the expected form.",
    "1.2 Implicit": "The information is conveyed indirectly; the reader must infer it or it is not in the expected form.",
    "2.1 Dodging": "The answer ignores the question altogether and does not meaningfully address it.",
    "2.2 Deflection": "The answer starts on topic but shifts focus to a different point instead of answering.",
    "2.3 Partial/half-answer": "The answer provides only one specific component of the requested information.",
    "2.4 General": "The answer is relevant but too general or lacks the requested specificity.",
    "2.6 Declining to answer": "The speaker directly or indirectly refuses to answer at that time.",
    "2.7 Claims ignorance": "The speaker claims or admits not to know the requested information.",
    "2.8 Clarification": "The speaker does not provide the requested information and asks for clarification.",
}

# The higher-level mapping from the dataset's published two-level taxonomy.
CLARITY_MAP: dict[str, str] = {
    "1.1 Explicit": "Clear Reply",
    "1.2 Implicit": "Ambivalent",
    "2.1 Dodging": "Ambivalent",
    "2.2 Deflection": "Ambivalent",
    "2.3 Partial/half-answer": "Ambivalent",
    "2.4 General": "Ambivalent",
    "2.6 Declining to answer": "Clear Non-Reply",
    "2.7 Claims ignorance": "Clear Non-Reply",
    "2.8 Clarification": "Clear Non-Reply",
}
CLARITY_LABELS: tuple[str, ...] = ("Clear Reply", "Ambivalent", "Clear Non-Reply")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "title", "date", "president", "url", "interview_question",
    "interview_answer", "gpt3.5_summary", "gpt3.5_prediction",
    "question", "label", "annotator_id",
)
INPUT_COLUMNS: tuple[str, ...] = ("question", "interview_answer")

SYSTEM_PROMPT = """You are a careful research annotator for political interview response clarity.
Classify how the answer relates to the specific question. Do not judge whether the
speaker is politically right or wrong, and do not infer intent. Use only the question
and answer supplied. Choose exactly one of these nine labels:

1.1 Explicit — The requested information is explicitly stated in the expected form.
1.2 Implicit — The information is conveyed indirectly; the reader must infer it or it is not in the expected form.
2.1 Dodging — The answer ignores the question altogether and does not meaningfully address it.
2.2 Deflection — The answer starts on topic but shifts focus to a different point instead of answering.
2.3 Partial/half-answer — The answer provides only one specific component of the requested information.
2.4 General — The answer is relevant but too general or lacks the requested specificity.
2.6 Declining to answer — The speaker directly or indirectly refuses to answer at that time.
2.7 Claims ignorance — The speaker claims or admits not to know the requested information.
2.8 Clarification — The speaker does not provide the requested information and asks for clarification.

Return only the label exactly as written, including its numeric code. Do not provide an
explanation, summary, or any other text. If the answer could fit multiple labels, select
the best single label based on the dataset definitions above."""

# Most specific/longest labels are checked first so parser aliases don't collide.
_LABEL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(re.escape(x) for x in sorted(LABELS, key=len, reverse=True))
    + r")(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def make_item_id(url: Any, question: Any, answer: Any) -> str:
    """Stable ID for an inference input; annotation IDs are kept separately."""
    raw = "\x1f".join((_normalise_text(url), _normalise_text(question), _normalise_text(answer)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def load_qaevasion_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate the supplied QEvasion CSV without altering its gold labels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QEvasion dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"QEvasion CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError("QEvasion CSV contains no rows.")

    for col in REQUIRED_COLUMNS:
        # Keep text intact except for trimming outer whitespace; don't normalize the source file.
        if col in df.columns and col != "annotator_id":
            df[col] = df[col].fillna("").astype(str).str.strip()
    df["label"] = df["label"].str.replace(r"\s+", " ", regex=True)
    unknown = sorted(set(df["label"].dropna()) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected gold labels found: {unknown}. Expected exactly the QEvasion taxonomy.")
    if (df["question"].str.len() == 0).any() or (df["interview_answer"].str.len() == 0).any():
        raise ValueError("Some rows have an empty question or interview_answer.")

    df.insert(0, "source_row_id", np.arange(len(df), dtype=np.int64))
    df["item_id"] = [make_item_id(u, q, a) for u, q, a in zip(df["url"], df["question"], df["interview_answer"])]
    df["clarity_label"] = df["label"].map(CLARITY_MAP)
    return df


def sample_annotations(df: pd.DataFrame, profile: str = "paper", seed: int = 42) -> pd.DataFrame:
    """Sample annotation rows by fine-grained label; paper profile uses all rows.

    Smoke/pilot sampling is label-stratified to ensure rare categories are exercised.
    Sampling happens at annotation-row level. If multiple annotators label the same
    QA input, the model is still run once for that input and each selected annotation
    remains available for evaluation.
    """
    profile = profile.lower().strip()
    if profile == "paper":
        return df.copy().reset_index(drop=True)
    per_label = {"smoke": 3, "pilot": 20}
    if profile not in per_label:
        raise ValueError(f"Unknown profile {profile!r}; use smoke, pilot, or paper.")
    n = per_label[profile]
    pieces: list[pd.DataFrame] = []
    for i, label in enumerate(LABELS):
        subset = df[df["label"] == label]
        if subset.empty:
            raise ValueError(f"No rows found for required class {label!r}.")
        pieces.append(subset.sample(n=min(n, len(subset)), random_state=seed + i))
    return pd.concat(pieces, ignore_index=True).sort_values("source_row_id").reset_index(drop=True)


def unique_inference_items(annotation_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one inference row per unique question/answer pair."""
    cols = ["item_id", "question", "interview_answer", "url", "title", "date", "president"]
    items = annotation_rows[cols].drop_duplicates("item_id", keep="first").copy()
    return items.sort_values("item_id").reset_index(drop=True)


def build_user_prompt(question: str, answer: str) -> str:
    return f"QUESTION:\n{question.strip()}\n\nINTERVIEW ANSWER:\n{answer.strip()}\n\nRequired output: one label only."


def parse_label(output: Any) -> str | None:
    """Parse a generated label; return None rather than fabricating a class on failure."""
    if output is None:
        return None
    text = str(output).strip()
    if not text:
        return None
    # Prefer the first line if it is itself a label (optionally quoted or Markdown-wrapped).
    first = text.splitlines()[0].strip().strip("`*_\"' ")
    for label in LABELS:
        if first.casefold() == label.casefold():
            return label
        if re.match(r"^label\s*[:=-]\s*" + re.escape(label) + r"(?:\s|$|[.!])", first, flags=re.I):
            return label
    match = _LABEL_PATTERN.search(text)
    if match:
        found = match.group(0).casefold()
        for label in LABELS:
            if label.casefold() == found:
                return label
    # Allow a label written without its code, but only if it is unambiguous.
    aliases = {
        "explicit": "1.1 Explicit",
        "implicit": "1.2 Implicit",
        "dodging": "2.1 Dodging",
        "deflection": "2.2 Deflection",
        "partial/half-answer": "2.3 Partial/half-answer",
        "partial": "2.3 Partial/half-answer",
        "general": "2.4 General",
        "declining to answer": "2.6 Declining to answer",
        "claims ignorance": "2.7 Claims ignorance",
        "clarification": "2.8 Clarification",
    }
    stripped = re.sub(r"^[\s>*#\-]*(?:label|classification|category)?\s*[:=\-]*\s*", "", first, flags=re.I)
    alias = aliases.get(stripped.casefold().rstrip(". "))
    return alias


def _truncate_token_ids(tokenizer: Any, text: str, max_tokens: int, head_fraction: float = 0.75) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    if max_tokens <= 0:
        return ""
    head_n = max(1, int(max_tokens * head_fraction))
    tail_n = max(0, max_tokens - head_n)
    kept = ids[:head_n] + (ids[-tail_n:] if tail_n else [])
    return tokenizer.decode(kept, skip_special_tokens=True)


def _render_prompt(tokenizer: Any, question: str, answer: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, answer)},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            combined = SYSTEM_PROMPT + "\n\n" + build_user_prompt(question, answer)
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": combined}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
    return f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{build_user_prompt(question, answer)}\n\nASSISTANT:\n"


def _make_inputs(tokenizer: Any, question: str, answer: str, max_input_tokens: int) -> tuple[str, dict[str, Any], bool]:
    """Build bounded prompt, preserving the start and end of long interview answers."""
    question = str(question).strip()
    answer = str(answer).strip()
    truncated = False
    # The dataset's question field is short; only trim it if the source changes drastically.
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    max_q = max(64, min(512, max_input_tokens // 3))
    if len(q_tokens) > max_q:
        question = _truncate_token_ids(tokenizer, question, max_q, head_fraction=0.85)
        q_tokens = tokenizer.encode(question, add_special_tokens=False)
        truncated = True

    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_budget = max(128, max_input_tokens - len(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)) - len(q_tokens) - 160)
    if len(answer_ids) > answer_budget:
        answer = _truncate_token_ids(tokenizer, answer, answer_budget, head_fraction=0.75)
        truncated = True

    # If the chat template adds more tokens than estimated, shrink the answer until safe.
    while True:
        rendered = _render_prompt(tokenizer, question, answer)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=False, add_special_tokens=False)
        if int(inputs["input_ids"].shape[-1]) <= max_input_tokens:
            return rendered, inputs, truncated
        current_len = len(tokenizer.encode(answer, add_special_tokens=False))
        new_budget = int(current_len * 0.85)
        if new_budget < 32 or new_budget >= current_len:
            # Final safety fallback: retain question/system prompt and a short answer excerpt.
            answer = _truncate_token_ids(tokenizer, answer, max(16, current_len // 2), head_fraction=0.75)
            rendered = _render_prompt(tokenizer, question, answer)
            inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_input_tokens, add_special_tokens=False)
            truncated = True
            return rendered, inputs, truncated
        answer = _truncate_token_ids(tokenizer, answer, new_budget, head_fraction=0.75)
        truncated = True


def load_hf_model(model_id: str, use_4bit: bool = True, revision: str | None = None):
    """Load an open causal LM; use 4-bit weights when CUDA and bitsandbytes are available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, revision=revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"device_map": "auto" if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "4-bit loading was requested but bitsandbytes/quantization could not be initialized. "
                    "Install requirements-colab.txt or rerun with --no-4bit."
                ) from exc
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs)
    model.eval()
    return model, tokenizer


def generate_one(model: Any, tokenizer: Any, question: str, answer: str,
                 max_input_tokens: int = 4096, max_new_tokens: int = 16) -> dict[str, Any]:
    import torch

    rendered, inputs, truncated = _make_inputs(tokenizer, question, answer, max_input_tokens)
    embedding_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(embedding_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "prediction_raw": raw,
        "predicted_label": parse_label(raw),
        "parse_failure": parse_label(raw) is None,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": int(new_ids.shape[-1]),
        "answer_truncated": bool(truncated),
        "prompt_version": PROMPT_VERSION,
    }


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")
    return slug[:80] or "model"


def run_inference(model_id: str, items: pd.DataFrame, output_path: str | Path,
                  max_input_tokens: int = 4096, max_new_tokens: int = 16,
                  checkpoint_every: int = 10, use_4bit: bool = True,
                  resume: bool = True, revision: str | None = None) -> pd.DataFrame:
    """Run resumable inference once per unique item and checkpoint to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior: pd.DataFrame = pd.DataFrame()
    if resume and output_path.exists():
        try:
            prior = pd.read_parquet(output_path)
            prior = prior[prior.get("model_id", pd.Series(dtype=str)) == model_id]
            if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision) and "model_revision" in prior.columns:
                prior = prior[prior["model_revision"].astype(str).str.casefold() == revision.casefold()]
        except Exception:
            prior = pd.DataFrame()
    done_ids = set()
    if not prior.empty and "item_id" in prior.columns:
        # Error rows are retried; completed valid or unparseable generations are retained.
        if "error" in prior.columns:
            done_ids = set(prior.loc[prior["error"].fillna("").eq(""), "item_id"].astype(str))
        else:
            done_ids = set(prior["item_id"].astype(str))
    results = {str(row["item_id"]): row for row in prior.to_dict(orient="records") if str(row.get("item_id")) in done_ids}
    pending = items[~items["item_id"].astype(str).isin(done_ids)].copy()
    if pending.empty:
        return pd.DataFrame(list(results.values())).sort_values("item_id").reset_index(drop=True)

    model, tokenizer = load_hf_model(model_id, use_4bit=use_4bit, revision=revision)
    try:
        for idx, row in enumerate(pending.to_dict(orient="records"), start=1):
            item_id = str(row["item_id"])
            base = {
                "item_id": item_id,
                "model_id": model_id,
                "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
                "requested_revision": revision or "main (resolved commit recorded after load)",
                "question": row["question"],
                "url": row.get("url", ""),
                "president": row.get("president", ""),
                "profile": "",
                "error": "",
                "created_unix": time.time(),
            }
            try:
                generated = generate_one(
                    model, tokenizer, row["question"], row["interview_answer"],
                    max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                )
                base.update(generated)
            except Exception as exc:  # retain a row for diagnosis and retry on next run
                base.update({
                    "prediction_raw": "",
                    "predicted_label": None,
                    "parse_failure": True,
                    "input_tokens": None,
                    "output_tokens": None,
                    "answer_truncated": None,
                    "prompt_version": PROMPT_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            results[item_id] = base
            if idx % max(1, checkpoint_every) == 0 or idx == len(pending):
                pd.DataFrame(list(results.values())).sort_values("item_id").to_parquet(output_path, index=False)
    finally:
        del model, tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    out = pd.DataFrame(list(results.values())).sort_values("item_id").reset_index(drop=True)
    out.to_parquet(output_path, index=False)
    return out


def _classification_metrics(y_true: Iterable[str], y_pred: Iterable[str], labels: tuple[str, ...]) -> dict[str, Any]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return {"n": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None,
                "weighted_f1": None, "macro_precision": None, "macro_recall": None, "per_class": {}}
    report = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "per_class": {label: report.get(label, {}) for label in labels},
    }


def annotation_agreement(full_annotations: pd.DataFrame) -> pd.DataFrame:
    """Summarize multiple human annotations for identical question/answer inputs."""
    rows = []
    for item_id, group in full_annotations.groupby("item_id", sort=True):
        counts = Counter(group["label"].tolist())
        max_count = max(counts.values())
        modes = sorted(label for label, n in counts.items() if n == max_count)
        majority = modes[0] if len(modes) == 1 else None
        total = sum(counts.values())
        probs = [n / total for n in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        rows.append({
            "item_id": item_id,
            "annotation_count": total,
            "annotator_count": group["annotator_id"].nunique(),
            "majority_gold_label": majority,
            "majority_tie": len(modes) != 1,
            "majority_fraction": max_count / total,
            "label_entropy_bits": entropy,
            "label_votes_json": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)


def evaluate_predictions(annotation_rows: pd.DataFrame, all_annotations: pd.DataFrame,
                         item_predictions: pd.DataFrame, output_dir: str | Path,
                         model_id: str, profile: str) -> dict[str, Any]:
    """Save annotation-level 9-way and 3-way metrics plus item-majority secondary metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_cols = ["item_id", "predicted_label", "prediction_raw", "parse_failure", "error",
                 "input_tokens", "output_tokens", "answer_truncated", "model_id", "model_revision", "prompt_version"]
    pred_cols = [c for c in pred_cols if c in item_predictions.columns]
    joined = annotation_rows.merge(item_predictions[pred_cols].drop_duplicates("item_id"), on="item_id", how="left")
    joined["predicted_clarity"] = joined["predicted_label"].map(CLARITY_MAP)
    joined["gold_clarity"] = joined["label"].map(CLARITY_MAP)
    valid = joined[joined["predicted_label"].isin(LABELS)].copy()
    coverage = float(len(valid) / len(joined)) if len(joined) else 0.0

    fine_metrics = _classification_metrics(valid["label"], valid["predicted_label"], LABELS)
    clarity_valid = valid[valid["predicted_clarity"].isin(CLARITY_LABELS)]
    clarity_metrics = _classification_metrics(clarity_valid["gold_clarity"], clarity_valid["predicted_clarity"], CLARITY_LABELS)

    # Secondary item-majority metric uses all supplied annotations for selected unique items.
    agreement = annotation_agreement(all_annotations)
    selected_ids = set(annotation_rows["item_id"].astype(str))
    agreement_selected = agreement[agreement["item_id"].astype(str).isin(selected_ids)]
    item_join = agreement_selected.merge(item_predictions, on="item_id", how="left", suffixes=("", "_pred"))
    item_valid = item_join[item_join["majority_gold_label"].isin(LABELS) & item_join["predicted_label"].isin(LABELS)]
    item_majority_metrics = _classification_metrics(item_valid["majority_gold_label"], item_valid["predicted_label"], LABELS)

    report = {
        "dataset": "QEvasion / QAEvasion.csv",
        "profile": profile,
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "resolved_model_revisions": sorted(item_predictions.get("model_revision", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "taxonomy": "9-way evasion labels; 3-way clarity mapping from the published taxonomy",
        "n_annotation_rows_sampled": int(len(annotation_rows)),
        "n_unique_items_sampled": int(annotation_rows["item_id"].nunique()),
        "n_valid_annotation_predictions": int(len(valid)),
        "prediction_coverage": coverage,
        "parse_failure_count": int(joined["predicted_label"].isna().sum()),
        "generation_error_count": int(joined.get("error", pd.Series("", index=joined.index)).fillna("").ne("").sum()),
        "fine_grained_9_way": fine_metrics,
        "clarity_3_way": clarity_metrics,
        "item_majority_9_way_secondary": {
            **item_majority_metrics,
            "n_items_with_unambiguous_majority": int(len(item_valid)),
            "n_selected_items_with_majority_tie": int(agreement_selected["majority_tie"].sum()),
        },
    }

    joined.to_parquet(output_dir / "phase3_qaevasion_annotation_predictions.parquet", index=False)
    item_predictions.to_parquet(output_dir / "phase3_qaevasion_item_predictions.parquet", index=False)
    agreement.to_csv(output_dir / "phase3_qaevasion_annotation_agreement.csv", index=False)
    (output_dir / "phase3_qaevasion_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(valid):
        per_class = classification_report(valid["label"], valid["predicted_label"], labels=list(LABELS),
                                          target_names=list(LABELS), output_dict=True, zero_division=0)
        pd.DataFrame(per_class).T.to_csv(output_dir / "phase3_qaevasion_per_class.csv", index=True)
        cm = confusion_matrix(valid["label"], valid["predicted_label"], labels=list(LABELS))
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(output_dir / "phase3_qaevasion_confusion_matrix.csv")
        clarity_cm = confusion_matrix(clarity_valid["gold_clarity"], clarity_valid["predicted_clarity"], labels=list(CLARITY_LABELS)) if len(clarity_valid) else np.zeros((3, 3), dtype=int)
        pd.DataFrame(clarity_cm, index=CLARITY_LABELS, columns=CLARITY_LABELS).to_csv(output_dir / "phase3_qaevasion_clarity_confusion_matrix.csv")
    return report


def run_phase3_qaevasion(input_path: str | Path, output_dir: str | Path,
                         model_id: str, profile: str = "smoke", seed: int = 42,
                         max_input_tokens: int = 4096, max_new_tokens: int = 16,
                         checkpoint_every: int = 10, use_4bit: bool = True,
                         resume: bool = True, revision: str | None = None) -> dict[str, Any]:
    """End-to-end Phase 3 run for the supplied QEvasion annotations."""
    full = load_qaevasion_csv(input_path)
    selected = sample_annotations(full, profile=profile, seed=seed)
    items = unique_inference_items(selected)
    root_output_dir = Path(output_dir)
    slug = _safe_slug(model_id)
    model_output_dir = root_output_dir / slug / profile
    model_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = model_output_dir / f"phase3_qaevasion_{slug}_{profile}_item_predictions.parquet"
    predictions = run_inference(
        model_id, items, prediction_path,
        max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
        checkpoint_every=checkpoint_every, use_4bit=use_4bit, resume=resume, revision=revision,
    )
    # Mark the profile for traceability without affecting the prompt.
    predictions["profile"] = profile
    predictions.to_parquet(prediction_path, index=False)
    return evaluate_predictions(selected, full, predictions, model_output_dir, model_id, profile)


def run_phase3_from_project_config(cfg: dict[str, Any], model_cfg: dict[str, Any],
                                  root: str | Path = ".", profile: str | None = None) -> dict[str, Any]:
    """Adapter for the existing project's YAML-driven pipeline.

    This is the recommended call to use from the project orchestrator when replacing
    the former synthetic Phase 3 stage. Model config may use common Hugging Face key
    names (`model_id`, `hf_id`, `repo_id`, `model_name`, or `id`).
    """
    root = Path(root)
    phase_cfg = cfg.get("phase3", {}) or {}
    model_id = (model_cfg.get("model_id") or model_cfg.get("hf_id") or
                model_cfg.get("repo_id") or model_cfg.get("model_name") or model_cfg.get("id"))
    if not model_id:
        raise ValueError("model_cfg must provide model_id, hf_id, repo_id, model_name, or id")
    selected_profile = profile or phase_cfg.get("profile") or cfg.get("profile") or "smoke"
    input_path = Path(phase_cfg.get("dataset_path", "data/QAEvasion.csv"))
    if not input_path.is_absolute():
        input_path = root / input_path
    output_dir = Path(phase_cfg.get("output_dir", "results/phase3_qaevasion"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return run_phase3_qaevasion(
        input_path=input_path,
        output_dir=output_dir,
        model_id=str(model_id),
        profile=str(selected_profile),
        seed=int(phase_cfg.get("seed", cfg.get("seed", 42))),
        max_input_tokens=int(phase_cfg.get("max_input_tokens", 4096)),
        max_new_tokens=int(phase_cfg.get("max_new_tokens", 16)),
        checkpoint_every=int(phase_cfg.get("checkpoint_every", 10)),
        use_4bit=bool(phase_cfg.get("use_4bit", True)),
        resume=bool(phase_cfg.get("resume", True)),
        revision=phase_cfg.get("revision"),
    )
