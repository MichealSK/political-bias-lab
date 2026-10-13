from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3_evasion import (  # noqa: E402
    CLARITY_MAP,
    LABELS,
    annotation_agreement,
    build_user_prompt,
    evaluate_predictions,
    load_qaevasion_csv,
    make_item_id,
    parse_label,
    sample_annotations,
    unique_inference_items,
)

DATA = ROOT / "data" / "QAEvasion.csv"


def test_dataset_schema_and_label_inventory():
    df = load_qaevasion_csv(DATA)
    assert len(df) == 3448
    assert set(df["label"].unique()) == set(LABELS)
    assert df["url"].nunique() == 287
    assert df["president"].nunique() == 4
    assert df["item_id"].nunique() < len(df)  # some inputs have multiple annotation rows
    assert set(df["clarity_label"].unique()) == {"Clear Reply", "Ambivalent", "Clear Non-Reply"}


def test_parser_exact_and_verbose_outputs():
    assert parse_label("1.1 Explicit") == "1.1 Explicit"
    assert parse_label("Label: 2.3 Partial/half-answer") == "2.3 Partial/half-answer"
    assert parse_label("2.7 Claims ignorance\nNo explanation") == "2.7 Claims ignorance"
    assert parse_label("I cannot decide") is None
    assert parse_label("") is None


def test_stratified_samples_are_deterministic_and_include_all_labels():
    df = load_qaevasion_csv(DATA)
    a = sample_annotations(df, "smoke", seed=42)
    b = sample_annotations(df, "smoke", seed=42)
    assert a["source_row_id"].tolist() == b["source_row_id"].tolist()
    assert len(a) == 27
    assert set(a["label"].unique()) == set(LABELS)
    assert len(sample_annotations(df, "pilot", seed=42)) == 180
    assert len(sample_annotations(df, "paper", seed=42)) == len(df)


def test_unique_inference_items_deduplicate_only_model_inputs():
    df = load_qaevasion_csv(DATA)
    sampled = sample_annotations(df, "paper")
    items = unique_inference_items(sampled)
    assert items["item_id"].is_unique
    assert len(items) == df["item_id"].nunique()
    # Original annotations remain separate for evaluation.
    assert len(sampled) == 3448


def test_prompt_uses_only_question_and_answer():
    prompt = build_user_prompt("Was the bill signed?", "Yes, it was signed yesterday.")
    assert "Was the bill signed?" in prompt
    assert "Yes, it was signed yesterday." in prompt
    assert "gpt3.5_prediction" not in prompt
    assert "gpt3.5_summary" not in prompt
    assert "1.1 Explicit" not in prompt
    assert "2.7 Claims ignorance" not in prompt


def test_stable_item_id_normalizes_whitespace_but_preserves_case():
    one = make_item_id("https://example.com", "What happened?", "The answer.\nNext line.")
    two = make_item_id("https://example.com", " What happened? ", "The answer. Next line.")
    case_variant = make_item_id("https://example.com", "what happened?", "The answer. Next line.")
    assert one == two
    assert one != case_variant


def test_published_clarity_mapping_covers_all_fine_grained_labels():
    assert set(CLARITY_MAP) == set(LABELS)
    assert CLARITY_MAP["1.1 Explicit"] == "Clear Reply"
    assert CLARITY_MAP["2.2 Deflection"] == "Ambivalent"
    assert CLARITY_MAP["2.7 Claims ignorance"] == "Clear Non-Reply"


def test_annotation_agreement_reports_ties_without_breaking_them():
    frame = pd.DataFrame({
        "item_id": ["a", "a", "b", "b", "b"],
        "label": ["1.1 Explicit", "2.1 Dodging", "1.1 Explicit", "1.1 Explicit", "1.2 Implicit"],
        "annotator_id": [85, 86, 85, 86, 89],
    })
    result = annotation_agreement(frame).set_index("item_id")
    assert bool(result.loc["a", "majority_tie"])
    assert pd.isna(result.loc["a", "majority_gold_label"])
    assert not bool(result.loc["b", "majority_tie"])
    assert result.loc["b", "majority_gold_label"] == "1.1 Explicit"
    assert result.loc["b", "majority_fraction"] == 2 / 3


def test_evaluation_generates_9way_and_3way_metrics_without_gpu(tmp_path, monkeypatch):
    df = load_qaevasion_csv(DATA)
    sampled = sample_annotations(df, "smoke", seed=42)
    items = unique_inference_items(sampled)
    first_labels = sampled.drop_duplicates("item_id").set_index("item_id")["label"]
    preds = items[["item_id"]].copy()
    preds["predicted_label"] = preds["item_id"].map(first_labels)
    preds["prediction_raw"] = preds["predicted_label"]
    preds["parse_failure"] = False
    preds["error"] = ""
    preds["input_tokens"] = 100
    preds["output_tokens"] = 3
    preds["answer_truncated"] = False
    preds["model_id"] = "test/dummy"
    preds["model_revision"] = "test-revision"
    preds["prompt_version"] = "test-prompt"

    # The container used for packaging does not ship pyarrow; mock only the Parquet writer.
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("mock parquet"))
    report = evaluate_predictions(sampled, df, preds, tmp_path, "test/dummy", "smoke")
    assert report["n_annotation_rows_sampled"] == 27
    assert report["prediction_coverage"] == 1.0
    assert report["fine_grained_9_way"]["n"] == 27
    assert report["clarity_3_way"]["n"] == 27
    assert (tmp_path / "phase3_qaevasion_metrics.json").exists()
    assert (tmp_path / "phase3_qaevasion_confusion_matrix.csv").exists()
