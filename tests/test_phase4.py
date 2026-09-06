import pandas as pd

from src.phase4_pluralism import TfidfBiasRetriever, judge_quality_summary, protocol_heuristics


def test_tfidf_retrieval():
    df = pd.DataFrame([
        {"exemplar_id": 1, "input_text": "equal weight false claim", "bias_label": "false equivalence", "corrected_response": "do not force symmetry"},
        {"exemplar_id": 2, "input_text": "attack supporters", "bias_label": "stereotyping", "corrected_response": "avoid group stereotypes"},
    ])
    r = TfidfBiasRetriever(df).retrieve("false equivalence and symmetry", k=1)
    assert r[0]["exemplar_id"] == 1


def test_protocol_heuristics():
    h = protocol_heuristics("Both sides may disagree. Transparency note: evidence is uncertain.")
    assert h["has_transparency_note"] is True
    assert h["uncertainty_marker_count"] >= 1


def test_judge_order_consistency_maps_x_y_back_to_conditions():
    # Repeat 0 prefers Y=adaptive; repeat 1 prefers X=adaptive after order reversal.
    df = pd.DataFrame([
        {"judge_model": "judge", "generator_model": "gen", "case_id": 1, "repeat_idx": 0,
         "condition_x": "baseline", "condition_y": "adaptive_pluralism", "overall_preference": "Y",
         "parse_success": True, "parse_attempts": 1},
        {"judge_model": "judge", "generator_model": "gen", "case_id": 1, "repeat_idx": 1,
         "condition_x": "adaptive_pluralism", "condition_y": "baseline", "overall_preference": "X",
         "parse_success": True, "parse_attempts": 1},
    ])
    q = judge_quality_summary(df)
    assert float(q.iloc[0]["parse_success_rate"]) == 1.0
    assert float(q.iloc[0]["order_consistency"]) == 1.0
