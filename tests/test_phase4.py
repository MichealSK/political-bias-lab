import pandas as pd

from src.phase4_pluralism import TfidfBiasRetriever, protocol_heuristics


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
