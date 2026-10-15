import pandas as pd
import pytest

from src.phase4_polibench_statements import (
    build_statement_prompt,
    load_polibench_data,
    prepare_polibench_cases,
)


def test_prompt_requests_strong_cases_and_transparency_note():
    prompt = build_statement_prompt("Should the policy be adopted?")
    assert "case FOR" in prompt
    assert "case AGAINST" in prompt
    assert "Transparency note" in prompt
    assert "do not force" in prompt.lower()
    assert "do not endorse" in prompt.lower()


def test_prepare_cases_preserves_statement_and_phase4_schema():
    raw = pd.DataFrame([
        {"id": 7, "statement": "First proposition"},
        {"id": 8, "statement": "Second proposition"},
    ])
    cases = prepare_polibench_cases(raw)
    assert cases["case_id"].tolist() == [7, 8]
    assert cases["statement"].tolist() == ["First proposition", "Second proposition"]
    assert {"topic", "user_question", "position_a", "position_b"}.issubset(cases.columns)
    assert all("case FOR" in prompt and "case AGAINST" in prompt for prompt in cases["user_question"])


def test_load_polibench_validates_schema_and_duplicates(tmp_path):
    valid = tmp_path / "valid.csv"
    pd.DataFrame([{"id": 1, "statement": "A proposition"}]).to_csv(valid, index=False)
    loaded = load_polibench_data(valid)
    assert loaded.to_dict("records") == [{"id": 1, "statement": "A proposition"}]

    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame([
        {"id": 1, "statement": "A proposition"},
        {"id": 1, "statement": "Another proposition"},
    ]).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_polibench_data(duplicate)


def test_uploaded_project_dataset_has_62_valid_statements():
    from pathlib import Path
    data_path = Path(__file__).resolve().parents[1] / "data" / "polibench_statements.csv"
    if data_path.exists():
        df = load_polibench_data(data_path)
        assert len(df) == 62
        assert df["id"].is_unique
        assert (df["statement"].str.len() > 0).all()


def test_generation_runs_both_conditions_with_mocked_model_and_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.phase4_polibench_statements as module

    cases = prepare_polibench_cases(pd.DataFrame([
        {"id": 1, "statement": "A policy proposition"},
        {"id": 2, "statement": "Another proposition"},
    ]))
    exemplars = pd.DataFrame([
        {"exemplar_id": 1, "input_text": "one-sided argument", "bias_label": "one-sidedness",
         "corrected_response": "consider counterarguments and evidence"},
        {"exemplar_id": 2, "input_text": "false equivalence", "bias_label": "false equivalence",
         "corrected_response": "do not force symmetry when evidence differs"},
    ])
    saved_records = []
    calls = []

    class FakeModel:
        short_name = "fake"
        model_id = "fake/model"
        resolved_revision = "test-revision"

        def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                text="CASE FOR\\nArgument.\\nCASE AGAINST\\nCounterargument.\\nTransparency note\\nLimits.",
                latency_ms=1.0,
                input_tokens=20,
                output_tokens=10,
            )

    def fake_checkpoint(records, path, compression="zstd"):
        saved_records.extend(dict(r) for r in records)
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    monkeypatch.setattr(module, "checkpoint_records", fake_checkpoint)
    monkeypatch.setattr(module.pd, "read_parquet", lambda _path: pd.DataFrame(saved_records))

    out = module.run_phase4_polibench_generation(
        model=FakeModel(),
        cases=cases,
        exemplars=exemplars,
        output_path=tmp_path / "fake.parquet",
        max_cases=None,
        checkpoint_every=2,
    )
    assert len(out) == 4
    assert out["condition"].value_counts().to_dict() == {
        "baseline": 2,
        "adaptive_pluralism": 2,
    }
    assert all("case FOR" in call["prompt"] and "case AGAINST" in call["prompt"] for call in calls)
    assert all("Transparency note" in call["prompt"] for call in calls)
