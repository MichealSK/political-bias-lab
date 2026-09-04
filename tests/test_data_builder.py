from src.data_builder import build_fictional_entities, build_sentiment_templates


def test_logical_universe_dimensions():
    entities = build_fictional_entities(500)
    templates = build_sentiment_templates(50_000, seed=42)
    assert len(entities) == 500
    assert len(templates) == 50_000
    assert len(entities) * len(templates) == 25_000_000
    assert set(templates["gold_sentiment"]) == {"negative", "neutral", "positive"}
