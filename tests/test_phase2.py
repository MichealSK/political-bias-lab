from types import SimpleNamespace

from src.phase2_persona import render_persona


def test_persona_binds_to_semantic_policy_text():
    item = SimpleNamespace(option_for_a="increase funding", option_for_b="decrease funding")
    persona = {"prompt": "Advocate assigned direction.", "alignment": "semantic_b"}
    rendered = render_persona(persona, item)
    assert "decrease funding" in rendered
    assert "increase funding" not in rendered
