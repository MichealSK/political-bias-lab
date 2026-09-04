from src.parsing import extract_json_object


def test_extract_json_from_fence():
    obj = extract_json_object('```json\n{"a": 1}\n```')
    assert obj == {"a": 1}
