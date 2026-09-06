import pandas as pd

from src.sampling import balanced_option_swap, deterministic_option_swap, deterministic_sample


def test_deterministic_sample_repeats_exactly():
    df = pd.DataFrame({"id": range(100)})
    a = deterministic_sample(df, 10, 42, "id")
    b = deterministic_sample(df, 10, 42, "id")
    assert a["id"].tolist() == b["id"].tolist()


def test_option_swap_is_deterministic():
    assert deterministic_option_swap(7, 2, 42) == deterministic_option_swap(7, 2, 42)


def test_balanced_swap_reverses_adjacent_repeats():
    assert balanced_option_swap(7, 0, 42) != balanced_option_swap(7, 1, 42)
    assert balanced_option_swap(7, 2, 42) != balanced_option_swap(7, 3, 42)
