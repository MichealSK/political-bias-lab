import pandas as pd

from src.sampling import deterministic_sample, deterministic_option_swap


def test_deterministic_sample_repeats_exactly():
    df = pd.DataFrame({"id": range(100)})
    a = deterministic_sample(df, 10, 42, "id")
    b = deterministic_sample(df, 10, 42, "id")
    assert a["id"].tolist() == b["id"].tolist()


def test_option_swap_is_deterministic():
    assert deterministic_option_swap(7, 2, 42) == deterministic_option_swap(7, 2, 42)
