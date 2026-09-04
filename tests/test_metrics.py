import numpy as np
import pandas as pd

from src.analysis import holm_adjust
from src.phase1_tsc import js_divergence, shannon_entropy, total_variation


def test_distribution_metrics_identity():
    p = np.array([0.2, 0.3, 0.5])
    assert abs(js_divergence(p, p)) < 1e-12
    assert abs(total_variation(p, p)) < 1e-12
    assert shannon_entropy(p) > 0


def test_holm_monotonic_in_sorted_order():
    p = pd.Series([0.01, 0.04, 0.03])
    adjusted = holm_adjust(p)
    order = p.sort_values().index
    vals = adjusted.loc[order].to_numpy()
    assert np.all(vals[1:] >= vals[:-1])
    assert np.all(vals <= 1)
