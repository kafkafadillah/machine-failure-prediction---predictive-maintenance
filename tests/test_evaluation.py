import numpy as np

from mfp.evaluation import (bootstrap_ci, corrected_resampled_ttest, full_metrics,
                            select_threshold)
from mfp.monitoring import psi


def _toy(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.6 * y + rng.normal(0.1, 0.12, n), 0, 1)
    return y, p


def test_cost_formula():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.1, 0.8, 0.2])
    m = full_metrics(y, p, 0.5, c_fn=100, c_fp=10)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["expected_cost"] == 1 * 100 + (1 + 1) * 10


def test_threshold_respects_min_recall():
    y, p = _toy()
    thr, curve = select_threshold(y, p, c_fn=1, c_fp=1000, min_recall=0.9)
    assert curve.set_index("threshold").loc[thr, "recall"] >= 0.9


def test_bootstrap_ci_contains_estimate():
    y, p = _toy()
    ci = bootstrap_ci(y, p, 0.3, n_boot=100)
    assert ((ci.ci_low <= ci.estimate + 1e-9) & (ci.estimate <= ci.ci_high + 1e-9)).all()


def test_corrected_ttest_more_conservative_than_naive():
    rng = np.random.default_rng(1)
    a = 0.80 + rng.normal(0, 0.02, 15); b = a - 0.01 + rng.normal(0, 0.01, 15)
    _, p_corr = corrected_resampled_ttest(a, b, 6400, 1600)
    from scipy import stats
    assert p_corr >= stats.ttest_rel(a, b).pvalue


def test_psi_identical_is_zero_and_shift_detected():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-6
    assert psi(x, x + 1.0) > 0.25
