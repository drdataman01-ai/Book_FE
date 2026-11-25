"""Tests for Lab 9. Run with:  python tests/test_lab09.py   (or: pytest tests/)

Runs offline: the weight rules against closed forms and known cases,
Ledoit-Wolf shrinkage, hierarchical risk parity, the walk-forward timing and
turnover, the Sharpe-difference test, and the full lab on simulated data
(fast rules only).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.portfolio import (  # noqa: E402
    ledoit_wolf,
    sharpe_diff_test,
    w_equal_risk,
    w_hrp,
    w_inverse_vol,
    w_min_variance,
    w_tangency,
    walk_forward,
)


def _data(T=600, seed=0):
    rng = np.random.default_rng(seed)
    sd = np.array([0.02, 0.04, 0.06])
    return rng.standard_normal((T, 3)) * sd + np.array([0.004, 0.006, 0.008])


def test_min_variance_and_tangency_closed_forms():
    X = _data()
    cov = np.cov(X, rowvar=False)
    w = w_min_variance(X)
    w_cf = np.linalg.solve(cov, np.ones(3))
    assert np.allclose(w, w_cf / w_cf.sum())
    t = w_tangency(X, max_gross=100)
    t_cf = np.linalg.solve(cov, X.mean(0))
    assert np.allclose(t, t_cf / t_cf.sum())
    lo = w_min_variance(X, long_only=True)
    assert lo.min() >= -1e-10 and abs(lo.sum() - 1) < 1e-10


def test_risk_based_rules():
    X = _data()
    iv = w_inverse_vol(X)
    sd = X.std(0, ddof=1)
    assert np.allclose(iv, (1 / sd) / (1 / sd).sum())
    erc = w_equal_risk(X)
    cov = np.cov(X, rowvar=False)
    rc = erc * (cov @ erc)
    assert np.allclose(rc / rc.sum(), 1 / 3, atol=1e-3)  # equal risk contributions
    h = w_hrp(X)
    assert abs(h.sum() - 1) < 1e-12 and h.min() > 0 and h[0] > h[2]  # low-volatility asset gets more weight


def test_ledoit_wolf_shrinks_toward_identity():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 20)) * 0.05  # few observations, many assets: heavy shrinkage expected
    cov, delta = ledoit_wolf(X)
    assert 0.3 < delta <= 1.0
    S = np.cov(X, rowvar=False, bias=True)
    assert np.linalg.cond(cov) < np.linalg.cond(S)


def test_walk_forward_timing_and_turnover():
    idx = pd.period_range("2000-01", periods=5, freq="M")
    R = pd.DataFrame({"A": [0.1, 0.0, 0.1, 0.0, 0.2], "B": [0.0, 0.1, 0.0, 0.1, -0.1]}, index=idx)
    res = walk_forward(R, {"eq": lambda X: np.array([0.5, 0.5])}, window=2)
    r = res["eq"]
    assert list(r["ret"].index) == list(idx[2:]) and abs(r["ret"].iloc[0] - 0.05) < 1e-12
    # after month 3 (A +10%, B 0%) weights drift to 0.55/0.5 ... ; rebalancing back costs |drift|
    drift = np.array([0.55, 0.5]) / 1.05
    assert abs(r["turnover"][1] - np.abs(drift - 0.5).sum()) < 1e-12


def test_sharpe_difference_test():
    rng = np.random.default_rng(2)
    a = 0.01 + 0.04 * rng.standard_normal(5000)
    b = 0.5 * a + 0.00207 + 0.02 * rng.standard_normal(5000)  # same true Sharpe ratio (0.25), partly correlated
    assert sharpe_diff_test(a, b)["p"] > 0.05
    c = 0.002 + 0.04 * rng.standard_normal(5000)
    assert sharpe_diff_test(a, c)["p"] < 0.05


def test_full_lab_runs_on_simulated_data():
    import lab09

    with tempfile.TemporaryDirectory() as tmp:
        s = lab09.main(["--source", "sim", "--fast", "--out", tmp])
        assert len(list(Path(tmp).glob("fig9_*.png"))) == 5
        rep = (Path(tmp) / "lab09_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        m = {r["rule"]: r for r in s["main"]}
        assert m["Mean-variance (sample)"]["turnover_monthly"] > 5 * m["1/N"]["turnover_monthly"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
