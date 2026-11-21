"""Tests for Lab 11. Run with:  python tests/test_lab11.py   (or: pytest tests/)

Runs offline: the excess-return arithmetic, the Fama regression on data
where uncovered interest parity holds exactly and where it fails, the carry
portfolio's weights, timing and costs, and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.fx import carry_portfolio, excess_returns, fama_regressions  # noqa: E402


def _world(slope, n=1500, seed=0, k=4):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1900-01", periods=n, freq="M")
    us = pd.Series(0.04, idx)
    diff = np.zeros((n, k))
    for t in range(1, n):
        diff[t] = 0.95 * diff[t - 1] + 0.01 * rng.standard_normal(k)
    ds = np.zeros((n, k))
    ds[1:] = -slope * diff[:-1] / 12 + 0.0005 * rng.standard_normal((n - 1, k))
    cols = [f"C{i}" for i in range(k)]
    spot = pd.DataFrame(np.exp(np.cumsum(ds, axis=0)), idx, cols)
    rate = pd.DataFrame(0.04 + diff, idx, cols)
    return spot, rate, us


def test_excess_return_arithmetic():
    idx = pd.period_range("2000-01", periods=3, freq="M")
    spot = pd.DataFrame({"X": [1.0, 1.1, 1.0]}, idx)
    rate = pd.DataFrame({"X": [0.06, 0.06, 0.06]}, idx)
    er = excess_returns(spot, rate, pd.Series(0.03, idx))
    assert abs(er["rx"]["X"].iloc[1] - (0.03 / 12 + np.log(1.1))) < 1e-12


def test_fama_under_and_against_uip():
    s, r, u = _world(slope=1.0)
    er = excess_returns(s, r, u)
    f = fama_regressions(er["fx"], er["diff"]).set_index("currency")
    assert all(abs(f.loc[f"C{i}", "slope"] - 1) < 0.1 for i in range(4))
    s, r, u = _world(slope=-1.0, seed=1)
    er = excess_returns(s, r, u)
    f = fama_regressions(er["fx"], er["diff"]).set_index("currency")
    assert all(f.loc[f"C{i}", "slope"] < -0.8 for i in range(4))
    cp = carry_portfolio(er["rx"], er["diff"], n_side=1)
    assert cp["carry"].mean() > 0  # high-rate currency appreciates: carry pays


def test_carry_weights_timing_and_costs():
    idx = pd.period_range("2000-01", periods=4, freq="M")
    diff = pd.DataFrame({"A": [0.05] * 4, "B": [0.0] * 4, "C": [-0.05] * 4}, idx)
    rx = pd.DataFrame({"A": [0, 0.02, 0.02, 0.02], "B": [0, 0.0, 0.0, 0.0], "C": [0, -0.01, -0.01, -0.01]}, idx,
                      dtype=float)
    cp = carry_portfolio(rx, diff, n_side=1)
    assert abs(cp["carry"].iloc[0] - 0.03) < 1e-12 and cp.index[0] == idx[1]  # signal from the previous month
    cp2 = carry_portfolio(rx, diff, n_side=1, cost=0.001)
    assert abs(cp2["carry"].iloc[0] - (0.03 - 0.002)) < 1e-12 and abs(cp2["carry"].iloc[1] - 0.03) < 1e-12


def test_full_lab_runs_on_simulated_data():
    import lab11

    with tempfile.TemporaryDirectory() as tmp:
        s = lab11.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig11_*.png"))) == 5
        rep = (Path(tmp) / "lab11_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        carry = s["carry"][0]
        assert carry["mean_annual"] > 0 and carry["skewness"] < 0  # built-in premium with crashes


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
