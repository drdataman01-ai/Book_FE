"""Tests for Lab 10. Run with:  python tests/test_lab10.py   (or: pytest tests/)

Runs offline: bond returns from yields against hand calculation, exceedance
correlations and tail dependence against a bivariate normal, the
Forbes-Rigobon correction on data with a known correlation, and the full lab
on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.corr import (bond_returns_from_yields, exceedance_corr, forbes_rigobon, normal_exceedance,  # noqa: E402
                          par_bond_price, tail_dependence)


def test_bond_returns():
    assert abs(par_bond_price(0.04, 0.04, 10) - 1) < 1e-12
    idx = pd.period_range("2000-01", periods=3, freq="M")
    r = bond_returns_from_yields(pd.Series([4.0, 4.0, 5.0], index=idx))
    # unchanged yield: about one month of coupon plus a tiny roll-down
    assert abs(r.iloc[0] - 0.04 / 12) < 2e-4
    # yield up 1 point on a 10-year bond (modified duration about 8.1): loss of about 7.5% after coupon and convexity
    assert -0.085 < r.iloc[1] < -0.065


def test_exceedance_normal_benchmark():
    qs = np.array([0.1, 0.3, 0.7, 0.9])
    c = normal_exceedance(0.5, qs)
    assert np.all(c < 0.5)                       # truncation lowers correlation even for a normal distribution
    assert abs(c[0] - c[3]) < 0.03 and abs(c[1] - c[2]) < 0.03   # symmetric
    assert c[0] < c[1]                           # falls further into the tails
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    assert np.isnan(exceedance_corr(x, x, 0.01))  # too few observations


def test_forbes_rigobon_recovers_constant_correlation():
    rng = np.random.default_rng(3)
    n = 400_000
    x = rng.standard_normal(n)
    y = 0.5 * x + np.sqrt(0.75) * rng.standard_normal(n)
    high = np.abs(x) > 1.2                       # condition on the turbulent part of x
    rho_c = np.corrcoef(x[high], y[high])[0, 1]
    assert rho_c > 0.6                           # the raw conditional correlation is biased upward
    # rescaled to the full-sample variance of x, the corrected correlation is the true one
    assert abs(forbes_rigobon(rho_c, x[high].var(), x.var()) - 0.5) < 0.02
    # rescaled to the calm subsample, it matches the calm subsample's own correlation
    rho_low = np.corrcoef(x[~high], y[~high])[0, 1]
    assert abs(forbes_rigobon(rho_c, x[high].var(), x[~high].var()) - rho_low) < 0.02


def test_tail_dependence():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal(100_000), rng.standard_normal(100_000)
    assert abs(tail_dependence(a, b, 0.1) - 0.1) < 0.01
    assert tail_dependence(a, a, 0.1) == 1.0


def test_full_lab_runs_on_simulated_data():
    import lab10

    with tempfile.TemporaryDirectory() as tmp:
        s = lab10.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig10_*.png"))) == 5
        rep = (Path(tmp) / "lab10_report.md").read_text()
        assert "Simulated data" in rep and "Questions to think about" in rep
        c = {r["pair"]: r for r in s["conditional"]}
        assert c["US stocks vs Intl stocks"]["high_vol"] > c["US stocks vs Intl stocks"]["calm"]
        assert s["infl_fit"]["slope_per_1pt_inflation"] > 0      # built in: higher inflation, higher stock-bond corr
        p = {r["portfolio"]: r for r in s["portfolios"]}
        assert p["60/40 stocks/bonds"]["worst_5pct_stock_months_avg"] > p["100% US stocks"]["worst_5pct_stock_months_avg"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
