"""Tests for Lab 7. Run with:  python tests/test_lab07.py   (or: pytest tests/)

Runs offline: the EWMA and GARCH recursions against explicit loops, GARCH
estimation on data with known parameters, the multi-step forecast formula,
HAR features and future sums, the loss functions, the FRED reader on a file
in FRED's layout, and the full lab on simulated data.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.vol import (  # noqa: E402
    ewma_variance,
    fit_garch,
    future_sum,
    garch_forecast_sum,
    garch_variance,
    har_features,
    load_fred,
    mincer_zarnowitz,
    qlike_loss,
)


def test_recursions_match_loops():
    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_normal(500)
    e = ewma_variance(r, 0.94, init=1e-4)
    s = 1e-4
    for t in range(1, 500):
        s = 0.94 * s + 0.06 * r[t - 1] ** 2
        assert abs(e[t] - s) < 1e-15
    g = garch_variance(r, 2e-6, 0.05, 0.9, 0.08, init=1e-4)
    s = 1e-4
    for t in range(1, 500):
        s = 2e-6 + (0.05 + 0.08 * (r[t - 1] < 0)) * r[t - 1] ** 2 + 0.9 * s
        assert abs(g[t] - s) < 1e-15


def _simulate_gjr(n, om, a, g, b, seed):
    rng = np.random.default_rng(seed)
    r, s2 = np.zeros(n), np.zeros(n)
    s2[0] = om / (1 - a - b - g / 2)
    for t in range(1, n):
        s2[t] = om + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * rng.standard_normal()
    return r


def test_garch_estimation_recovers_parameters():
    r = _simulate_gjr(15000, 2e-6, 0.03, 0.10, 0.90, seed=3)
    p = fit_garch(r, gjr=True)
    assert abs(p["beta"] - 0.90) < 0.03 and abs(p["gamma"] - 0.10) < 0.04 and abs(p["alpha"] - 0.03) < 0.03
    p0 = fit_garch(_simulate_gjr(15000, 2e-6, 0.08, 0.0, 0.90, seed=4))
    assert abs(p0["alpha"] - 0.08) < 0.02 and abs(p0["beta"] - 0.90) < 0.03


def test_multistep_forecast_formula():
    p = {"long_run_var": 1e-4, "persistence": 0.98}
    s_next, h = 4e-4, 10
    direct, s = 0.0, s_next
    for _ in range(h):
        direct += s
        s = p["long_run_var"] + p["persistence"] * (s - p["long_run_var"])
    assert abs(garch_forecast_sum(s_next, p, h) - direct) < 1e-15
    assert abs(garch_forecast_sum(p["long_run_var"], p, h) - h * 1e-4) < 1e-15


def test_har_features_and_future_sum():
    x = np.arange(1.0, 31.0)
    f = har_features(x)
    assert f[29, 0] == 30 and f[29, 1] == np.mean(x[25:30]) and f[29, 2] == np.mean(x[8:30])
    y = future_sum(x, 3)
    assert y[0] == 2 + 3 + 4 and np.isnan(y[-1]) and np.isnan(y[-3]) and y[-4] == 28 + 29 + 30


def test_losses():
    # QLIKE is minimized, in expectation, at the true variance
    rng = np.random.default_rng(1)
    rv = 2.0 * rng.chisquare(1, 200000)
    grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    losses = [qlike_loss(rv, np.full_like(rv, f)).mean() for f in grid]
    assert int(np.argmin(losses)) == 2
    f = rng.uniform(1, 3, 5000)
    mz = mincer_zarnowitz(0.5 + 2 * f, f)
    assert abs(mz["slope"] - 2) < 1e-10 and abs(mz["intercept"] - 0.5) < 1e-10


def test_fred_reader():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "VIXCLS.csv"
        p.write_text("observation_date,VIXCLS\n1990-01-02,17.24\n1990-01-03,.\n1990-01-04,19.22\n")
        s = load_fred("VIXCLS", p)
        assert len(s) == 2 and abs(s.iloc[-1] - 19.22) < 1e-12


def test_full_lab_runs_on_simulated_data():
    import lab07

    with tempfile.TemporaryDirectory() as tmp:
        s = lab07.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig7_*.png"))) == 5
        rep = (Path(tmp) / "lab07_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        assert s["acf_abs_lag1"] > 0.1 and abs(s["acf_r_lag1"]) < 0.1
        ev = {(e["horizon"], e["model"]): e for e in s["evaluation"]}
        assert ev[(1, "GARCH")]["qlike_gain_vs_rw"] > 0  # GARCH beats the random walk on GARCH data
        assert ev[(22, "Long-run average")]["qlike_gain_vs_rw"] < ev[(22, "GJR")]["qlike_gain_vs_rw"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
