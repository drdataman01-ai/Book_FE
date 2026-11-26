"""Tests for Lab 20. Run with:  python tests/test_lab20.py   (or: pytest tests/)

Runs offline: feature timing and rank scaling, the walk-forward split (no
training target realized after the refit date), the long-short portfolio and
turnover, the R-squared, random-feature timing, models recovering a planted
signal and finding nothing when there is none, and the full lab on a small
simulated panel.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.ml import (_rank_scale, build_panel, long_short, make_models, r2_oos,  # noqa: E402
                        random_features_timing, simulated_panel, walk_forward)


def test_features_use_only_past_data():
    d = simulated_panel(n_assets=20, n_months=200, seed=1)
    P = build_panel(d["ret"], d["mkt"], d["size"], d["nfirms"], d["bm"])
    a = d["ret"].columns[3]
    rows = P[P["asset"] == a]
    t = rows.index[10]
    # target is next month's return
    assert abs(rows.loc[t, "ret_next"] - d["ret"].loc[t + 1, a]) < 1e-12
    # changing a future return changes no feature dated t
    ret2 = d["ret"].copy()
    ret2.loc[t + 1:, :] += 0.5
    P2 = build_panel(ret2, d["mkt"], d["size"], d["nfirms"], d["bm"])
    r1 = P[(P.index == t)].drop(columns=["ret_next", "ret_next_dm"]).set_index("asset")
    r2 = P2[(P2.index == t)].drop(columns=["ret_next", "ret_next_dm"]).set_index("asset")
    assert np.allclose(r1.to_numpy(), r2.loc[r1.index].to_numpy())
    # demeaned target averages to zero each month
    assert P.groupby(level=0)["ret_next_dm"].mean().abs().max() < 1e-12


def test_rank_scale():
    df = pd.DataFrame([[1.0, 5.0, 3.0, np.nan]])
    r = _rank_scale(df).iloc[0].to_numpy()
    assert r[3] == 0 and r[1] > r[2] > r[0] and abs(r[0] + r[1]) < 1e-12 and r.max() < 1


def test_walk_forward_split():
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(len(X))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    idx = pd.period_range("1960-01", periods=360, freq="M")
    panel = pd.DataFrame({"x": np.arange(360.0), "ret_next": 0.0, "ret_next_dm": 0.0, "asset": "A"}, index=idx)
    res = walk_forward(panel, {"spy": (lambda p, s: Spy(), [{}], ["x"], [0])}, test_start="1980-01", refit_years=5,
                       val_years=5)
    # first refit: rows dated up to 1979-11 (targets realized by 1979-12) -> 239 rows
    assert seen[0] == 239
    assert res["pred"].index.min() == pd.Period("1980-01", "M")


def test_long_short_and_r2():
    idx = pd.period_range("2000-01", periods=24, freq="M")
    rows = [(t, f"A{i}", i / 10, i / 10) for t in idx for i in range(10)]
    pred = pd.DataFrame(rows, columns=["date", "asset", "m", "y"]).set_index("date")
    ls = long_short(pred, "m", frac=0.1)
    assert np.allclose(ls["ls"], 0.9) and ls["turnover"].iloc[1:].abs().max() == 0 and ls["turnover"].iloc[0] == 2
    assert abs(r2_oos([1, -1], [1, -1]) - 1) < 1e-12 and r2_oos([1, -1], [0, 0]) == 0


def test_models_find_planted_signal_and_nothing_otherwise():
    models = {k: v for k, v in make_models(fast=True).items() if k in ("OLS (all)", "Gradient-boosted trees")}
    for signal, check in ((3.0, lambda r: r > 0.5), (0.0, lambda r: r < 0.1)):
        d = simulated_panel(n_assets=60, n_months=480, seed=2, signal=signal)
        P = build_panel(d["ret"], d["mkt"], d["size"], d["nfirms"], d["bm"])
        pr = walk_forward(P, models, test_start="1950-01", refit_years=10, val_years=5)["pred"]
        for n in models:
            assert check(100 * r2_oos(pr["y"], pr[n])), (signal, n, 100 * r2_oos(pr["y"], pr[n]))


def test_random_features_timing_runs():
    rng = np.random.default_rng(0)
    idx = pd.period_range("1950-01", periods=300, freq="M")
    X = pd.DataFrame(rng.standard_normal((300, 4)), index=idx)
    y = pd.Series(0.005 + 0.04 * rng.standard_normal(300), index=idx)
    t = random_features_timing(X, y, n_features=(2, 24), ridge=(1.0,))
    assert len(t) == 2 and t["months"].iloc[0] > 200 and np.isfinite(t["r2_oos"]).all()


def test_full_lab_runs_on_simulated_data():
    import lab20

    with tempfile.TemporaryDirectory() as tmp:
        s = lab20.main(["--source", "sim", "--fast", "--sim-assets", "40", "--sim-months", "600", "--test-start",
                        "1955-01", "--out", tmp])
        assert len(list(Path(tmp).glob("fig20_*.png"))) == 5
        rep = (Path(tmp) / "lab20_report.md").read_text()
        assert "Simulated data" in rep and "Questions to think about" in rep
        assert len(s["main"]) == 5 and len(s["seeds"]) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
