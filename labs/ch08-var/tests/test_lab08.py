"""Tests for Lab 8. Run with:  python tests/test_lab08.py   (or: pytest tests/)

Runs offline: VaR and ES formulas against known values and Monte Carlo,
historical simulation, the alignment of breaches with forecasts, the
Kupiec and Christoffersen statistics against hand calculation and on data
with known properties, the Basel zones, the ES backtest, and the full lab
on simulated data.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.risk import (  # noqa: E402
    basel_zone,
    christoffersen,
    empirical_var_es,
    es_backtest,
    es_normal,
    es_t,
    hits,
    historical_var_es,
    kupiec,
    var_normal,
    var_t,
)


def test_parametric_var_es():
    assert abs(var_normal(0.01, 0.01) - 0.0232635) < 1e-6
    assert abs(es_normal(0.01, 0.025) - 0.0233780) < 1e-6  # ES 97.5% ~ VaR 99% for a normal
    rng = np.random.default_rng(0)
    nu = 5
    x = rng.standard_t(nu, 2_000_000) * math.sqrt((nu - 2) / nu) * 0.02
    v, e = empirical_var_es(x, 0.01)
    assert abs(var_t(0.02, nu, 0.01) - v) / v < 0.01
    assert abs(es_t(0.02, nu, 0.01) - e) / e < 0.02


def test_historical_simulation():
    r = pd.Series(np.linspace(-0.05, 0.05, 101), index=pd.bdate_range("2000-01-03", periods=101))
    hs = historical_var_es(r, window=100, alpha=0.05)
    w = r.iloc[1:101].to_numpy()
    assert abs(hs["var"].iloc[100] + np.quantile(w, 0.05)) < 1e-12
    assert abs(hs["es"].iloc[100] + np.sort(w)[:5].mean()) < 1e-12
    assert np.isnan(hs["var"].iloc[98])


def test_hits_alignment():
    r = np.array([0.0, -0.03, 0.01, -0.05])
    v = np.array([0.02, 0.02, 0.06, 0.02])
    h = hits(r, v)  # day 1 loss 3% > 2% (forecast made day 0); day 3 loss 5% < 6% (forecast made day 2)
    assert np.isnan(h[0]) and h[1] == 1 and h[2] == 0 and h[3] == 0


def test_kupiec_against_hand_calculation():
    h = np.array([1] * 10 + [0] * 240, float)
    k = kupiec(h, 0.01)
    lr = -2 * (240 * math.log(0.99) + 10 * math.log(0.01) - 240 * math.log(0.96) - 10 * math.log(0.04))
    assert abs(k["lr"] - lr) < 1e-10 and k["p"] < 0.001
    rng = np.random.default_rng(1)
    ps = [kupiec((rng.random(1000) < 0.01).astype(float), 0.01)["p"] for _ in range(400)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10  # about 5% false rejections (discrete test)


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(2)
    iid = (rng.random(5000) < 0.01).astype(float)
    assert christoffersen(iid, 0.01)["p_ind"] > 0.01
    clustered = np.zeros(5000)
    for s in rng.choice(4990, 17, replace=False):
        clustered[s:s + 3] = 1  # breaches in runs of three
    c = christoffersen(clustered, 0.01)
    assert c["p11"] > 0.5 and c["p_ind"] < 1e-6


def test_basel_zones():
    assert [basel_zone(k) for k in (0, 4, 5, 9, 10, 15)] == ["green", "green", "yellow", "yellow", "red", "red"]


def test_es_backtest_unbiased_and_biased():
    rng = np.random.default_rng(3)
    n, s = 200_000, 0.01
    r = s * rng.standard_normal(n)
    v, e = np.full(n, var_normal(s, 0.025)), np.full(n, es_normal(s, 0.025))
    ok = es_backtest(r, v, e)
    assert abs(ok["ratio"] - 1) < 0.02 and abs(ok["t"]) < 3
    r_fat = s * rng.standard_t(3, n) / math.sqrt(3)  # fatter tails than the normal ES assumes
    bad = es_backtest(r_fat, v, e)
    assert bad["ratio"] > 1.1 and bad["t"] > 3


def test_full_lab_runs_on_simulated_data():
    import lab08

    with tempfile.TemporaryDirectory() as tmp:
        s = lab08.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig8_*.png"))) == 5
        rep = (Path(tmp) / "lab08_report.md").read_text()
        assert "Simulated data" in rep and "Questions to think about" in rep
        bt = {r["model"]: r for r in s["backtest_99"]}
        # the simulation is GJR-GARCH with t shocks, so the matching model should pass and the normal one fail
        assert bt["GJR-GARCH, Student-t"]["kupiec_p"] > 0.01 and bt["Normal, 250-day"]["kupiec_p"] < 0.01
        assert bt["Normal, 250-day"]["independence_p"] < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
