"""Tests for Lab 19. Run with:  python tests/test_lab19.py   (or: pytest tests/)

Runs offline: Black-Scholes prices against textbook values, put-call
parity, deltas and implied volatility, the monthly option-writing returns
against hand calculation, the variance-swap payoff, the manipulation-proof
performance measure, the Cboe file reader, and the full lab on simulated
data.
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

from fintest.options import (  # noqa: E402
    bs_delta,
    bs_price,
    bs_vega,
    capture_ratios,
    implied_vol,
    monthly_option_writes,
    mppm,
    read_cboe_index,
    variance_swap_pnl,
)


def test_black_scholes_textbook_values():
    c = float(bs_price(100, 100, 1.0, 0.05, 0.20, "call"))
    p = float(bs_price(100, 100, 1.0, 0.05, 0.20, "put"))
    assert abs(c - 10.4506) < 1e-4 and abs(p - 5.5735) < 1e-4
    assert abs(c - p - (100 - 100 * math.exp(-0.05))) < 1e-10  # put-call parity
    h = 1e-4
    num = (float(bs_price(100 + h, 100, 1, 0.05, 0.2)) - float(bs_price(100 - h, 100, 1, 0.05, 0.2))) / (2 * h)
    assert abs(float(bs_delta(100, 100, 1, 0.05, 0.2)) - num) < 1e-6
    numv = (float(bs_price(100, 100, 1, 0.05, 0.2 + h)) - float(bs_price(100, 100, 1, 0.05, 0.2 - h))) / (2 * h)
    assert abs(float(bs_vega(100, 100, 1, 0.05, 0.2)) - numv) < 1e-5
    assert abs(implied_vol(c, 100, 100, 1.0, 0.05, "call") - 0.20) < 1e-8


def _toy(days=60):
    idx = pd.bdate_range("2020-01-01", periods=days)
    rng = np.random.default_rng(0)
    ret = pd.Series(0.01 * rng.standard_normal(days), idx)
    rf = pd.Series(0.0001, idx)
    iv = pd.Series(20.0, idx)
    return ret, rf, iv


def test_monthly_writes_hand_calculation_and_parity():
    ret, rf, iv = _toy()
    w = monthly_option_writes(ret, rf, iv)
    assert np.allclose(w["put_write"], w["buy_write"], atol=1e-12)  # same strike: same position
    ends = ret.groupby(ret.index.to_period("M")).tail(1).index
    t0, t1 = ends[0], ends[1]
    seg = ret.loc[(ret.index > t0) & (ret.index <= t1)]
    S1 = float((1 + seg).prod())
    rf_m = (1 + 0.0001) ** len(seg) - 1
    T = len(seg) / 252
    P = float(bs_price(1, 1, T, math.log(1 + rf_m) / T, 0.20, "put"))
    expected = rf_m + P * (1 + rf_m) - max(1 - S1, 0)
    assert abs(w["put_write"].iloc[0] - expected) < 1e-12
    w2 = monthly_option_writes(ret, rf, iv, call_strike=1.02)
    C = float(bs_price(1, 1.02, T, math.log(1 + rf_m) / T, 0.20, "call"))
    assert abs(w2["buy_write"].iloc[0] - (S1 - max(S1 - 1.02, 0) + C * (1 + rf_m) - 1)) < 1e-12


def test_variance_swap_payoff():
    ret, rf, iv = _toy(90)
    vs = variance_swap_pnl(ret, iv, horizon=21)
    first_end = ret.groupby(ret.index.to_period("M")).tail(1).index[0]
    i = ret.index.get_loc(first_end)
    realized = float((ret.iloc[i + 1:i + 22] ** 2).sum())
    assert abs(vs["realized"].iloc[0] - realized) < 1e-15
    assert abs(vs["implied"].iloc[0] - 0.04 * 21 / 252) < 1e-15
    assert abs(vs["pnl"].iloc[0] - (vs["implied"].iloc[0] - realized)) < 1e-15


def test_mppm_penalizes_negative_skew():
    rf = np.full(1200, 0.002)
    assert abs(mppm(np.full(1200, 0.007), rf, 3) - 12 * math.log(1.007 / 1.002)) < 1e-10
    rng = np.random.default_rng(1)
    sym = 0.007 + 0.03 * rng.standard_normal(1200)
    skew = np.where(rng.random(1200) < 0.02, -0.15, 0.007 + 0.15 * 0.02 / 0.98 + 0.01 * rng.standard_normal(1200))
    skew = skew - skew.mean() + sym.mean()  # same mean
    skew = (skew - skew.mean()) / skew.std() * sym.std() + sym.mean()  # same mean and volatility
    assert mppm(skew, rf, 3) < mppm(sym, rf, 3)
    cap = capture_ratios(np.array([0.5, -0.5, 1.0, -2.0]), np.array([1.0, -1.0, 2.0, -2.0]))
    assert abs(cap["upside_capture"] - 0.5) < 1e-12 and abs(cap["downside_capture"] - (2.5 / 3)) < 1e-12


def test_cboe_reader():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "PUT_History.csv"
        p.write_text("Cboe S&P 500 PutWrite Index\nSource: Cboe\n\nDATE,PUT\n01/02/2020,1000.00\n01/31/2020,1010.00\n"
                     "02/28/2020,\"1,020.10\"\n")
        s = read_cboe_index(p)
        assert len(s) == 3 and s.name == "PUT" and abs(s.iloc[-1] - 1020.10) < 1e-9


def test_full_lab_runs_on_simulated_data():
    import lab19

    with tempfile.TemporaryDirectory() as tmp:
        s = lab19.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig19_*.png"))) == 5
        rep = (Path(tmp) / "lab19_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        assert s["vrp"]["mean_gap_vol_points"] > 0 and s["vrp"]["share_months_positive"] > 0.6  # built-in premium
        pw = [r for r in s["strategies"] if r["strategy"].startswith("Put-write") and r["sample"] == "Common period"][0]
        assert 0.3 < pw["beta"] < 0.8 and pw["skewness"] < 0 and pw["beta_down"] > pw["beta_up"] + 0.3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
