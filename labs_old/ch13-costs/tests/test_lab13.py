"""Tests for Lab 13. Run with:  python tests/test_lab13.py   (or: pytest tests/)

Runs offline. Cost formulas are checked by hand, turnover against a direct
calculation, signals for look-ahead, the factor files' parsers on the
library's layouts, and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.costs import (  # noqa: E402
    backtest_sort,
    break_even_cost,
    capacity_curve,
    monthly_cost,
    net_returns,
    signals,
    simulate_universe,
    sort_weights,
    sqrt_impact_cost,
)
from fintest.data import parse_french_csv  # noqa: E402

MOM_SAMPLE = """This file was created using the 202508 CRSP database.
The momentum factor is constructed from six value-weight portfolios formed using independent sorts on size and prior return.

Missing data are indicated by -99.99.

,Mom
192701,    0.57
192702,   -1.50
192703,    3.52

Annual Factors:
,Mom
1927,  25.00
"""

REV_SAMPLE = """Short-Term Reversal Factor (ST Rev) is constructed from six value-weight portfolios.

,ST_Rev
192602,   -2.10
192603,    1.12
"""


def test_parse_factor_files():
    import lab13

    mom = parse_french_csv(MOM_SAMPLE)
    assert len(mom) == 3 and abs(lab13._first_col(mom, "Mom").iloc[2] - 0.0352) < 1e-12
    rev = parse_french_csv(REV_SAMPLE)
    assert abs(lab13._first_col(rev, "ST_Rev").iloc[0] + 0.021) < 1e-12


def test_cost_formulas():
    assert abs(monthly_cost(0.35, 0.0045) - 4 * 0.35 * 0.0045) < 1e-15
    assert abs(monthly_cost(0.10, 0.002, sides=1) - 2 * 0.10 * 0.002) < 1e-15
    be = break_even_cost(0.0133, 0.3452)
    assert abs(monthly_cost(0.3452, be) - 0.0133) < 1e-12
    assert break_even_cost(0.01, 0.0) == float("inf")
    g = np.array([0.01, 0.02])
    n = net_returns(g, 0.5, 0.001, borrow_fee_annual=0.012)
    assert np.allclose(n, g - 4 * 0.5 * 0.001 - 0.001)


def test_novy_marx_velikov_costs_are_consistent():
    """Their Table 3 costs imply a similar one-way cost (~45 bp) for all three strategies under 4 T c."""
    import lab13

    costs = {"value": 0.0005, "momentum": 0.0065, "reversal": 0.0165}
    implied = [costs[k] / (4 * lab13.PUBLISHED_TURNOVER[k]) for k in costs]
    assert max(implied) - min(implied) < 0.0006 and 0.003 < np.mean(implied) < 0.006


def test_turnover_matches_direct_calculation():
    rng = np.random.default_rng(0)
    T, N = 30, 50
    R = rng.normal(0.01, 0.05, (T, N))
    ME = np.cumprod(1 + R, axis=0) * rng.uniform(1, 10, N)
    u = {"returns": R, "me": ME, "be": ME.copy()}
    # A constant signal: membership never changes, so value-weighted drift needs no trading.
    const = {"value": np.tile(np.arange(N, dtype=float), (T, 1))}
    import fintest.costs as fc

    orig = fc.signals
    fc.signals = lambda u_: const
    try:
        bt = backtest_sort(u, "value", start=1)
    finally:
        fc.signals = orig
    assert np.nanmax(bt["turnover"]) < 1e-12
    # Direct one-sided turnover for one rebalance.
    w_prev = np.array([0.5, 0.5, 0, 0])
    r = np.array([0.1, -0.1, 0, 0])
    drift = w_prev * (1 + r) / (w_prev @ (1 + r))
    new = np.array([0, 0.5, 0.5, 0])
    assert abs(0.5 * np.abs(new - drift).sum() - drift[0]) < 1e-12


def test_signals_use_only_past_data():
    u = simulate_universe(200, 60, seed=1)
    s1 = signals(u)
    t = 40
    u2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in u.items()}
    u2["returns"][t + 1:] += 0.5  # change the future
    u2["me"][t + 1:] *= 2.0
    u2["be"][t + 1:] *= 3.0
    s2 = signals(u2)
    for k in s1:
        assert np.allclose(s1[k][:t + 1], s2[k][:t + 1], equal_nan=True), k
    # Value is updated once a year and held in between.
    v = s1["value"]
    changes = [i for i in range(1, 60) if not np.allclose(v[i], v[i - 1], equal_nan=True)]
    assert all(i % 12 == 5 for i in changes)


def test_turnover_ordering_and_banding():
    u = simulate_universe(600, 300, seed=2)
    tv = {k: np.nanmean(backtest_sort(u, k)["turnover"]) for k in ("value", "momentum", "reversal")}
    assert tv["value"] < tv["momentum"] < tv["reversal"]
    assert 0.10 < tv["momentum"] < 0.45 and tv["reversal"] > 0.6 and tv["value"] < 0.06
    band = backtest_sort(u, "momentum", hold_frac=0.5)
    std = backtest_sort(u, "momentum")
    assert np.nanmean(band["turnover"]) < 0.7 * tv["momentum"]
    assert 0 < band["expected"].mean() < std["expected"].mean()
    # Banding keeps every stock a standard sort would buy.
    sig = signals(u)["momentum"][100]
    top, _ = sort_weights(sig, u["me"][100])
    btop, _ = sort_weights(sig, u["me"][100], hold_frac=0.5, prev_long=np.ones_like(top), prev_short=np.zeros_like(top))
    assert (btop >= top).all()


def test_true_premia_are_recovered():
    u = simulate_universe(800, 300, seed=3, premia={"value": 0.0, "momentum": 0.004, "reversal": 0.0})
    assert backtest_sort(u, "momentum")["expected"].mean() > 0.004
    assert abs(backtest_sort(u, "reversal")["expected"].mean()) < 0.001


def test_impact_and_capacity():
    c = sqrt_impact_cost(np.array([1e6]), np.array([1e8]), np.array([0.02]), y=1.0)
    assert abs(c[0] - 0.02 * np.sqrt(0.01)) < 1e-15
    n = 50
    w = np.full(n, 1 / n)
    cap = capacity_curve(np.logspace(6, 12, 30), 0.006, 0.3, w, np.full(n, 5e7), np.full(n, 0.02),
                         np.full(n, 0.001))
    assert np.all(np.diff(cap["net"]) <= 1e-15)
    assert abs(cap["net"][0] - (0.006 - 4 * 0.3 * 0.001 - cap["cost"][0] + 4 * 0.3 * 0.001)) < 1e-12
    assert cap["half_net_aum"] < cap["zero_net_aum"]


def test_full_lab_runs_on_simulated_data():
    import lab13

    with tempfile.TemporaryDirectory() as tmp:
        summary = lab13.main(["--source", "sim", "--n-stocks", "400", "--out", tmp])
        out = Path(tmp)
        assert len(list(out.glob("fig13_*.png"))) == 5
        report = (out / "lab13_report.md").read_text()
        assert "Simulated factor data" in report and "Your verdict" in report
        assert summary["simulated"] is True
        be = summary["break_even_one_way"]
        assert be["value"] > be["momentum"] > be["reversal"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
