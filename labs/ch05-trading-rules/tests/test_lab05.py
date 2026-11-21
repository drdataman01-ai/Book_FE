"""Tests for Lab 5. Run with:  python tests/test_lab05.py   (or: pytest tests/)

Runs offline: signals use only past prices; the holding-period and thinning
rules; the Reality Check and SPA tests reject a rule with real skill and keep
their size under the null; the AQR and Stooq readers on files in their
published layouts; and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lab05  # noqa: E402
from fintest.search import (  # noqa: E402
    breakout_signal,
    hold_after_events,
    ma_signal,
    reality_check,
    rule_positions,
    rule_universe,
    stationary_bootstrap_matrix,
)


def test_signals_use_only_past_prices():
    rng = np.random.default_rng(0)
    lp = np.cumsum(0.01 * rng.standard_normal(600))
    rules = rule_universe(shorts=(1, 5), longs=(50, 150), breakouts=(20,), bands=(0.0, 0.01), holds=(0, 10))
    p1 = rule_positions(lp, rules)
    lp2 = lp.copy()
    lp2[400:] += 5 * rng.standard_normal(200).cumsum()  # change the future only
    p2 = rule_positions(lp2, rules)
    assert np.array_equal(p1[:400], p2[:400])


def test_ma_and_breakout_definitions():
    lp = np.log(np.r_[np.full(10, 100.0), np.full(5, 120.0)])
    s = ma_signal(lp, 1, 5)
    assert s[9] == 0 and s[10] == 1  # price jumps above the 5-day average
    b = breakout_signal(lp, 5)
    assert b[10] == 1 and b[11] == 0  # a new 5-day high only on the first day
    pos = hold_after_events(np.r_[1, 0, 0, 0, -1, 0, 0], 3)
    assert list(pos) == [1, 1, 1, 0, -1, -1, -1]
    ev = lab05.thin_events(np.r_[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], 10)
    assert list(np.flatnonzero(ev)) == [0, 10]


def test_bootstrap_blocks():
    idx = stationary_bootstrap_matrix(2000, 50, 10, np.random.default_rng(1))
    assert idx.min() >= 0 and idx.max() < 2000
    share_new = (np.diff(idx, axis=1) % 2000 != 1).mean()
    assert 0.08 < share_new < 0.12


def test_reality_check_size_and_power():
    rejections = 0
    for s in range(20):
        d = np.random.default_rng(s).normal(0, 0.01, (1500, 100))
        rejections += reality_check(d, n_boot=200, seed=s)["p_reality_check"] < 0.05
    assert rejections <= 4  # about 1 expected
    d = np.random.default_rng(99).normal(0, 0.01, (1500, 100))
    d[:, 3] += 0.002
    r = reality_check(d, n_boot=300)
    assert r["best"] == 3 and r["p_reality_check"] < 0.01 and r["p_spa"] < 0.01


def test_aqr_and_stooq_readers():
    rows = [["AQR Capital Management", None, None], ["Time Series Momentum Factors", None, None], [None] * 3,
            ["DATE", "TSMOM", "TSMOM^EQ"], ["1985-01-31", 0.012, -0.004], ["1985-02-28", -0.020, 0.010]]
    t = lab05.parse_aqr_tsmom(pd.DataFrame(rows))
    assert list(t.columns) == ["TSMOM", "TSMOM_EQ"] and abs(t.iloc[1, 0] + 0.02) < 1e-12
    assert str(t.index[0]) == "1985-01"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "dji.csv"
        f.write_text("Date,Open,High,Low,Close,Volume\n1990-01-03,1,1,1,2753.2,0\n1990-01-02,1,1,1,2810.15,0\n")
        s = lab05.read_stooq(f)
        assert s.index[0] < s.index[1] and abs(s.iloc[0] - 2810.15) < 1e-9


def test_full_lab_runs_on_simulated_data():
    with tempfile.TemporaryDirectory() as tmp:
        s = lab05.main(["--source", "sim", "--fast", "--out", tmp])
        assert s["simulated"] is True
        assert len(list(Path(tmp).glob("fig5_*.png"))) == 5
        assert (Path(tmp) / "lab05_report.md").exists()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
