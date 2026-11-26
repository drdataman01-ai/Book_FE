"""Tests for Lab 17. Run with:  python tests/test_lab17.py   (or: pytest tests/)

Runs offline: the expected-loss and break-even formulas, the excess bond
premium reader, the decomposition on data with a known split between
expected loss and premium, and the full lab on simulated data.
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

from fintest.credit import breakeven_default_rate, expected_loss_spread, read_ebp  # noqa: E402


def test_expected_loss_formulas():
    s = float(expected_loss_spread(0.01, 0.4))
    assert abs(s - (-math.log(1 - 0.006))) < 1e-15 and abs(s - 0.006) < 2e-5
    assert float(expected_loss_spread(0.0, 0.4)) == 0.0
    assert abs(float(breakeven_default_rate(s, 0.4)) - 0.01) < 1e-12  # inverse of each other
    # a risk-neutral investor is indifferent: expected payoff of the risky bond equals the riskless one
    pd_, R, y = 0.02, 0.5, 0.03
    price_risky = math.exp(-(y + float(expected_loss_spread(pd_, R))))
    assert abs(price_risky - math.exp(-y) * ((1 - pd_) + pd_ * R)) < 1e-12  # price = discounted expected payoff


def test_ebp_reader():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ebp_csv.csv"
        p.write_text("date,gz_spread,ebp,est_prob\n1973-01-01,1.50,0.20,0.1\n1973-02-01,1.60,-0.10,0.1\n")
        e = read_ebp(p)
        assert abs(e.loc["1973-02", "ebp"] + 0.001) < 1e-12 and abs(e.loc["1973-01", "gz_spread"] - 0.015) < 1e-12


def test_decomposition_on_known_split():
    import lab17

    idx = pd.period_range("1960-01", periods=240, freq="M")
    df = pd.DataFrame({"GS10": 0.05, "AAA": 0.055, "BAA": 0.055 + 0.01, "USREC": 0, "INDPRO": 100.0}, index=idx)
    s = lab17.spreads(df)
    b = lab17.part_b(s, pd_baa=0.005, pd_aaa=0.0, recovery=0.6)  # expected loss = 0.2% of a 1.5% Baa-Treasury spread
    t = b["table"].set_index("spread")
    assert abs(t.loc["Baa–Treasury", "share_expected_loss"] - float(expected_loss_spread(0.005, 0.6)) / 0.015) < 1e-12
    assert abs(t.loc["Baa–Aaa", "share_expected_loss"] - float(expected_loss_spread(0.005, 0.6)) / 0.01) < 1e-12


def test_full_lab_runs_on_simulated_data():
    import lab17

    with tempfile.TemporaryDirectory() as tmp:
        s = lab17.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig17_*.png"))) == 5
        rep = (Path(tmp) / "lab17_report.md").read_text()
        assert "Simulated data" in rep and "Questions to think about" in rep
        dec = {r["spread"]: r for r in s["decomposition"]}
        assert dec["Baa–Aaa"]["share_expected_loss"] < 0.5  # simulation: a premium on top of 0.15% expected loss
        sp = {r["spread"]: r for r in s["spreads"]}
        assert sp["Baa–Aaa"]["mean_recession"] > sp["Baa–Aaa"]["mean_expansion"]
        assert s["forecast"][0]["effect_of_1sd"] < 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
