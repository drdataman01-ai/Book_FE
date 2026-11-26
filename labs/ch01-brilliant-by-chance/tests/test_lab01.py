"""Tests for Lab 1. Run with:  python tests/test_lab01.py   (or: pytest tests/)

No network access is needed: the French-library parser is tested on a file
written in the library's format, and the full lab runs on simulated data.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.data import load_french_factors, parse_french_csv, simulated_market  # noqa: E402
from fintest.stats import (  # noqa: E402
    bonferroni_threshold,
    normal_ppf,
    ols_newey_west,
    sharpe_ratio,
    years_to_detect,
)

# A file laid out like F-F_Research_Data_Factors.csv: description lines, a
# monthly table, a blank line, then an annual table that must be ignored.
FRENCH_SAMPLE = """This file was created by CMPT_ME_BEME_RETS using the 202508 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
192609,    0.38,   -1.36,    0.05,    0.23
192610,   -3.27,   -0.14,    0.82,    0.32
192611,    2.54,   -0.11,   -0.61,    0.31

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.46,   -3.75,    3.12
1928,   35.39,    4.41,   -5.83,    3.56

Copyright 2025 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_monthly():
    df = parse_french_csv(FRENCH_SAMPLE, "M")
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 5, "annual rows must not be included"
    assert str(df.index[0]) == "1926-07" and str(df.index[-1]) == "1926-11"
    assert abs(df.loc["1926-07", "Mkt-RF"] - 0.0289) < 1e-12  # percent -> decimal


def test_load_from_zip_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "F-F_Research_Data_Factors_CSV.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("F-F_Research_Data_Factors.csv", FRENCH_SAMPLE)
        path.write_bytes(buf.getvalue())
        df = load_french_factors(path=path)
        assert len(df) == 5 and abs(df["RF"].iloc[1] - 0.0025) < 1e-12


def test_parse_daily_and_missing_codes():
    text = ",Mkt-RF,RF\n19260701,    0.10,    0.009\n19260702,  -99.99,    0.009\n\n"
    df = parse_french_csv(text, "D")
    assert len(df) == 2 and np.isnan(df["Mkt-RF"].iloc[1])


def test_statistics():
    assert abs(years_to_detect(0.5) - 15.3664) < 1e-3
    assert abs(normal_ppf(0.975) - 1.959964) < 1e-5
    assert abs(bonferroni_threshold(1000) - 3.8906) < 1e-3
    r = np.full(120, 0.01) + np.tile([0.02, -0.02], 60)
    assert abs(float(sharpe_ratio(r)) - 0.01 / np.std(r, ddof=1) * np.sqrt(12)) < 1e-12


def test_newey_west_matches_manual():
    rng = np.random.default_rng(3)
    t = 300
    x = rng.standard_normal(t)
    y = 0.2 + 0.7 * x + rng.standard_normal(t)
    res = ols_newey_west(y, x, lags=4)
    X = np.column_stack([np.ones(t), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    g = X * e[:, None]
    S = g.T @ g
    for lag in range(1, 5):
        G = g[lag:].T @ g[:-lag]
        S += (1 - lag / 5) * (G + G.T)
    inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(inv @ S @ inv))
    assert np.allclose(res["coef"][:, 0], b) and np.allclose(res["se"][:, 0], se)


def test_random_strategies_have_no_skill_on_average():
    """Across many seeds, about 2.5% of zero-skill strategies should pass t > 1.96."""
    from lab01 import part_b

    mkt = simulated_market(seed=11)
    mkt = mkt.rename(columns={"Mkt-RF": "mkt_rf", "RF": "rf"})
    shares = [float((part_b(mkt, 1000, None, seed).t_is > 1.96).mean()) for seed in range(8)]
    assert 0.012 < np.mean(shares) < 0.045, shares


def test_positions_have_no_look_ahead():
    from lab01 import random_positions

    rng = np.random.default_rng(0)
    pos = random_positions(240, 50, rng)
    assert set(np.unique(pos)) <= {0, 1}
    # Positions are generated without access to returns: re-running with the
    # same seed and different market data must give the same positions.
    assert np.array_equal(pos, random_positions(240, 50, np.random.default_rng(0)))


def test_full_lab_runs_on_simulated_data():
    import lab01

    with tempfile.TemporaryDirectory() as tmp:
        summary = lab01.main(["--source", "sim", "--n-strategies", "300", "--out", tmp])
        out = Path(tmp)
        assert len(list(out.glob("fig1_*.png"))) == 5
        report = (out / "lab01_report.md").read_text()
        assert "Simulated data" in report and "Questions to think about" in report
        assert json.loads((out / "summary.json").read_text())["simulated"] is True
        assert summary["n_strategies"] == 300


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
