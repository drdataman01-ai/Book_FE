"""Tests for Lab 15. Run with:  python tests/test_lab15.py   (or: pytest tests/)

Runs offline: parsing a file in the French library's layout for portfolios
formed on beta, the AQR reader on a CSV in AQR's layout, ex-ante betas
against direct calculation, beta neutrality of the BAB-style portfolio,
alphas on data with a known alpha, and the full lab on simulated data.
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

import lab15  # noqa: E402
from fintest.data import load_french_tables  # noqa: E402

HEADER = ",<= 0,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10"


def _french_file(tmp: Path) -> Path:
    rows = []
    for title, base in (("Value Weight Returns -- Monthly", 1.0), ("Equal Weight Returns -- Monthly", 2.0)):
        rows += ["", f"  {title}", HEADER]
        for ym in ("196307", "196308"):
            vals = [-99.99] + [base + q for q in range(5)] + [base + 0.1 * k for k in range(10)]
            rows.append(ym + "," + ",".join(f"{v:.2f}" for v in vals))
    rows += ["", "  Value Weight Returns -- Annual", HEADER, "1964," + ",".join(["1.0"] * 16)]
    p = tmp / "Portfolios_Formed_on_BETA.csv"
    p.write_text("This file was created using the CRSP database.\n" + "\n".join(rows) + "\n")
    return p


def test_french_beta_layout():
    with tempfile.TemporaryDirectory() as tmp:
        tables = load_french_tables("beta_portfolios", path=_french_file(Path(tmp)))
        vw, ew = lab15._weighted_tables(tables)
        dec, q = lab15._block(vw, "Lo 10", 10), lab15._block(vw, "Lo 20", 5)
        assert list(dec.columns) == [f"D{i}" for i in range(1, 11)] and list(q.columns) == [f"Q{i}" for i in range(1, 6)]
        assert abs(dec.iloc[0, 0] - 0.010) < 1e-12 and abs(dec.iloc[0, 9] - 0.019) < 1e-12  # percent -> decimal
        assert abs(lab15._block(ew, "Lo 10", 10).iloc[0, 0] - 0.020) < 1e-12
        assert len(dec) == 2  # annual table ignored


def test_aqr_reader_csv():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bab.csv"
        p.write_text("Betting Against Beta Equity Factors,,\nSource: AQR,,\n,,\nDATE,AUS,USA\n"
                     "01/31/1931,,0.0123\n02/28/1931,0.01,-0.0040\n")
        s = lab15.read_aqr_bab(p)
        assert len(s) == 2 and str(s.index[0]) == "1931-01" and abs(s.iloc[1] + 0.004) < 1e-12


def test_exante_beta_matches_direct_calculation():
    rng = np.random.default_rng(0)
    idx = pd.period_range("1990-01", periods=120, freq="M")
    m = pd.Series(0.04 * rng.standard_normal(120), idx)
    y = 0.7 * m + 0.01 * rng.standard_normal(120)
    b = lab15.exante_beta(y, m, window=60)
    t = 80
    direct = np.cov(y.iloc[t - 60:t], m.iloc[t - 60:t])[0, 1] / m.iloc[t - 60:t].var()
    assert abs(b.iloc[t] - direct) < 1e-12 and np.isnan(b.iloc[30])


def _known(alpha_m=0.003, n=600, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1963-07", periods=n, freq="M")
    f = {k: pd.Series(s * rng.standard_normal(n), idx) for k, s in (("mkt", 0.045), ("smb", 0.03), ("hml", 0.03),
                                                                     ("umd", 0.04))}
    d = {**f, "rf": pd.Series(0.003, idx)}
    betas = np.linspace(0.5, 1.6, 10)
    ex = pd.DataFrame({f"D{i + 1}": (alpha_m if i < 3 else 0.0) + betas[i] * f["mkt"] + 0.005 * rng.standard_normal(n)
                       for i in range(10)})
    ex.index = idx
    return d, ex


def test_bab_is_beta_neutral_and_alpha_recovered():
    d, ex = _known()
    lh = lab15.low_high(ex, d)
    a = lab15.alpha(lh["bab"], d, "capm")
    assert abs(a["beta_mkt"]) < 0.05
    # low leg (beta ~0.62) has alpha 0.3% a month, scaled by 1/beta in the BAB portfolio
    assert abs(a["alpha_annual"] - 12 * 0.003 / np.mean(np.linspace(0.5, 1.6, 10)[:3])) < 0.01
    a4 = lab15.alpha(lh["lo"], d, "ff4")
    assert abs(a4["alpha_annual"] - 0.036) < 0.006 and abs(a4["beta_mkt"] - np.mean(np.linspace(0.5, 1.6, 10)[:3])) < 0.02


def test_full_lab_runs_on_simulated_data():
    with tempfile.TemporaryDirectory() as tmp:
        s = lab15.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig15_*.png"))) == 5
        rep = (Path(tmp) / "lab15_report.md").read_text()
        assert "Simulated data" in rep and "Questions to think about" in rep
        assert s["sml_slopes"]["beta_vw"] < s["market_premium"]  # flat SML built into the simulation
        bab = [r for r in s["long_short"] if r["construction"].startswith("BAB") and r["sort"] == "Market beta"][0]
        assert abs(bab["capm_beta_mkt"]) < 0.1 and bab["capm_alpha_annual"] > 0
        assert all(r["dy10_coef"] < 0 for r in s["rates"])  # low-risk legs built to be hurt by rising yields
        assert not math.isnan(s["investor"][0]["tracking_error"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
