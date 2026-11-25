"""Tests for Lab 12. Run with:  python tests/test_lab12.py   (or: pytest tests/)

Runs offline: the yield-curve reader on a file in the Federal Reserve's
layout, forwards and excess returns against hand calculation, the
expectations-hypothesis regressions on data where the hypothesis holds
exactly (slopes 0 and 1) and on data where it fails, and the full lab on
simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.rates import campbell_shiller, fama_bliss, fama_bliss_short, read_gsw, term_structure  # noqa: E402


def test_reader_layout():
    hdr = "Date," + ",".join(f"SVENY{n:02d}" for n in range(1, 11)) + ",BETA0"
    rows = ["2020-01-30," + ",".join(f"{1 + 0.1 * n:.4f}" for n in range(1, 11)) + ",1",
            "2020-01-31," + ",".join(f"{2 + 0.1 * n:.4f}" for n in range(1, 11)) + ",1",
            "2020-02-28," + ",".join(["NA"] * 10) + ",1"]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "feds200628.csv"
        p.write_text("Federal Reserve Board\nThe U.S. Treasury Yield Curve\n\n\n" + hdr + "\n" + "\n".join(rows) + "\n")
        y = read_gsw(p)
        assert abs(y.loc["2020-01", 1] - 0.021) < 1e-12 and abs(y.loc["2020-01", 10] - 0.030) < 1e-12  # month end, decimals


def test_forwards_and_returns_by_hand():
    idx = pd.period_range("2000-01", periods=30, freq="M")
    y = pd.DataFrame({1: 0.02, 2: 0.03, 3: 0.035}, index=idx).astype(float)
    ts = term_structure(y)
    assert abs(ts["f"][2].iloc[0] - (2 * 0.03 - 0.02)) < 1e-12  # f(2) = p(1) - p(2)
    # constant curve: rx(2) = p(1)_{t+1} - p(2)_t - y1 = -0.02 + 0.06 - 0.02
    assert abs(ts["rx"][2].iloc[0] - 0.02) < 1e-12 and np.isnan(ts["rx"][2].iloc[-1])


def _eh_world(n_months=900, seed=0, premium=False):
    """Yields from an AR(1) one-year rate; with premium=False the expectations hypothesis holds exactly."""
    rng = np.random.default_rng(seed)
    phi = 0.9  # annual persistence; data are annual steps observed monthly via interpolation of a monthly AR
    phim = phi ** (1 / 12)
    r = np.zeros(n_months)
    tp = np.zeros(n_months)
    for t in range(1, n_months):
        r[t] = phim * r[t - 1] + 0.004 * rng.standard_normal()
        tp[t] = 0.97 * tp[t - 1] + 0.002 * rng.standard_normal()
    cols = {}
    for n in range(1, 6):
        k = np.arange(n)
        cols[n] = 0.04 + (r[:, None] * phi ** k).mean(axis=1) + (tp * (n - 1) / 4 if premium else 0)
    return pd.DataFrame(cols, index=pd.period_range("1950-01", periods=n_months, freq="M"))


def test_regressions_under_and_against_the_hypothesis():
    ts = term_structure(_eh_world(n_months=8000))  # long sample: persistent regressors make short samples noisy
    fb = fama_bliss(ts).set_index("n")
    cs = campbell_shiller(ts).set_index("n")
    fbs = fama_bliss_short(ts).set_index("n")
    assert abs(fb.loc[3, "slope"]) < 2.5 * fb.loc[3, "se"] and abs(fb.loc[3, "slope"]) < 0.3
    assert abs(cs.loc[3, "slope"] - 1) < 2.5 * cs.loc[3, "se"] and abs(fbs.loc[2, "slope"] - 1) < 2.5 * fbs.loc[2, "se"]
    ts2 = term_structure(_eh_world(premium=True, seed=1))
    fb2 = fama_bliss(ts2).set_index("n")
    cs2 = campbell_shiller(ts2).set_index("n")
    assert fb2.loc[5, "slope"] > 0.4 and cs2.loc[5, "slope"] < 0.6


def test_full_lab_runs_on_simulated_data():
    import lab12

    with tempfile.TemporaryDirectory() as tmp:
        s = lab12.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig12_*.png"))) == 5
        rep = (Path(tmp) / "lab12_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        assert s["fama_bliss"][3]["slope"] > 0.2  # time-varying premium built into the simulation
        assert s["campbell_shiller"][-1]["slope"] < 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
