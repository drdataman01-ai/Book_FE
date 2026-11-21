"""Tests for Lab 3. Run with:  python tests/test_lab03.py   (or: pytest tests/)

Runs offline: the 25-portfolio parser on a file in the French library's
layout, the regressions against a direct least-squares calculation, the
false-discovery experiment against its known rate, the break-even cost
formula, and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lab03  # noqa: E402
from fintest.data import find_table, parse_french_tables  # noqa: E402

HEADER = ",SMALL LoBM,ME1 BM2,ME1 BM3,ME1 BM4,SMALL HiBM,ME2 BM1,ME2 BM2,ME2 BM3,ME2 BM4,ME2 BM5," \
         "ME3 BM1,ME3 BM2,ME3 BM3,ME3 BM4,ME3 BM5,ME4 BM1,ME4 BM2,ME4 BM3,ME4 BM4,ME4 BM5," \
         "BIG LoBM,ME5 BM2,ME5 BM3,ME5 BM4,BIG HiBM"
P25_SAMPLE = (
    "This file was created using the 202508 CRSP database.\n\n"
    "  Average Value Weighted Returns -- Monthly\n" + HEADER + "\n"
    "192607," + ",".join(f"{0.1 * i:.2f}" for i in range(25)) + "\n"
    "192608," + ",".join(f"{0.2 * i:.2f}" for i in range(25)) + "\n\n"
    "  Average Equal Weighted Returns -- Monthly\n" + HEADER + "\n"
    "192607," + ",".join("1.00" for _ in range(25)) + "\n"
)


def test_parser_and_labels():
    vw = lab03.label_25(find_table(parse_french_tables(P25_SAMPLE), "value", "weight"))
    assert list(vw.columns[:2]) == ["S1B1", "S1B2"] and vw.columns[-1] == "S5B5"
    assert abs(vw.loc[pd.Period("1926-08", "M"), "S5B5"] - 0.048) < 1e-12  # 4.80% -> decimal


def test_regressions_match_least_squares():
    d = lab03.load("sim", sim_seed=1)
    Y, F = lab03.panel(d)
    a = lab03.part_a(Y, F)
    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    b = np.linalg.lstsq(X, Y["S3B3"].to_numpy(), rcond=None)[0]
    row = a["Three-factor"]["table"].set_index("portfolio").loc["S3B3"]
    assert abs(row.alpha_annual - 12 * b[0]) < 1e-10 and abs(row["beta_HML"] - b[3]) < 1e-10
    # Nested models: the three-factor R2 is never below the CAPM R2.
    assert (a["Three-factor"]["table"]["r2"] >= a["CAPM"]["table"]["r2"] - 1e-12).all()
    # The simulation gives only S1B1 a true alpha (-4.8% a year).
    assert a["Three-factor"]["table"].set_index("portfolio").loc["S1B1", "alpha_annual"] < -0.02


def test_false_discoveries_match_the_test_size():
    b = lab03.part_b(n_months=600, n_factors=1000, n_reps=60, seed=3)
    t = b["table"].set_index(["design", "threshold"])
    ind = t.loc[("Independent factors", 1.96), "mean_false_discoveries"]
    assert 42 < ind < 58  # about 5% of 1,000
    assert t.loc[("Independent factors", 3.0), "mean_false_discoveries"] < 6


def test_break_even_cost():
    idx = pd.period_range("1963-07", periods=240, freq="M")
    F = pd.DataFrame({"Mkt-RF": 0.005, "SMB": 0.002, "HML": 0.004}, index=idx)
    F = F + np.random.default_rng(0).normal(0, 1e-9, F.shape)
    c = lab03.part_c(F, turnover=0.05)
    be = c["break_even"].set_index("factor")
    assert np.isnan(be.loc["Mkt-RF", "break_even_one_way_cost"])
    assert abs(be.loc["HML", "break_even_one_way_cost"] - 0.048 / (4 * 12 * 0.05)) < 1e-6


def test_full_lab_runs_on_simulated_data():
    with tempfile.TemporaryDirectory() as tmp:
        s = lab03.main(["--source", "sim", "--n-reps", "20", "--out", tmp])
        assert s["simulated"] is True
        assert len(list(Path(tmp).glob("fig3_*.png"))) == 5
        assert (Path(tmp) / "lab03_report.md").exists()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
