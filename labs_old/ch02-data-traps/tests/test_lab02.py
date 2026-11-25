"""Tests for Lab 2. Run with:  python tests/test_lab02.py   (or: pytest tests/)

No network access is needed: the parsers are tested on files written in the
French library's layouts, and the full lab runs on simulated data.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.data import find_table, load_french_tables, parse_french_tables  # noqa: E402
from fintest.stats import geometric_mean, max_drawdown  # noqa: E402

# Laid out like Portfolios_Formed_on_ME.csv: several tables, each with a
# title line; monthly and annual versions; a firm-count table in units.
SIZE_SAMPLE = """This file was created by CMPT_ME_RETS using the 202508 CRSP database.
It contains value- and equal-weighted returns for size portfolios.

  Average Value Weighted Returns -- Monthly
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
192607,-99.99,0.12,1.30,3.20,-0.10,1.10,1.40,1.60,3.30,-0.50,0.20,1.00,1.20,1.30,1.50,1.40,1.80,2.60,3.60
192608,-99.99,2.30,2.10,2.70,2.40,2.00,2.20,2.30,2.80,2.60,2.20,2.00,2.00,2.10,2.30,2.30,2.30,2.50,2.90

  Average Equal Weighted Returns -- Monthly
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
192607,-99.99,0.50,1.20,3.00,0.40,1.00,1.30,1.50,3.10,0.30,0.50,1.00,1.10,1.20,1.40,1.30,1.70,2.40,3.20
192608,-99.99,2.60,2.00,2.60,2.70,2.10,2.10,2.20,2.70,3.10,2.30,2.00,2.00,2.00,2.20,2.20,2.20,2.40,2.80

  Average Value Weighted Returns -- Annual
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
1927,-99.99,20.0,30.0,33.0,18.0,28.0,30.0,31.0,33.0,15.0,22.0,27.0,29.0,30.0,30.0,31.0,31.0,32.0,34.0

  Number of Firms in Portfolios
,<= 0,Lo 30,Med 40,Hi 30,Lo 20,Qnt 2,Qnt 3,Qnt 4,Hi 20,Lo 10,Dec 2,Dec 3,Dec 4,Dec 5,Dec 6,Dec 7,Dec 8,Dec 9,Hi 10
192607,0,149,198,149,99,99,99,99,99,50,49,50,49,50,49,50,49,50,49
192608,0,149,198,149,99,99,99,99,99,50,49,50,49,50,49,50,49,50,49
"""


def test_parse_multi_table_size_file():
    tables = parse_french_tables(SIZE_SAMPLE, "M")
    titles = list(tables)
    assert any("Value Weighted Returns -- Monthly" in t for t in titles)
    assert any("Equal Weighted Returns -- Monthly" in t for t in titles)
    assert not any("Annual" in t for t in titles), "annual tables must be skipped"
    vw = find_table(tables, "value", "weight")
    assert vw.shape == (2, 19)
    assert abs(vw.loc["1926-07", "Hi 10"] - 0.036) < 1e-12
    assert np.isnan(vw.loc["1926-07", "<= 0"])
    firms = find_table(tables, "number", "firms")
    assert firms.loc["1926-07", "Lo 10"] == 50, "counts must stay in units, not percent"


def test_decile_columns_are_the_last_ten():
    from lab02 import decile_columns

    vw = find_table(parse_french_tables(SIZE_SAMPLE, "M"), "value", "weight")
    dec = decile_columns(vw)
    assert list(dec.columns) == [f"D{i}" for i in range(1, 11)]
    assert abs(dec.loc["1926-07", "D1"] - (-0.005)) < 1e-12
    assert abs(dec.loc["1926-07", "D10"] - 0.036) < 1e-12


def test_load_tables_from_zip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Portfolios_Formed_on_ME_CSV.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Portfolios_Formed_on_ME.csv", SIZE_SAMPLE)
        path.write_bytes(buf.getvalue())
        tables = load_french_tables("size_portfolios", path=path)
        assert len(tables) == 3


def test_return_statistics():
    r = np.array([0.10, -0.10])
    assert abs(geometric_mean(r, periods=1) - (np.sqrt(1.1 * 0.9) - 1)) < 1e-12
    assert abs(max_drawdown(np.array([0.1, -0.5, 0.2])) - (-0.5)) < 1e-12


def test_lookahead_is_detected_on_known_data():
    """With look-ahead, a moving-average rule must beat the correctly lagged version."""
    from fintest.data import simulated_daily_market
    from lab02 import part_c

    c = part_c(simulated_daily_market(n_days=6000, seed=3), window=50, grid=(20, 50, 100), warmup_years=3)
    p = c["performance"]
    la = [k for k in p.index if "look-ahead" in k][0]
    ok = [k for k in p.index if "correctly" in k][0]
    assert p.loc[la, "sharpe"] > p.loc[ok, "sharpe"] + 0.3
    assert c["hindsight_sharpe"] >= c["walk_forward_sharpe"] - 1e-9


def test_survivorship_simulation_direction():
    """Survivors-only data must overstate returns relative to the full, investable data."""
    from lab02 import part_b

    b = part_b(seed=5, n_firms=800, n_months=240)
    t = b["table"].set_index("dataset")
    full, missing, surv = t["ew_return"]
    assert surv > missing > full
    assert t["vol_spread"].iloc[2] > t["vol_spread"].iloc[0]
    st = b["stats"]
    assert st["survivors"] == 800 and st["performance_delistings"] > 0


def test_no_lookahead_in_volatility_sort():
    """The volatility sort for month t must use only months before t."""
    from lab02 import _vol_spread

    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.1, (40, 200))
    base = _vol_spread(R)
    R2 = R.copy()
    R2[30:] = rng.normal(0, 0.1, (10, 200))  # change the future only
    assert np.allclose(base[:30], _vol_spread(R2)[:30], equal_nan=True)


def test_full_lab_runs_on_simulated_data():
    import lab02

    with tempfile.TemporaryDirectory() as tmp:
        summary = lab02.main(["--source", "sim", "--sim-firms", "500", "--sim-years", "15", "--out", tmp])
        out = Path(tmp)
        assert len(list(out.glob("fig2_*.png"))) == 5
        report = (out / "lab02_report.md").read_text()
        assert "Simulated data" in report and "Your verdict" in report
        assert summary["simulated"] is True
        assert pd.read_csv(out / "table_size_premium.csv").shape[0] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
