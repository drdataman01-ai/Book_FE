"""Tests for Lab 4. Run with:  python tests/test_lab04.py   (or: pytest tests/)

Runs offline: momentum-file parsers on the library's layouts, volatility
scaling for look-ahead, time-series momentum signals, the crash regression
on simulated data with a known crash beta, and the full lab.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.data import find_table, parse_french_csv, parse_french_tables  # noqa: E402

MOM_DAILY = "This file was created using the CRSP database.\n\n,Mom   \n19261103,    0.56\n19261104,   -0.50\n"
DEC = ("  Average Value Weighted Returns -- Monthly\n,Lo PRIOR,PRIOR 2,PRIOR 3,PRIOR 4,PRIOR 5,PRIOR 6,PRIOR 7,PRIOR 8,"
       "PRIOR 9,Hi PRIOR\n192701," + ",".join(f"{i:.1f}" for i in range(10)) + "\n")


def test_parsers():
    d = parse_french_csv(MOM_DAILY, "D")
    assert len(d) == 2 and abs(d.iloc[0, 0] - 0.0056) < 1e-12
    t = find_table(parse_french_tables(DEC), "value", "weight")
    assert abs((t.iloc[:, -1] - t.iloc[:, 0]).iloc[0] - 0.09) < 1e-12


def test_vol_scaling_uses_only_past_data():
    import lab04

    d = lab04.simulated(n_months=120, seed=1)
    vs1 = lab04.vol_scaled(d["mom"], d["mom_daily"])
    dd = d["mom_daily"].copy()
    cut = pd.Period("1933-06", "M")
    dd[dd.index.to_period("M") > cut] *= 5  # make the future much more volatile
    vs2 = lab04.vol_scaled(d["mom"], dd)
    common = vs1.index[vs1.index <= cut + 1]  # weight for month cut+1 uses data to the end of month cut
    assert np.allclose(vs1.loc[common, "weight"], vs2.loc[common, "weight"])
    assert not np.allclose(vs1.loc[vs1.index > cut + 1, "weight"], vs2.loc[vs2.index > cut + 1, "weight"])
    # Weight is target over annualized volatility.
    r = vs1.iloc[-1]
    assert abs(r.weight * r.sigma_hat - 0.12) < 1e-12


def test_crash_regression_recovers_known_beta():
    import lab04

    d = lab04.simulated(seed=3)
    b = lab04.part_b(d)
    reg = b["regression"].set_index("term")["estimate"]
    assert reg["extra beta in bear-market rebounds"] < -0.6  # simulation sets -1.2 in bear-market up months
    assert abs(reg["beta (all months)"]) < 0.3


def test_tsmom_signal_is_lagged():
    import lab04

    d = lab04.simulated(n_months=240, seed=4)
    r1 = lab04.part_d(d)["pnl"]
    d2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}
    last = d2["industries"].index[-1]
    d2["industries"].loc[last] += 0.5  # a shock in the last month cannot change the positions held in it
    r2 = lab04.part_d(d2)["pnl"]
    pos_unchanged = np.allclose(r1.iloc[:-1], r2.iloc[:-1])
    assert pos_unchanged


def test_full_lab_runs_on_simulated_data():
    import lab04

    with tempfile.TemporaryDirectory() as tmp:
        s = lab04.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig4_*.png"))) == 5
        rep = (Path(tmp) / "lab04_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        vs = pd.DataFrame(s["vol_scaled"])
        assert vs.iloc[2]["max_drawdown"] > vs.iloc[1]["max_drawdown"]  # scaling reduces the worst drawdown


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
