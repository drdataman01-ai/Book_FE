"""Tests for Lab 6. Run with:  python tests/test_lab06.py   (or: pytest tests/)

Runs offline: the Goyal-Welch reader on a file in the published layout, the
predictor definitions, no look-ahead in the walk-forward forecasts, the
Campbell-Thompson restrictions, the known truth of the simulated data, the
Reality Check on rules with no skill, and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lab06  # noqa: E402
from fintest.forecast import walk_forward  # noqa: E402
from fintest.predictors import build_predictors, parse_goyal_welch  # noqa: E402

GW_SAMPLE = """yyyymm,Index,D12,E12,b/m,tbl,AAA,BAA,lty,ntis,Rfree,infl,ltr,corpr,svar,csp,CRSP_SPvw,CRSP_SPvwx
192612,"13.49",0.69,1.24,0.44,0.0307,0.0468,0.0595,0.0354,0.0509,0.0026,-0.0011,0.0078,0.0067,0.0003,,0.0256,0.0218
192701,13.21,0.70,1.23,0.45,0.0323,0.0466,0.0592,0.0351,0.0508,0.0025,-0.0112,0.0075,0.0072,0.0005,,-0.0021,-0.0062
"""


def test_reader_and_definitions():
    raw = parse_goyal_welch(GW_SAMPLE)
    p = build_predictors(raw)
    r = p.loc["1927-01"]
    assert abs(r["dp"] - (np.log(0.70) - np.log(13.21))) < 1e-12
    assert abs(r["tms"] - (0.0351 - 0.0323)) < 1e-12
    assert abs(r["dfy"] - (0.0592 - 0.0466)) < 1e-12
    assert abs(r["infl"] - (-0.0011)) < 1e-12  # inflation enters with a one-month publication lag
    assert abs(r["eqp"] - (-0.0021 - 0.0025)) < 1e-12


def test_walk_forward_uses_only_past_data():
    df = lab06.load("sim", sim_seed=2)
    z = lab06.next_premium(df)
    x = df["dp"].to_numpy()
    first = lab06._first_index(df, "1965-01")
    w1 = walk_forward(z, x, first, min_train=240)
    z2 = z.copy()
    z2[first + 100:] += 1.0  # change future premiums
    w2 = walk_forward(z2, x, first, min_train=240)
    assert np.allclose(w1["model"][first:first + 101], w2["model"][first:first + 101], equal_nan=True)


def test_campbell_thompson_forecasts_are_never_negative():
    df = lab06.load("sim", sim_seed=1)
    fc = lab06.forecasts(df)
    for n in lab06.NAMES:
        m = fc["fc"][n]["ct"]["model"]
        assert np.nanmin(m) >= 0


def test_simulated_truth_is_recovered():
    df = lab06.load("sim", sim_seed=0)
    a = lab06.part_a(df).set_index("predictor")
    assert a.loc["dp", "t_white"] > 2  # the only true predictor in the simulation
    b = lab06.part_b(df, lab06.forecasts(df))["table"].set_index("predictor")
    assert b.loc["dp", "r2_oos"] > 0


def test_reality_check_with_no_skill():
    e = lab06.part_e(n_days=3000, n_boot=200, seed=4)
    assert e["n_rules"] > 500 and e["p_reality_check"] > 0.05


def test_full_lab_runs_on_simulated_data():
    with tempfile.TemporaryDirectory() as tmp:
        s = lab06.main(["--source", "sim", "--n-boot", "100", "--out", tmp])
        assert s["simulated"] is True
        assert len(list(Path(tmp).glob("fig6_*.png"))) == 5
        assert (Path(tmp) / "lab06_report.md").exists()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
