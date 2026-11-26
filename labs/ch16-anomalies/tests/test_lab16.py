"""Tests for Lab 16. Run with:  python tests/test_lab16.py   (or: pytest tests/)

Runs offline: the Open Source Asset Pricing readers on files in their
layouts (wide and long, percent units), period statistics against direct
calculation, the pooled decay regression on data with a known decline,
empirical Bayes shrinkage, and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.anomalies import empirical_bayes, mclean_pontiff, read_ls_returns, read_signal_doc, signal_periods  # noqa: E402


def test_readers_wide_long_and_doc():
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp) / "wide.csv"
        w.write_text("date,AccrualsBM,Mom12m\n1990-01-31,1.5,-2.0\n1990-02-28,0.5,3.0\n")
        r = read_ls_returns(w)
        assert abs(r.loc["1990-01", "AccrualsBM"] - 0.015) < 1e-12  # percent converted
        lg = Path(tmp) / "long.csv"
        lg.write_text("signalname,port,date,ret\nMom12m,LS,1990-01-31,2.0\nMom12m,01,1990-01-31,9.0\n"
                      "Mom12m,LS,1990-02-28,1.0\n")
        r2 = read_ls_returns(lg)
        assert list(r2.columns) == ["Mom12m"] and abs(r2.iloc[0, 0] - 0.02) < 1e-12
        doc = Path(tmp) / "SignalDoc.csv"
        doc.write_text("Acronym,Authors,Year,SampleStartYear,SampleEndYear,LongDescription\n"
                       "Mom12m,Jegadeesh and Titman,1993,1965,1989,Momentum\n")
        dd = read_signal_doc(doc)
        assert dd.loc["Mom12m", "pub_year"] == 1993 and dd.loc["Mom12m", "sample_end"] == 1989


def _toy(decline=0.5, n_sig=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1960-01", "2019-12", freq="M")
    yrs = idx.year.to_numpy()
    cols, docs = {}, []
    for i in range(n_sig):
        mu = 0.005
        r = mu + 0.01 * rng.standard_normal(len(idx))
        r[yrs > 2000] -= decline * mu
        cols[f"s{i}"] = r
        docs.append({"signal": f"s{i}", "pub_year": 2000, "sample_start": 1960, "sample_end": 1995})
    return pd.DataFrame(cols, index=idx), pd.DataFrame(docs).set_index("signal")


def test_periods_and_decay_regression():
    ret, doc = _toy(decline=0.5)
    st = signal_periods(ret, doc)
    x = ret["s0"]
    assert abs(st.loc["s0", "mean_in"] - x[(x.index.year >= 1960) & (x.index.year <= 1995)].mean()) < 1e-15
    mp = mclean_pontiff(ret, doc, st)
    assert abs(mp["coef"]["post_pub"] + 0.5) < 0.08 and abs(mp["coef"]["post_sample"]) < 0.1


def test_empirical_bayes_shrinks_noisy_estimates_more():
    m = np.array([0.01, 0.01, 0.0, 0.002])
    s = np.array([0.001, 0.01, 0.001, 0.005])
    eb = empirical_bayes(m, s)
    assert eb["shrink"][1] > eb["shrink"][0]
    assert abs(eb["post_mean"][1] - eb["mu0"]) < abs(m[1] - eb["mu0"])


def test_selection_creates_apparent_decay():
    import lab16

    e = lab16.part_e(seed=1, n_signals=120).set_index("true_post_publication_decay")
    assert e.loc[0.0, "measured_post_pub_change"] < -0.1  # no true decay, yet returns appear to fall


def test_full_lab_runs_on_simulated_data():
    import lab16

    with tempfile.TemporaryDirectory() as tmp:
        s = lab16.main(["--source", "sim", "--out", tmp])
        assert len(list(Path(tmp).glob("fig16_*.png"))) == 4
        rep = (Path(tmp) / "lab16_report.md").read_text()
        assert "Simulated data" in rep and "Questions to think about" in rep
        assert s["n_signals"] > 100


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
