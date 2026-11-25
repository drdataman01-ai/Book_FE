"""Tests for Lab 20. Run with:  python tests/test_lab20.py   (or: pytest tests/)

Runs offline: the rolling-origin evaluation uses only the past, the AR and
HAR-type forecasters recover known dynamics, the scores and the before/after
cutoff split behave as expected, the contamination demo shows its signature,
missing foundation-model packages are reported rather than crashing, and the
full lab runs on simulated data. The foundation-model adapters themselves are
not tested here (their packages are not installed in the build environment).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.tsfm import (Memorizer, StandIn, f_ar, f_har, load_foundation_models, rolling_origin,  # noqa: E402
                          scores, split_scores)


def _ar1(T=600, phi=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.1 + phi * x[t - 1] + rng.standard_normal()
    return pd.Series(x, index=pd.period_range("1960-01", periods=T, freq="M"))


def test_rolling_origin_uses_only_the_past():
    s = _ar1()
    fc = rolling_origin(s, "2000-01", models=[StandIn()])
    t = pd.Period("2005-06", "M")
    s2 = s.copy()
    s2.loc[t:] += 10.0                      # change the target month and everything after it
    fc2 = rolling_origin(s2, "2000-01", models=[StandIn()])
    cols = ["AR (BIC)", "HAR-type (1, 3, 12)", "Naive (last value)", "Stand-in (not a foundation model)"]
    assert np.allclose(fc.loc[t, cols].astype(float), fc2.loc[t, cols].astype(float))   # forecast for t made at t-1
    assert fc.loc[t, "actual"] == s.loc[t]
    assert fc.index.min() == pd.Period("2000-01", "M")


def test_ar_and_har_recover_dynamics():
    s = _ar1(T=2000, phi=0.7, seed=1).to_numpy()
    f, sd = f_ar(s, window=2000)
    expected = 0.1 + 0.7 * s[-1]
    assert abs(f - expected) < 0.15 and abs(sd - 1) < 0.1
    f2, _ = f_har(s, window=2000)
    assert abs(f2 - expected) < 0.25


def test_scores_and_split():
    s = _ar1(T=500, seed=2)
    fc = rolling_origin(s, "1990-01")
    fc["Perfect"] = fc["actual"]
    t = scores(fc).set_index("model")
    assert t.loc["Perfect", "mse_ratio_vs_AR (BIC)"] == 0 and t.loc["Perfect", "dm_t_vs_AR (BIC)"] > 2
    assert t.loc["Naive (last value)", "mse_ratio_vs_AR (BIC)"] > 1          # AR(1) with phi 0.7 beats the naive
    assert 0.7 < t.loc["AR (BIC)", "coverage_80"] < 0.9


def test_memorizer_signature():
    s = _ar1(T=500, seed=3)
    mem = Memorizer(s, "1995-01")
    fc = rolling_origin(s, "1990-01", memorizer=mem)
    sp = split_scores(fc, "1995-01")
    m = sp[sp.model == Memorizer.name].set_index("window")["mse_ratio_vs_AR (BIC)"]
    assert m["before cutoff"] < 0.5 and abs(m["after cutoff"] - 1) < 1e-9


def test_missing_packages_reported():
    models, msgs = load_foundation_models(["chronos-bolt-small", "timesfm", "standin"])
    assert any(isinstance(m, StandIn) for m in models)
    if importlib.util.find_spec("chronos") is None:
        assert any("chronos-forecasting" in m for m in msgs)
    if importlib.util.find_spec("timesfm") is None:
        assert any("timesfm" in m for m in msgs)


def test_full_lab_runs_on_simulated_data():
    import lab20

    with tempfile.TemporaryDirectory() as tmp:
        s = lab20.main(["--source", "sim", "--models", "standin", "--out", tmp])
        assert len(list(Path(tmp).glob("fig20_*.png"))) == 5
        rep = (Path(tmp) / "lab20_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        sc = pd.DataFrame(s["scores"])
        lrv = sc[sc.series == "Log realized variance"].set_index("model")
        assert lrv.loc["Historical mean", "mse_ratio_vs_AR (BIC)"] > 2        # persistent series: the mean is poor
        ret = sc[sc.series == "Market excess return"].set_index("model")
        assert ret.loc["Naive (last value)", "mse_ratio_vs_AR (BIC)"] > 1.5   # iid returns: last value is poor


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
