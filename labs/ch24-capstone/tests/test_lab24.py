"""Tests for Lab 24. Run with:  python tests/test_lab24.py   (or: pytest tests/)

Runs offline: the analytic power formulas against simulation, the R-squared sample size, multiple-testing
thresholds, Bayesianized p-values against Harvey's example, the specification grid, the forking-paths experiment's
false-positive rates, the pre-registration hash, and the full lab on simulated data.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.design import (months_for_r2, multiple_testing_threshold, noise_grid_experiment,  # noqa: E402
                            posterior_null, power_for_sharpe, preregistration, simulated_power, specification_grid,
                            verify_preregistration, years_for_sharpe)


def test_power_formulas():
    y = years_for_sharpe(0.5)
    assert abs(y - ((1.96 + 0.8416) / 0.5) ** 2) < 0.01
    assert abs(power_for_sharpe(0.5, y) - 0.8) < 1e-3
    assert abs(simulated_power(0.5, y, df=np.inf, n_sims=4000) - 0.8) < 0.03


def test_r2_and_multiple_testing():
    assert 3000 < months_for_r2(0.0025) < 3200        # (2.80)^2 x 0.9975 / 0.0025
    assert abs(multiple_testing_threshold(1) - 1.96) < 0.01
    assert multiple_testing_threshold(300) > 3.4


def test_bayesianized_p():
    # Harvey (2017): p = 0.05 with prior odds 4:1 against gives about a 37% chance the null is true
    assert abs(posterior_null(1.96, 4) - 0.37) < 0.02


def test_specification_grid():
    f = lambda a, b: {"estimate": a * b, "t": a + b}  # noqa: E731
    g = specification_grid(f, {"a": [1, 2], "b": [3, 4, 5]})
    assert len(g) == 6 and list(g["estimate"]) == sorted(g["estimate"])


def test_forking_paths():
    n = noise_grid_experiment(n_months=240, n_assets=40, n_trials=60, seed=1)
    assert n["pre_specified_rate"] < 0.15
    assert n["any_spec_rate"] > n["pre_specified_rate"] + 0.1


def test_preregistration_hash():
    fields = {k: "x" for k in ("title", "claim", "hypotheses", "data", "sample", "primary", "secondary", "power",
                               "decision", "reporting")}
    text = preregistration(fields)
    assert verify_preregistration(text)
    assert not verify_preregistration(text.replace("## 5. Primary test\nx", "## 5. Primary test\ny"))


def test_full_lab_runs_on_simulated_data():
    import lab24

    with tempfile.TemporaryDirectory() as tmp:
        s = lab24.main(["--source", "sim", "--noise-trials", "40", "--out", tmp])
        assert len(list(Path(tmp).glob("fig24_*.png"))) == 5
        assert verify_preregistration((Path(tmp) / "preregistration.md").read_text())
        assert s["spec_n"] == 36


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
