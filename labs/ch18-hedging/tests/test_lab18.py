"""Tests for Lab 18. Run with:  python tests/test_lab18.py   (or: pytest tests/)

Runs offline: the delta-hedging engine on Black-Scholes paths (mean near
zero, error shrinking with hedging frequency at the rate theory predicts),
the unhedged payoff, put-call equivalence of hedged positions, transaction
costs, the gamma-theta approximation, and the full lab on simulated data.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.options import bs_price, cash_gamma_pnl, delta_hedge  # noqa: E402


def _gbm(n, sigma, n_paths, seed, r=0.0, per_year=252):
    rng = np.random.default_rng(seed)
    dt = 1 / per_year
    lr = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rng.standard_normal((n_paths, n))
    return np.column_stack([np.ones(n_paths), np.exp(np.cumsum(lr, axis=1))])


def test_hedging_error_on_black_scholes_paths():
    n, sig = 21, 0.20
    rf = np.zeros(n)
    paths = _gbm(n, sig, 4000, 0)
    pnl = np.array([delta_hedge(p, 1.0, rf, sig)["pnl"] for p in paths])
    prem = float(bs_price(1, 1, n / 252, 0.0, sig))
    vega = 0.3989 * math.sqrt(n / 252)  # ATM vega per unit sigma for S = 1 (approximately)
    theory = math.sqrt(math.pi / 4) * sig * vega / math.sqrt(n)  # Kamal-Derman approximation
    assert abs(pnl.mean()) < 0.1 * pnl.std()
    assert 0.6 < pnl.std() / theory < 1.3
    paths4 = _gbm(4 * n, sig, 1500, 1, per_year=4 * 252)
    pnl4 = np.array([delta_hedge(p, 1.0, np.zeros(4 * n), sig, days_per_year=4 * 252)["pnl"] for p in paths4])
    assert pnl4.std() < 0.65 * pnl.std()  # four times as many rebalances: error roughly halves
    assert abs(prem - 0.0232) < 0.001


def test_unhedged_and_parity_and_costs():
    n = 21
    rf = np.full(n, 0.0002)
    p = _gbm(n, 0.25, 1, 3)[0]
    u = delta_hedge(p, 1.0, rf, 0.25, hedge_every=0)
    assert abs(u["pnl"] - (u["premium"] * np.prod(1 + rf) - max(p[-1] - 1, 0))) < 1e-12
    c = delta_hedge(p, 1.0, rf, 0.25, "call")
    q = delta_hedge(p, 1.0, rf, 0.25, "put")
    assert abs(c["pnl"] - q["pnl"]) < 1e-10  # hedged call and hedged put are the same position
    k = delta_hedge(p, 1.0, rf, 0.25, cost=0.001)
    assert k["costs"] > 0 and abs((c["pnl"] - k["pnl"]) / k["costs"] - 1) < 0.01  # costs plus a little interest


def test_gamma_theta_approximation():
    n, sig = 63, 0.20
    paths = _gbm(n, 0.30, 200, 4)  # realized volatility above implied: the seller should lose
    act = np.array([delta_hedge(p, 1.0, np.zeros(n), sig)["pnl"] for p in paths])
    app = np.array([cash_gamma_pnl(p, 1.0, np.zeros(n), sig) for p in paths])
    assert np.corrcoef(act, app)[0, 1] > 0.9 and act.mean() < 0 and app.mean() < 0


def test_full_lab_runs_on_simulated_data():
    import lab18

    with tempfile.TemporaryDirectory() as tmp:
        s = lab18.main(["--source", "sim", "--n-sims", "2", "--out", tmp])
        assert len(list(Path(tmp).glob("fig18_*.png"))) == 5
        rep = (Path(tmp) / "lab18_report.md").read_text()
        assert "Simulated data" in rep and "Your verdict" in rep
        he = {(r["paths"][:4], r["hedging"]): r for r in s["hedging_error"]}
        assert he[("Real", "Daily")]["risk_removed"] > 0.5
        assert abs(he[("Blac", "Daily")]["mean"]) < 0.05  # no premium on Black-Scholes paths


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
