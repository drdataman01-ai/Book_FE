"""Tests for Lab 13. Run with:  python tests/test_lab13.py   (or: pytest tests/)

Runs offline. Selling rules are checked on hand-made price paths and for
look-ahead; optional stopping and Wald's identity are checked by
simulation; tax accounting is checked by hand; the disposition test is
checked on a panel built so that losers keep losing; the industry-file
parser is checked on the library's layout; and the full lab runs on
simulated data.
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

from fintest.data import find_table, parse_french_tables  # noqa: E402
from fintest.holding import (  # noqa: E402
    Rule,
    disposition_test,
    entries_from_panel,
    excess_per_year,
    exit_times,
    mean_log_return,
    simulate_paths,
    simulate_taxes,
    standard_rules,
    summarize,
)

# gains P_k / P_0 - 1 at the end of months 1..6
PATHS = np.array([
    [0.05, 0.12, 0.21, 0.30, 0.10, 0.00],   # reaches +20% in month 3
    [-0.10, -0.22, -0.15, -0.02, 0.01, 0.25],  # falls 20% in month 2, back to cost in month 5
    [0.02, -0.05, 0.03, -0.04, 0.06, -0.01],  # never touches a threshold
])


def test_rules_on_hand_made_paths():
    tau, hpr, why = exit_times(PATHS, Rule("t", "target", up=0.2, cap=6))
    assert list(tau) == [3, 6, 6] and list(why) == ["target", "target", "cap"]
    assert abs(hpr[0] - 0.21) < 1e-12 and abs(hpr[1] - 0.25) < 1e-12
    tau, hpr, why = exit_times(PATHS, Rule("s", "stop", down=0.2, cap=6))
    assert list(tau) == [6, 2, 6] and list(why) == ["cap", "stop", "cap"]
    tau, _, why = exit_times(PATHS, Rule("b", "bracket", up=0.2, down=0.2, cap=6))
    assert list(tau) == [3, 2, 6] and list(why) == ["target", "stop", "cap"]
    tau, hpr, why = exit_times(PATHS, Rule("g", "get_even", up=0.2, down=0.2, cap=6))
    assert list(tau) == [3, 5, 6] and list(why) == ["target", "even", "cap"]
    assert abs(hpr[1] - 0.01) < 1e-12
    tau, hpr, why = exit_times(PATHS, Rule("f", "fixed", cap=4))
    assert list(tau) == [4, 4, 4] and list(why) == ["cap"] * 3 and abs(hpr[2] + 0.04) < 1e-12


def test_no_look_ahead():
    """Changing prices after the sale cannot change when or at what return a position is sold."""
    rng = np.random.default_rng(1)
    rel = simulate_paths(2000, 60, sigma=0.3, seed=3)
    for rule in standard_rules(cap=60):
        tau, hpr, _ = exit_times(rel, rule)
        noisy = rel.copy()
        for j, t in enumerate(tau):
            noisy[j, t:] = rng.uniform(-0.9, 3.0, 60 - t)
        tau2, hpr2, _ = exit_times(noisy, rule)
        assert np.array_equal(tau, tau2) and np.allclose(hpr, hpr2), rule.name


def _summ(res):
    tau, hpr, why = res
    return summarize(hpr, tau, why)


def test_optional_stopping():
    """With martingale prices and a bounded horizon, every rule's mean holding-period return is zero."""
    rel = simulate_paths(200000, 60, mu=0.0, sigma=0.3, seed=11)
    for rule in standard_rules(cap=60):
        tau, hpr, why = exit_times(rel, rule)
        s = summarize(hpr, tau, why)
        assert abs(s["mean_hpr"]) < 4 * s["se_mean_hpr"], (rule.name, s["mean_hpr"], s["se_mean_hpr"])


def test_rules_reshape_the_distribution():
    rel = simulate_paths(50000, 120, mu=0.0, sigma=0.3, seed=5)
    rules = {r.kind + str(r.cap): r for r in standard_rules()}
    s_t = _summ(exit_times(rel, rules["target120"]))
    s_s = _summ(exit_times(rel, rules["stop120"]))
    assert s_t["win_rate"] > 0.6 > 0.2 > s_s["win_rate"]
    assert s_t["skewness"] < 0 < s_s["skewness"]


def test_wald_identity():
    """Log return per year invested equals mu - sigma^2/2 for every rule."""
    mu, sigma = 0.08, 0.3
    rel = simulate_paths(100000, 120, mu=mu, sigma=sigma, seed=7)
    for rule in standard_rules():
        s = _summ(exit_times(rel, rule))
        assert abs(s["per_year_log"] - (mu - sigma**2 / 2)) < 0.004, (rule.name, s["per_year_log"])


def test_naive_annualizing_rewards_quick_wins():
    rel = simulate_paths(50000, 120, mu=0.08, sigma=0.3, seed=8)
    rules = {r.kind + str(r.cap): r for r in standard_rules()}
    t = _summ(exit_times(rel, rules["target120"]))
    f = _summ(exit_times(rel, rules["fixed120"]))
    assert t["naive_annual"] > 0.3 > f["naive_annual"]


def test_tax_accounting_by_hand():
    # no volatility and no drift: price never moves, so only costs matter
    r = simulate_taxes("every", years=2, n=3, mu=0.0, sigma=0.0, tax=0.2, cost=0.01, every=12)
    # buy: 0.99 of value for basis 1; sell after 12 months: 0.9801 cash, loss 0.0199 -> credit 0.00398
    cash = 0.99 * 0.99 + 0.2 * (1 - 0.99 * 0.99)
    value = cash * 0.99
    final = value * 0.99 + 0.2 * (cash - value * 0.99)
    assert abs(r["mean_wealth"] - final) < 1e-12 and abs(r["sales_per_year"] - 0.5) < 1e-12
    # without tax or cost, buy-and-hold wealth is the price relative
    r = simulate_taxes("hold", years=1, n=4, mu=0.12, sigma=0.0, tax=0.0, cost=0.0)
    assert abs(r["mean_wealth"] - math.exp(0.12)) < 1e-12


def test_taxes_favor_deferral():
    res = {k: simulate_taxes(k, years=30, n=20000, seed=4)["mean_wealth"] for k in ("hold", "winners", "losers")}
    assert res["losers"] > res["hold"] > res["winners"]


def _panel(T=600, N=20, rho=0.0, seed=0):
    """Monthly returns with optional persistence in each asset's expected return (rho > 0: losers keep losing)."""
    rng = np.random.default_rng(seed)
    mu = np.zeros((T, N))
    for t in range(1, T):
        mu[t] = rho * mu[t - 1] + 0.004 * rng.standard_normal(N)
    r = 0.007 + mu + 0.05 * rng.standard_normal((T, N))
    idx = pd.period_range("1950-01", periods=T, freq="M")
    return pd.DataFrame(r, index=idx)


def test_entries_and_excess():
    df = _panel()
    df.iloc[:30, 3] = np.nan  # an asset that starts late
    pos = entries_from_panel(df, cap=24)
    assert pos["rel"].shape[1] == 24 and not np.isnan(pos["rel"]).any()
    assert (pos["start"][pos["asset"] == 3] >= 30).all()
    j = 5
    e, a = pos["start"][j], pos["asset"][j]
    direct = np.prod(1 + df.iloc[e:e + 24, a].to_numpy()) - 1
    assert abs(pos["rel"][j, -1] - direct) < 1e-10
    g = mean_log_return(df)
    tau, hpr, _ = exit_times(pos["rel"], Rule("f", "fixed", cap=24))
    ex = excess_per_year(pos, tau, hpr, g, block=24, n_boot=50)
    assert abs(ex["excess_per_year"]) < 0.01 and ex["ci_low"] < ex["excess_per_year"] < ex["ci_high"]


def test_disposition_detects_persistence():
    """When expected returns persist, the winners sold beat the losers held; without persistence, no gap."""
    rule = Rule("t", "target", up=0.2, cap=36)
    d1 = disposition_test(entries_from_panel(_panel(rho=0.98, seed=2), cap=36), rule)
    assert d1["diff_mean"] > 0 and d1["diff_t"] > 2
    d0 = disposition_test(entries_from_panel(_panel(rho=0.0, seed=2), cap=36), rule)
    assert abs(d0["diff_t"]) < 3


INDUSTRY_SAMPLE = """This file was created using the 202506 CRSP database.

  Average Value Weighted Returns -- Monthly
,Agric,Food ,Soda ,Beer
192607,   2.37,   0.12, -99.99,  -5.19
192608,   2.23,   2.68, -99.99,  27.03
192609,  -0.57,   1.58, -99.99,   4.02

  Average Equal Weighted Returns -- Monthly
,Agric,Food ,Soda ,Beer
192607,   1.00,   1.00, -99.99,   1.00
192608,   1.00,   1.00, -99.99,   1.00
"""


def test_parse_industry_file():
    tables = parse_french_tables(INDUSTRY_SAMPLE)
    vw = find_table(tables, "value weighted", "monthly")
    assert vw.shape == (3, 4) and abs(vw.iloc[1, 3] - 0.2703) < 1e-12 and vw.iloc[:, 2].isna().all()
    eq = find_table(tables, "equal weighted", "monthly")
    assert abs(eq.iloc[0, 0] - 0.01) < 1e-12


def test_full_lab_on_simulated_data():
    import lab13

    with tempfile.TemporaryDirectory() as tmp:
        args = lab13.parse_args(["--source", "sim", "--out", tmp, "--paths", "20000", "--tax-paths", "4000",
                                 "--boot", "60"])
        s = lab13.run(args)
        assert s["simulated"] is True
        for name in ["lab13_report.md", "summary.json", "fig13_1_distributions.png", "fig13_2_per_year.png",
                     "fig13_3_real_prices.png", "fig13_4_disposition.png", "fig13_5_taxes.png"]:
            assert (Path(tmp) / name).exists(), name
        assert "SIMULATED" in (Path(tmp) / "lab13_report.md").read_text().upper()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
