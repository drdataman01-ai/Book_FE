"""Lab 18: Does Delta Hedging Remove Option Risk?

Chapter 18 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "An option seller who delta-hedges as Black-Scholes prescribes is left
   with little risk."

WHAT THE LAB DOES
  Each month since 1990, sell a one-month at-the-money call on the U.S.
  market at its Black-Scholes price, using the VIX as the implied volatility,
  and delta-hedge it along the market's actual daily path. Repeat the same
  exercise on simulated Black-Scholes paths, where the theory holds exactly.

  Part A  Hedging error on real and simulated paths: how much risk is left?
  Part B  Where the error comes from: realized versus implied variance along
          the path (the gamma-theta approximation) and what it misses.
  Part C  Stress tests: hedging frequency, transaction costs, crash months.
  Part D  The worst months and the tail of the hedging error.

DATA (free)
  Kenneth R. French Data Library: Fama/French 3 Factors [Daily].
  FRED: VIXCLS (from 1990).

USAGE
  python lab18.py
  python lab18.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, load_french_factors
from fintest.inference import moments
from fintest.options import cash_gamma_pnl, delta_hedge
from fintest.vol import garch_forecast_sum, load_fred

ROOT = Path(__file__).resolve().parent
FREQS = [1, 2, 5, 10, 0]  # hedge every n days; 0 = never
FREQ_NAMES = {1: "Daily", 2: "Every 2 days", 5: "Weekly", 10: "Every 2 weeks", 0: "Unhedged"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        return simulated(seed)
    ff = load_french_factors("ff3_daily", path=files.get("ff3"))
    cols = {c.strip().lower(): c for c in ff.columns}
    rf = ff[cols["rf"]]
    return {"ret": (ff[cols["mkt-rf"]] + rf).rename("ret"), "rf": rf, "vix": load_fred("VIXCLS", files.get("vix")),
            "simulated": False}


def simulated(seed: int = 0, n: int = 9500) -> dict:
    """GJR-GARCH daily returns with t shocks and a VIX priced 12% above expected volatility. FOR TESTING ONLY."""
    rng = np.random.default_rng(seed)
    om, a, g, b = 1.2e-6, 0.03, 0.10, 0.905
    p = {"long_run_var": om / (1 - a - b - g / 2), "persistence": a + b + g / 2}
    r, s2 = np.zeros(n), np.zeros(n)
    s2[0] = p["long_run_var"]
    z = rng.standard_t(5, n) / math.sqrt(5 / 3)
    for t in range(1, n):
        s2[t] = om + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = 0.0004 + math.sqrt(s2[t]) * z[t]
    idx = pd.bdate_range("1988-01-04", periods=n)
    nxt = om + (a + g * (r < 0)) * r ** 2 + b * s2
    vix = 100 * np.sqrt(garch_forecast_sum(nxt, p, 21) / 21 * 252) * 1.12 * np.exp(0.04 * rng.standard_normal(n))
    vix_s = pd.Series(vix, idx)
    return {"ret": pd.Series(r, idx, name="ret"), "rf": pd.Series(0.00012, idx),
            "vix": vix_s[vix_s.index >= "1990-01-02"], "simulated": True}


def month_paths(d: dict) -> list[dict]:
    """One record per month: the index path from the previous month end, daily rates, and the VIX at the start."""
    df = pd.concat({"ret": d["ret"], "rf": d["rf"].reindex(d["ret"].index).fillna(0.0),
                    "vix": d["vix"].reindex(d["ret"].index).ffill(limit=5)}, axis=1).dropna(subset=["ret"])
    ends = df.groupby(df.index.to_period("M")).tail(1).index
    out = []
    for t0, t1 in zip(ends[:-1], ends[1:]):
        sig = df.at[t0, "vix"]
        if not np.isfinite(sig):
            continue
        seg = df.loc[(df.index > t0) & (df.index <= t1)]
        S = np.concatenate([[1.0], np.cumprod(1 + seg["ret"].to_numpy())])
        lr = np.diff(np.log(S))
        out.append({"month": t1.to_period("M"), "S": S, "rf": seg["rf"].to_numpy(), "sigma": sig / 100,
                    "realized": float(np.sqrt(np.sum(lr ** 2) * 252 / len(lr))), "market": float(S[-1] - 1),
                    "max_abs_day": float(np.max(np.abs(lr)))})
    return out


def bs_paths(m: dict, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Simulated geometric Brownian motion paths with the month's implied volatility and day count."""
    n = len(m["S"]) - 1
    dt = 1 / 252
    sig = m["sigma"]
    r = math.log(np.prod(1 + m["rf"])) / (n * dt)
    z = rng.standard_normal((n_sims, n))
    lr = (r - 0.5 * sig ** 2) * dt + sig * math.sqrt(dt) * z
    return np.column_stack([np.ones(n_sims), np.exp(np.cumsum(lr, axis=1))])


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_hedges(months: list[dict], freqs=FREQS, cost: float = 0.0, n_sims: int = 20, seed: int = 1,
               iv_shift: float = 0.0) -> dict:
    """Hedged P&L (as a share of the premium) for every month, on the real path and on simulated BS paths."""
    rng = np.random.default_rng(seed)
    real = {f: [] for f in freqs}
    sim = {f: [] for f in freqs}
    rows = []
    for m in months:
        sig = max(m["sigma"] + iv_shift / 100, 0.01)
        rec = {"month": m["month"], "sigma": sig, "realized": m["realized"], "market": m["market"],
               "max_abs_day": m["max_abs_day"]}
        for f in freqs:
            h = delta_hedge(m["S"], 1.0, m["rf"], sig, "call", hedge_every=f, cost=cost)
            real[f].append(h["pnl"] / h["premium"])
            rec[f"pnl_{f}"] = h["pnl"] / h["premium"]
            if f == 1:
                rec["premium"] = h["premium"]
                rec["pnl_index_bp"] = 1e4 * h["pnl"]
        rec["gamma_approx"] = cash_gamma_pnl(m["S"], 1.0, m["rf"], sig) / rec["premium"]
        rows.append(rec)
        mm = dict(m, sigma=sig)
        for path in bs_paths(mm, n_sims, rng):
            for f in freqs:
                h = delta_hedge(path, 1.0, m["rf"], sig, "call", hedge_every=f, cost=cost)
                sim[f].append(h["pnl"] / h["premium"])
    return {"real": {f: np.array(v) for f, v in real.items()}, "sim": {f: np.array(v) for f, v in sim.items()},
            "monthly": pd.DataFrame(rows).set_index("month")}


def _dist(x: np.ndarray) -> dict:
    mo = moments(x)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "skewness": mo["skewness"],
            "excess_kurtosis": mo["kurtosis"] - 3 if mo["kurtosis"] > 2 else mo["kurtosis"],
            "q01": float(np.quantile(x, 0.01)), "worst": float(x.min()), "n": len(x)}


def part_a(res: dict) -> pd.DataFrame:
    rows = []
    for src, label in (("real", "Real market paths"), ("sim", "Black–Scholes paths (simulated)")):
        sd0 = res[src][0].std(ddof=1)
        for f in FREQS:
            dd = _dist(res[src][f])
            rows.append({"paths": label, "hedging": FREQ_NAMES[f], **dd,
                         "risk_removed": 1 - dd["sd"] / sd0 if f else 0.0})
    return pd.DataFrame(rows)


def part_b(res: dict) -> dict:
    m = res["monthly"]
    x, y = m["gamma_approx"].to_numpy(), m["pnl_1"].to_numpy()
    c = np.polyfit(x, y, 1)
    resid = y - np.polyval(c, x)
    r2 = 1 - resid.var() / y.var()
    vgap = (m["sigma"] ** 2 - m["realized"] ** 2).to_numpy()
    r2_gap = np.corrcoef(vgap, y)[0, 1] ** 2
    return {"slope": float(c[0]), "intercept": float(c[1]), "r2": float(r2), "r2_variance_gap": float(r2_gap),
            "share_months_realized_below_implied": float((m["realized"] < m["sigma"]).mean()),
            "corr_pnl_with_max_day": float(np.corrcoef(m["max_abs_day"], y)[0, 1])}


def part_c(months: list[dict], costs=(0.0, 0.0002, 0.0005, 0.001), n_sims: int = 5) -> pd.DataFrame:
    rows = []
    for k in costs:
        res = run_hedges(months, freqs=[1, 5], cost=k, n_sims=n_sims)
        for f in (1, 5):
            x = res["real"][f]
            rows.append({"cost_bp": 1e4 * k, "hedging": FREQ_NAMES[f], "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                         "q01": float(np.quantile(x, 0.01))})
    return pd.DataFrame(rows)


def part_d(res: dict, n: int = 8) -> dict:
    m = res["monthly"]
    worst = m.nsmallest(n, "pnl_1")[["pnl_1", "pnl_5", "pnl_0", "market", "sigma", "realized", "max_abs_day"]]
    worst.index = worst.index.astype(str)
    y = m["pnl_1"]
    losses = -y.clip(upper=0)
    k = max(1, int(round(0.05 * len(y))))
    share_tail = float(losses.nlargest(k).sum() / losses.sum()) if losses.sum() > 0 else float("nan")
    return {"worst": worst, "share_of_losses_in_worst_5pct": share_tail, "mean_pnl": float(y.mean()),
            "share_months_positive": float((y > 0).mean())}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_hist(res, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    r, s = res["real"][1], res["sim"][1]
    lo, hi = np.quantile(np.concatenate([r, s]), [0.002, 0.998])
    bins = np.linspace(min(lo, -1.5), max(hi, 1.0), 60)
    ax.hist(s, bins=bins, density=True, color=plots.INK_3, alpha=0.5, label="Black–Scholes paths (theory holds)")
    ax.hist(r, bins=bins, density=True, color=plots.BLUE, alpha=0.6, label="Real market paths")
    ax.axvline(0, color=plots.INK, linewidth=1)
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Hedged profit and loss at expiry, as a share of the option premium")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 18.1  Daily delta hedging: theory versus the market",
                 _sub(sim, f"Short one-month at-the-money calls · sd {r.std():.0%} of premium on real paths vs "
                           f"{s.std():.0%} on Black–Scholes paths · worst {r.min():.0%}"))
    plots.finish(fig, out / "fig18_1_hist.png")
    plt.close(fig)


def fig_monthly(res, d4, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    m = res["monthly"]
    dates = m.index.to_timestamp()
    y = m["pnl_1"]
    ax.bar(dates, y.clip(lower=0), width=25, color=plots.BLUE)
    ax.bar(dates, y.clip(upper=0), width=25, color=plots.RED)
    for mon in d4["worst"].index[:4]:
        p = pd.Period(mon, "M")
        ax.annotate(mon, (p.to_timestamp(), m.loc[p, "pnl_1"]), textcoords="offset points", xytext=(4, -2),
                    fontsize=8, color=plots.INK_2, va="top")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.set_ylabel("Hedged P&L, share of premium")
    plots.titles(fig, "Lab Figure 18.2  The hedged seller's month-by-month record",
                 _sub(sim, f"Positive in {d4['share_months_positive']:.0%} of months · the worst 5% of months produced "
                           f"{d4['share_of_losses_in_worst_5pct']:.0%} of all losses"))
    plots.finish(fig, out / "fig18_2_monthly.png")
    plt.close(fig)


def fig_gamma(res, b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    m = res["monthly"]
    ax.scatter(m["gamma_approx"], m["pnl_1"], s=12, alpha=0.6, color=plots.BLUE, edgecolors="none")
    lim = [min(m["gamma_approx"].min(), m["pnl_1"].min()), max(m["gamma_approx"].max(), m["pnl_1"].max())]
    ax.plot(lim, lim, color=plots.INK_3, linestyle="--", linewidth=1, label="45° line")
    plots.percent_axis(ax)
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Gamma–theta approximation: ½ Γ S² (σ²_implied Δt − r²), summed over the month")
    ax.set_ylabel("Actual hedged P&L")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 18.3  Hedging P&L is a bet on realized versus implied variance",
                 _sub(sim, f"R² = {b['r2']:.2f} · realized volatility below implied in "
                           f"{b['share_months_realized_below_implied']:.0%} of months"))
    plots.finish(fig, out / "fig18_3_gamma.png")
    plt.close(fig)


def fig_freq(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    for paths, color in (("Real market paths", plots.BLUE), ("Black–Scholes paths (simulated)", plots.INK_3)):
        g = a[a.paths == paths].set_index("hedging").reindex([FREQ_NAMES[f] for f in FREQS])
        ax.plot(range(len(g)), g["sd"], marker="o", color=color, linewidth=1.6, label=f"{paths}: standard deviation")
        ax.plot(range(len(g)), -g["q01"], marker="s", color=color, linewidth=1.0, linestyle="--",
                label=f"{paths}: 1% worst loss")
    ax.set_xticks(range(len(FREQS)), [FREQ_NAMES[f] for f in FREQS])
    plots.percent_axis(ax)
    ax.set_ylabel("Share of premium")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plots.titles(fig, "Lab Figure 18.4  How much risk is left at each hedging frequency?",
                 _sub(sim, "Risk left after hedging, by rebalancing frequency"))
    plots.finish(fig, out / "fig18_4_frequency.png")
    plt.close(fig)


def fig_costs(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    for h, color in (("Daily", plots.BLUE), ("Weekly", plots.AQUA)):
        g = c[c.hedging == h]
        ax.plot(g["cost_bp"], g["mean"], marker="o", color=color, linewidth=1.6, label=f"{h} hedging: mean P&L")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.set_xlabel("Transaction cost (basis points of the index traded)")
    ax.set_ylabel("Mean hedged P&L, share of premium")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    plots.titles(fig, "Lab Figure 18.5  Trading costs eat the hedger's margin",
                 _sub(sim, "Short one-month at-the-money calls, hedged on real paths"))
    plots.finish(fig, out / "fig18_5_costs.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


def write_report(out, res, a, b, c, d4, sim):
    m = res["monthly"]
    md = ["# Lab 18 results: Does Delta Hedging Remove Option Risk?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**Claim on trial:** an option seller who delta-hedges as Black–Scholes prescribes is left with little risk.", "",
           f"Months: {len(m)} ({m.index[0]} to {m.index[-1]}). Each month: sell a one-month at-the-money call priced at "
           f"the VIX (average premium {m['premium'].mean():.2%} of the index), hedge with Black–Scholes deltas at the same "
           "volatility. P&L is shown as a share of the premium.", "", "## Part A: How much risk is left?", "",
           _md(a.rename(columns={"paths": "Paths", "hedging": "Hedging", "mean": "Mean", "sd": "Std. dev.",
                                 "skewness": "Skewness", "excess_kurtosis": "Excess kurtosis", "q01": "1% quantile",
                                 "worst": "Worst", "n": "Months", "risk_removed": "Risk removed"}),
               {"Mean": _num("+.1%"), "Std. dev.": _num(".1%"), "Skewness": _num("+.2f"), "Excess kurtosis": _num(".1f"),
                "1% quantile": _num(".0%"), "Worst": _num(".0%"), "Risk removed": _num(".0%")}), "",
           "Risk removed = 1 − standard deviation hedged / standard deviation unhedged. Black–Scholes paths: 20 simulated "
           "paths per month with that month's implied volatility.", "", "![Lab Figure 18.1](fig18_1_hist.png)", "",
           "## Part B: Where the error comes from", "",
           f"- Regression of actual daily-hedged P&L on the gamma–theta approximation: slope {b['slope']:.2f}, R² "
           f"{b['r2']:.2f}.",
           f"- R² on the gap between implied and realized variance alone: {b['r2_variance_gap']:.2f}.",
           f"- Realized volatility was below implied in {b['share_months_realized_below_implied']:.0%} of months.",
           f"- Correlation of P&L with the month's largest absolute daily move: {b['corr_pnl_with_max_day']:+.2f}.", "",
           "![Lab Figure 18.3](fig18_3_gamma.png)", "", "## Part C: Hedging frequency and costs", "",
           _md(c.rename(columns={"cost_bp": "Cost (bp)", "hedging": "Hedging", "mean": "Mean", "sd": "Std. dev.",
                                 "q01": "1% quantile"}),
               {"Cost (bp)": _num(".0f"), "Mean": _num("+.1%"), "Std. dev.": _num(".1%"), "1% quantile": _num(".0%")}), "",
           "![Lab Figure 18.4](fig18_4_frequency.png)", "", "![Lab Figure 18.5](fig18_5_costs.png)", "",
           "## Part D: The worst months", "",
           _md(d4["worst"].reset_index().rename(columns={"month": "Month", "pnl_1": "Daily hedge", "pnl_5": "Weekly hedge",
                                                         "pnl_0": "Unhedged", "market": "Market", "sigma": "Implied vol",
                                                         "realized": "Realized vol", "max_abs_day": "Largest daily move"}),
               {"Daily hedge": _num(".0%"), "Weekly hedge": _num(".0%"), "Unhedged": _num(".0%"), "Market": _num("+.1%"),
                "Implied vol": _num(".0%"), "Realized vol": _num(".0%"), "Largest daily move": _num(".1%")}), "",
           f"Mean daily-hedged P&L {d4['mean_pnl']:+.1%} of premium; positive in {d4['share_months_positive']:.0%} of "
           f"months; the worst 5% of months account for {d4['share_of_losses_in_worst_5pct']:.0%} of all losses.", "",
           "![Lab Figure 18.2](fig18_2_monthly.png)", "", "## Your verdict", "",
           "1. How much of the unhedged risk does daily hedging remove on real paths, compared with Black–Scholes paths?",
           "2. What is left, and when does it show up?",
           "3. Does hedging more often, or more cheaply, change the answer?", ""]
    (out / "lab18_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--ff3-daily-file", default=None)
    ap.add_argument("--vix-file", default=None)
    ap.add_argument("--iv-shift", type=float, default=0.0, help="vol points added to the VIX when pricing and hedging")
    ap.add_argument("--n-sims", type=int, default=20, help="simulated Black-Scholes paths per month")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, {"ff3": args.ff3_daily_file, "vix": args.vix_file}, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    months = month_paths(d)
    res = run_hedges(months, n_sims=args.n_sims, iv_shift=args.iv_shift)
    a, b = part_a(res), part_b(res)
    c = part_c(months)
    d4 = part_d(res)
    fig_hist(res, out, sim)
    fig_monthly(res, d4, out, sim)
    fig_gamma(res, b, out, sim)
    fig_freq(a, out, sim)
    fig_costs(c, out, sim)
    a.to_csv(out / "table_hedging_error.csv", index=False)
    c.to_csv(out / "table_costs.csv", index=False)
    res["monthly"].to_csv(out / "table_monthly.csv")
    write_report(out, res, a, b, c, d4, sim)
    summary = {"simulated": sim, "hedging_error": a.to_dict(orient="records"), "decomposition": b,
               "costs": c.to_dict(orient="records"), "share_tail_losses": d4["share_of_losses_in_worst_5pct"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a[["paths", "hedging", "mean", "sd", "skewness", "q01", "risk_removed"]].round(3).to_string(index=False))
    print(f"gamma-theta R2 {b['r2']:.2f}; Report: {out / 'lab18_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
