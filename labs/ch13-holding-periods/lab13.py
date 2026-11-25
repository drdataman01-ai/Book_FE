"""Lab 13: When Do Investors Sell, and What Does It Cost Them?

Chapter 13 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "Investors earn the returns their stocks earn."
  More precisely: an investor who holds a stock earns, on average and per
  year invested, the return that fixed-period studies report for it,
  whatever rule the investor uses to decide when to sell.

WHAT THE LAB DOES
  Part A  Optional stopping. Six selling rules applied to the same simulated
          martingale price paths: equal expected returns, very different
          distributions (win rates, tails, losers still held at the cap).
  Part B  Comparing returns over random horizons. With a positive drift,
          each rule's return annualized position by position is compared
          with its log return per year invested (Wald's identity).
  Part C  Real prices. The same rules on the 49 value-weighted industry
          portfolios of the Kenneth R. French Data Library, entering every
          month: does each rule earn the industries' average return per
          year invested?
  Part D  The disposition test (in the spirit of Odean 1998). Do the
          winners a disposition investor sells go on to beat the losers
          the investor keeps?
  Part E  Taxes and costs. After-tax wealth of buy-and-hold, annual
          selling, selling winners and selling losers, in simulation.

DATA
  49 Industry Portfolios (monthly, value-weighted), Kenneth R. French Data Library.

USAGE
  python lab13.py
  python lab13.py --source sim        # fake industry returns, to test the code
  python lab13.py --up 0.3 --down 0.3 # different thresholds

OUTPUT
  report/lab13_report.md, five figures (PNG), CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, find_table, load_french_tables
from fintest.holding import (
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

ROOT = Path(__file__).resolve().parent
RULE_COLORS = [plots.INK_3, plots.INK_2, plots.BLUE, plots.ORANGE, plots.AQUA, plots.VIOLET]
TAX_KINDS = {"hold": "Buy and hold", "every": "Sell every 12 months",
             "winners": "Sell winners at +20% (disposition)", "losers": "Sell losers at -20% (harvest losses)"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def simulated_industries(n_months: int = 1188, n_assets: int = 49, seed: int = 0,
                         start: str = "1926-07") -> pd.DataFrame:
    """Fake monthly industry returns with a common market factor and no predictability. FOR TESTING CODE ONLY."""
    rng = np.random.default_rng(seed)
    t5 = lambda size: rng.standard_t(5, size) / math.sqrt(5 / 3)  # noqa: E731
    m = 0.045 * t5(n_months)
    beta = rng.uniform(0.7, 1.3, n_assets)
    r = 0.009 + m[:, None] * beta[None, :] + 0.045 * t5((n_months, n_assets))
    idx = pd.period_range(start=start, periods=n_months, freq="M", name="date")
    df = pd.DataFrame(np.maximum(r, -0.95), index=idx, columns=[f"Ind{j + 1:02d}" for j in range(n_assets)])
    df.attrs["simulated"] = True
    return df


def load_industries(source: str, path=None, start=None, end=None, sim_seed: int = 0) -> pd.DataFrame:
    if source == "sim":
        df = simulated_industries(seed=sim_seed)
    else:
        tables = load_french_tables("industry_49", path=path)
        df = find_table(tables, "value weighted", "monthly").copy()
        df.columns = [c.strip() for c in df.columns]
        df.attrs["simulated"] = False
    sim = df.attrs["simulated"]
    if start:
        df = df.loc[pd.Period(start, "M"):]
    if end:
        df = df.loc[:pd.Period(end, "M")]
    df.attrs["simulated"] = sim
    return df


# ---------------------------------------------------------------------------
# Parts A and B: simulation
# ---------------------------------------------------------------------------


def run_rules(rel: np.ndarray, rules: list[Rule]) -> pd.DataFrame:
    rows = []
    for r in rules:
        tau, hpr, why = exit_times(rel, r)
        rows.append({"rule": r.name, **summarize(hpr, tau, why)})
    return pd.DataFrame(rows)


def part_a(n_paths: int, sigma: float, cap: int, up: float, down: float, seed: int) -> pd.DataFrame:
    """Martingale prices (zero expected return): every rule should have a zero mean holding-period return."""
    rel = simulate_paths(n_paths, cap, mu=0.0, sigma=sigma, seed=seed)
    return run_rules(rel, standard_rules(up, down, cap))


def part_b(n_paths: int, mu: float, sigma: float, cap: int, up: float, down: float, seed: int) -> pd.DataFrame:
    """Positive drift: naive annualized returns against log return per year invested."""
    rel = simulate_paths(n_paths, cap, mu=mu, sigma=sigma, seed=seed + 1)
    t = run_rules(rel, standard_rules(up, down, cap))
    t["wald_benchmark"] = mu - 0.5 * sigma**2
    return t


# ---------------------------------------------------------------------------
# Parts C and D: real prices
# ---------------------------------------------------------------------------


def part_c(df: pd.DataFrame, cap: int, up: float, down: float, n_boot: int, seed: int) -> dict:
    pos = entries_from_panel(df, cap)
    g = mean_log_return(df)
    rows = []
    for r in standard_rules(up, down, cap):
        tau, hpr, why = exit_times(pos["rel"], r)
        ex = excess_per_year(pos, tau, hpr, g, block=cap, n_boot=n_boot, seed=seed)
        s = summarize(hpr, tau, why)
        rows.append({"rule": r.name, "positions": s["n"], "mean_months": s["mean_months"], "win_rate": s["win_rate"],
                     "median_hpr": s["median_hpr"], "share_losers_at_cap": s["share_losers_at_cap"],
                     "per_year_log": s["per_year_log"], **ex})
    return {"table": pd.DataFrame(rows), "pos": pos, "avg_log_per_year": float(12 * np.nanmean(g))}


def part_d(pos: dict, cap: int, up: float, horizon: int = 12) -> dict:
    rule = Rule(f"Profit target +{up:.0%}", "target", up=up, cap=cap)
    return disposition_test(pos, rule, horizon=horizon)


def by_decade(d: dict, index: pd.PeriodIndex) -> pd.DataFrame:
    """Average sold-winner minus held-loser gap by decade. Month i of d['month_index'] is the end of month i."""
    dates = [index[i - 1] for i in d["month_index"]]
    decade = [(p.year // 10) * 10 for p in dates]
    s = pd.Series(d["diff_series"], index=decade)
    out = s.groupby(level=0).agg(["mean", "count"]).reset_index()
    out.columns = ["decade", "diff_mean", "months"]
    return out


# ---------------------------------------------------------------------------
# Part E: taxes and costs
# ---------------------------------------------------------------------------


def part_e(years: int, n: int, mu: float, sigma: float, tax: float, cost: float, up: float, down: float,
           seed: int) -> pd.DataFrame:
    rows = []
    base = None
    for kind, label in TAX_KINDS.items():
        after = simulate_taxes(kind, years, n, mu, sigma, tax, cost, up, down, seed=seed)
        pre = simulate_taxes(kind, years, n, mu, sigma, 0.0, cost, up, down, seed=seed)
        if base is None:
            base = pre["mean_wealth"]
        rows.append({"rule": label, "kind": kind, "sales_per_year": after["sales_per_year"],
                     "pre_tax_wealth": pre["mean_wealth"], "after_tax_wealth": after["mean_wealth"],
                     "after_tax_median": after["median_wealth"],
                     "pre_tax_annual": pre["annualized_mean_wealth"], "after_tax_annual": after["annualized_mean_wealth"],
                     "after_tax_vs_hold_pre": after["mean_wealth"] / base})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim: bool, text: str) -> str:
    return text + ("  ·  SIMULATED INDUSTRY DATA: tests the code only" if sim else "")


def _short(name: str) -> str:
    return name.replace(" (+20%; after -20%, sell at cost)", "").replace("Get even", "Get even")


def fig_distributions(a: pd.DataFrame, out: Path, sigma: float, cap: int):
    fig, ax, plt = plots.new_figure(9.5, 5.2)
    y = np.arange(len(a))[::-1]
    for yi, (_, r), col in zip(y, a.iterrows(), RULE_COLORS):
        ax.plot([r.q05, r.q95], [yi, yi], color=col, linewidth=1.2)
        ax.plot([r.q25, r.q75], [yi, yi], color=col, linewidth=7, solid_capstyle="butt", alpha=0.85)
        ax.plot(r.median_hpr, yi, "|", color=plots.SURFACE, markersize=11, markeredgewidth=2)
        ax.plot(r.mean_hpr, yi, "D", color=plots.INK, markersize=5.5)
        ax.text(max(a["q95"]) * 1.04, yi, f"wins {r.win_rate:.0%} · held {r.mean_months:.0f} mo · "
                f"losers at cap {r.share_losers_at_cap:.0%}", va="center", fontsize=8.3, color=plots.INK_2)
    ax.axvline(0, color=plots.INK_3, linewidth=1)
    ax.set_yticks(y, [_short(n) for n in a["rule"]], fontsize=9, color=plots.INK)
    plots.percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=plots.GRID, linewidth=0.8)
    ax.set_xlim(min(a["q05"]) * 1.1, max(a["q95"]) * 1.75)
    ax.set_xlabel("Holding-period return at the sale")
    plots.titles(fig, "Lab Figure 13.1  Same prices, same expected return, different experiences",
                 f"Simulated martingale prices (expected return zero), volatility {sigma:.0%}, longest hold {cap} months · "
                 "thin line: 5th-95th percentile · thick: 25th-75th · white tick: median · diamond: mean")
    plots.finish(fig, out / "fig13_1_distributions.png")
    plt.close(fig)


def fig_per_year(b: pd.DataFrame, out: Path, mu: float, sigma: float):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    x = np.arange(len(b))
    ax.bar(x - 0.2, b["naive_annual"], width=0.38, color=plots.ORANGE, label="Average of each position's annualized return")
    ax.bar(x + 0.2, b["per_year_log"], width=0.38, color=plots.BLUE, label="Log return per year invested")
    ax.axhline(mu - 0.5 * sigma**2, color=plots.BLUE, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_xticks(x, [_short(n).replace(" / ", "/\n").replace("Profit target", "Target").replace(" months", " mo")
                      for n in b["rule"]], fontsize=8.5)
    plots.percent_axis(ax, "y")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_ylabel("Per year")
    plots.titles(fig, "Lab Figure 13.2  Annualizing each sale on its own exaggerates quick wins",
                 f"Simulated prices, expected return {mu:.0%} a year, volatility {sigma:.0%} · dashed: "
                 f"mu - sigma^2/2 = {mu - 0.5 * sigma**2:.1%}, the log growth every rule earns per year invested")
    plots.finish(fig, out / "fig13_2_per_year.png")
    plt.close(fig)


def fig_real(c: pd.DataFrame, out: Path, sim: bool, period: str, cap: int):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    y = np.arange(len(c))[::-1]
    for yi, (_, r), col in zip(y, c.iterrows(), RULE_COLORS):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, linewidth=2.2)
        ax.plot(r.excess_per_year, yi, "o", color=col, markersize=7, markeredgecolor=plots.SURFACE)
        ax.text(r.ci_high, yi, f"  {r.excess_per_year:+.2%} (t = {r.t:.1f})", va="center", fontsize=8.5, color=col)
    ax.axvline(0, color=plots.INK_3, linewidth=1)
    ax.set_yticks(y, [_short(n) for n in c["rule"]], fontsize=9, color=plots.INK)
    plots.percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=plots.GRID, linewidth=0.8)
    lo, hi = c["ci_low"].min(), c["ci_high"].max()
    ax.set_xlim(min(lo, -0.005) * 1.3, max(hi, 0.005) * 1.9)
    ax.set_xlabel("Log return per year invested, minus the industries' average log return per year")
    plots.titles(fig, "Lab Figure 13.3  Do selling rules earn the assets' return on real prices?",
                 _sub(sim, f"49 value-weighted industry portfolios, {period}, a position bought every month in every "
                           f"industry, longest hold {cap} months · lines: 95% block-bootstrap intervals"))
    plots.finish(fig, out / "fig13_3_real_prices.png")
    plt.close(fig)


def fig_disposition(d: dict, dec: pd.DataFrame, out: Path, sim: bool, period: str, up: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    fig.patch.set_facecolor(plots.SURFACE)
    for ax in axes:
        plots.style_axes(ax)
    ax = axes[0]
    vals = [d["sold_winners_mean"], d["held_losers_mean"], d["ew_mean"]]
    labels = ["Winners sold\n(at the target)", "Losers held\n(below cost)", "All industries\n(equal weight)"]
    cols = [plots.BLUE, plots.ORANGE, plots.INK_3]
    ax.bar(np.arange(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=plots.INK)
    ax.set_xticks(np.arange(3), labels, fontsize=8.5)
    plots.percent_axis(ax, "y")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_title(f"Next {d['horizon']}-month log return", loc="left", fontsize=9.5, color=plots.INK)
    ax = axes[1]
    ax.bar(dec["decade"].astype(str) + "s", dec["diff_mean"], color=np.where(dec["diff_mean"] >= 0, plots.BLUE,
                                                                             plots.ORANGE), width=0.65)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax, "y")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title(f"Winners sold minus losers held, by decade (full sample {d['diff_mean']:+.1%}, "
                 f"t = {d['diff_t']:.1f})", loc="left", fontsize=9.5, color=plots.INK)
    plots.titles(fig, "Lab Figure 13.4  The disposition test: what the investor sold and what the investor kept",
                 _sub(sim, f"49 industry portfolios, {period} · rule: sell at +{up:.0%}, never sell below cost before the "
                           "cap · Newey-West t-statistic for overlapping horizons"))
    plots.finish(fig, out / "fig13_4_disposition.png")
    plt.close(fig)


def fig_taxes(e: pd.DataFrame, out: Path, tax: float, cost: float, mu: float, sigma: float, years: int):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    x = np.arange(len(e))
    ax.bar(x - 0.2, e["pre_tax_annual"], width=0.38, color=plots.INK_3, label="Before tax")
    ax.bar(x + 0.2, e["after_tax_annual"], width=0.38, color=plots.BLUE, label=f"After a {tax:.0%} tax on realized gains")
    for xi, v in zip(x, e["after_tax_annual"]):
        ax.text(xi + 0.2, v, f"{v:.2%}", ha="center", va="bottom", fontsize=8.5, color=plots.INK)
    ax.set_xticks(x, [r.replace(" (", "\n(") for r in e["rule"]], fontsize=8.5)
    plots.percent_axis(ax, "y")
    ax.set_ylim(min(e["after_tax_annual"].min(), e["pre_tax_annual"].min()) * 0.8,
                max(e["pre_tax_annual"].max(), e["after_tax_annual"].max()) * 1.08)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_ylabel("Growth of expected wealth, per year")
    plots.titles(fig, "Lab Figure 13.5  Selling winners pays the tax early; selling losers defers it",
                 f"Simulation: expected return {mu:.0%}, volatility {sigma:.0%}, {years} years, one-way cost "
                 f"{cost * 1e4:.0f} bp · losses earn a tax credit at the same rate · everything is sold at the end")
    plots.finish(fig, out / "fig13_5_taxes.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


def write_report(out, period, a, b, c, d, dec, e, args, sim):
    md = ["# Lab 13 results: When Do Investors Sell, and What Does It Cost Them?", ""]
    if sim:
        md += ["> **Simulated industry data.** Parts C and D test that the code runs; they say nothing about real "
               "markets. Parts A, B and E use simulated prices by design.", ""]
    md += ["**Claim on trial:** investors earn the returns their stocks earn, whatever rule they use to decide "
           "when to sell.", "",
           f"Thresholds: profit target +{args.up:.0%}, stop-loss -{args.down:.0%}. Longest hold {args.sim_cap} months "
           f"in Parts A and B and {args.cap} months in Parts C and D.", ""]
    pct, pct1 = _num("+.2%"), _num(".0%")
    md += ["## Part A: Optional stopping (simulated martingale prices)", "",
           _md(a[["rule", "mean_hpr", "se_mean_hpr", "median_hpr", "win_rate", "mean_gain_if_win", "mean_loss_if_loss",
                  "skewness", "mean_months", "share_losers_at_cap"]].rename(columns={
               "rule": "Rule", "mean_hpr": "Mean return", "se_mean_hpr": "Std. error", "median_hpr": "Median",
               "win_rate": "Win rate", "mean_gain_if_win": "Average win", "mean_loss_if_loss": "Average loss",
               "skewness": "Skewness", "mean_months": "Months held", "share_losers_at_cap": "Losers at cap"}),
               {"Mean return": pct, "Std. error": _num(".2%"), "Median": pct, "Win rate": pct1, "Average win": pct,
                "Average loss": pct, "Skewness": _num("+.2f"), "Months held": _num(".1f"), "Losers at cap": pct1}), "",
           f"{args.paths:,} paths, volatility {args.sigma:.0%}, expected return zero. Every mean should be zero within "
           "about two standard errors: no selling rule can change the expected value of a fair game.", "",
           "![Lab Figure 13.1](fig13_1_distributions.png)", ""]
    md += ["## Part B: Comparing returns over random horizons", "",
           _md(b[["rule", "mean_months", "mean_hpr", "naive_annual", "per_year_log"]].rename(columns={
               "rule": "Rule", "mean_months": "Months held", "mean_hpr": "Mean holding-period return",
               "naive_annual": "Average annualized", "per_year_log": "Log return per year invested"}),
               {"Months held": _num(".1f"), "Mean holding-period return": pct, "Average annualized": pct,
                "Log return per year invested": pct}), "",
           f"Expected return {args.mu:.0%}, volatility {args.sigma:.0%}. Wald's identity says every rule's log return "
           f"per year invested should be close to mu - sigma^2/2 = {args.mu - 0.5 * args.sigma ** 2:.2%}.", "",
           "![Lab Figure 13.2](fig13_2_per_year.png)", ""]
    t = c["table"]
    md += ["## Part C: Real prices (49 industry portfolios)", "",
           f"Sample {period}. Average log return of the industries: {c['avg_log_per_year']:.2%} a year.", "",
           _md(t[["rule", "positions", "mean_months", "win_rate", "median_hpr", "share_losers_at_cap", "per_year_log",
                  "excess_per_year", "t", "ci_low", "ci_high"]].rename(columns={
               "rule": "Rule", "positions": "Positions", "mean_months": "Months held", "win_rate": "Win rate",
               "median_hpr": "Median return", "share_losers_at_cap": "Losers at cap",
               "per_year_log": "Log return per year", "excess_per_year": "Minus industry average", "t": "t",
               "ci_low": "95% low", "ci_high": "95% high"}),
               {"Positions": lambda v: f"{int(v):,}", "Months held": _num(".1f"), "Win rate": pct1, "Median return": pct,
                "Losers at cap": pct1, "Log return per year": pct, "Minus industry average": pct, "t": _num(".2f"),
                "95% low": pct, "95% high": pct}), "",
           "Minus industry average: the rule's log return per year invested less the industries' own average log "
           "return per year. Intervals from a moving-block bootstrap over entry months.", "",
           "![Lab Figure 13.3](fig13_3_real_prices.png)", ""]
    md += ["## Part D: The disposition test", "",
           f"Rule: sell when the gain reaches +{args.up:.0%}; hold losers until the {args.cap}-month cap. "
           f"{d['n_sold_winners']:,} sales of winners; {d['n_held_loser_months']:,} position-months of held losers; "
           f"{d['months']:,} months with both.", "",
           f"| Group | Next {d['horizon']}-month log return |", "|---|---|",
           f"| Winners sold | {d['sold_winners_mean']:+.2%} |", f"| Losers held | {d['held_losers_mean']:+.2%} |",
           f"| All industries, equal weight | {d['ew_mean']:+.2%} |",
           f"| Sold winners minus held losers | {d['diff_mean']:+.2%} (Newey-West t = {d['diff_t']:.2f}) |", "",
           "**By decade**", "",
           _md(dec, {"decade": lambda v: f"{int(v)}s", "diff_mean": pct, "months": lambda v: f"{int(v)}"}), "",
           "![Lab Figure 13.4](fig13_4_disposition.png)", ""]
    md += ["## Part E: Taxes and costs (simulation)", "",
           _md(e.drop(columns=["kind"]).rename(columns={
               "rule": "Rule", "sales_per_year": "Sales per year", "pre_tax_wealth": "Wealth before tax",
               "after_tax_wealth": "Wealth after tax", "after_tax_median": "Median after tax",
               "pre_tax_annual": "Growth before tax", "after_tax_annual": "Growth after tax",
               "after_tax_vs_hold_pre": "After tax, share of buy-and-hold before tax"}),
               {"Sales per year": _num(".2f"), "Wealth before tax": _num(".2f"), "Wealth after tax": _num(".2f"),
                "Median after tax": _num(".2f"), "Growth before tax": _num(".2%"), "Growth after tax": _num(".2%"),
                "After tax, share of buy-and-hold before tax": _num(".0%")}), "",
           f"{args.tax_paths:,} investors over {args.years} years, expected return {args.mu:.0%}, volatility "
           f"{args.sigma:.0%}, tax {args.tax:.0%} on realized gains (losses earn a credit), one-way cost "
           f"{args.cost * 1e4:.0f} bp. Wealth is the average final wealth per $1 invested; all rules see the same "
           "price shocks.", "", "![Lab Figure 13.5](fig13_5_taxes.png)", ""]
    md += ["## Your verdict", "",
           "1. Part A: which rule would an investor most enjoy following, judged by its win rate? Which rule leaves the "
           "most losers still held at the cap? Did any rule change the expected return?",
           "2. Part B: why does the average annualized return rank the rules so differently from the log return per "
           "year invested? Which measure would you use to compare two investors with different holding periods?",
           "3. Part C: on real prices, does any rule earn significantly more or less than the industries' average? "
           "If so, what property of real returns, absent from Part A, could explain it?",
           "4. Part D: do the winners sold beat the losers held? How does your answer compare with Odean (1998), who "
           "used individual stocks rather than industries?",
           "5. Part E: rank the rules after tax. Which part of the claim, if any, survives taxes?", ""]
    (out / "lab13_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--industry-file", default=None, help="49 Industry Portfolios (monthly) zip or CSV")
    ap.add_argument("--start", default=None, help="first month, YYYY-MM (default: start of the file)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--up", type=float, default=0.2, help="profit target (decimal)")
    ap.add_argument("--down", type=float, default=0.2, help="stop-loss (decimal, positive)")
    ap.add_argument("--cap", type=int, default=60, help="longest hold in Parts C and D, months")
    ap.add_argument("--sim-cap", type=int, default=120, help="longest hold in Parts A and B, months")
    ap.add_argument("--paths", type=int, default=50000, help="simulated paths in Parts A and B")
    ap.add_argument("--sigma", type=float, default=0.30, help="annual volatility in the simulations")
    ap.add_argument("--mu", type=float, default=0.08, help="annual expected return in Parts B and E")
    ap.add_argument("--tax", type=float, default=0.20, help="tax rate on realized gains in Part E")
    ap.add_argument("--cost", type=float, default=0.001, help="one-way trading cost in Part E (0.001 = 10 bp)")
    ap.add_argument("--years", type=int, default=30, help="investment horizon in Part E")
    ap.add_argument("--tax-paths", type=int, default=20000, help="simulated investors in Part E")
    ap.add_argument("--boot", type=int, default=400, help="bootstrap replications in Part C")
    ap.add_argument("--seed", type=int, default=2026, help="seed for the simulations")
    ap.add_argument("--sim-seed", type=int, default=0, help="seed for simulated industry data (--source sim)")
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        df = load_industries(args.source, args.industry_file, args.start, args.end, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = df.attrs.get("simulated", False)
    period = f"{df.index[0]} to {df.index[-1]}"

    a = part_a(args.paths, args.sigma, args.sim_cap, args.up, args.down, args.seed)
    b = part_b(args.paths, args.mu, args.sigma, args.sim_cap, args.up, args.down, args.seed)
    c = part_c(df, args.cap, args.up, args.down, args.boot, args.seed)
    d = part_d(c["pos"], args.cap, args.up)
    dec = by_decade(d, df.index)
    e = part_e(args.years, args.tax_paths, args.mu, args.sigma, args.tax, args.cost, args.up, args.down, args.seed)

    fig_distributions(a, out, args.sigma, args.sim_cap)
    fig_per_year(b, out, args.mu, args.sigma)
    fig_real(c["table"], out, sim, period, args.cap)
    fig_disposition(d, dec, out, sim, period, args.up)
    fig_taxes(e, out, args.tax, args.cost, args.mu, args.sigma, args.years)

    a.to_csv(out / "table_optional_stopping.csv", index=False)
    b.to_csv(out / "table_per_year.csv", index=False)
    c["table"].to_csv(out / "table_real_prices.csv", index=False)
    dec.to_csv(out / "table_disposition_by_decade.csv", index=False)
    e.to_csv(out / "table_taxes.csv", index=False)
    write_report(out, period, a, b, c, d, dec, e, args, sim)

    summary = {
        "simulated": sim, "period": period, "up": args.up, "down": args.down, "cap": args.cap,
        "optional_stopping_mean_hpr": dict(zip(a["rule"], a["mean_hpr"])),
        "optional_stopping_win_rate": dict(zip(a["rule"], a["win_rate"])),
        "per_year_log": dict(zip(b["rule"], b["per_year_log"])),
        "naive_annual": dict(zip(b["rule"], b["naive_annual"])),
        "real_excess_per_year": dict(zip(c["table"]["rule"], c["table"]["excess_per_year"])),
        "real_excess_t": dict(zip(c["table"]["rule"], c["table"]["t"])),
        "disposition": {k: d[k] for k in ("sold_winners_mean", "held_losers_mean", "ew_mean", "diff_mean", "diff_t",
                                          "months", "n_sold_winners")},
        "taxes_after_tax_annual": dict(zip(e["rule"], e["after_tax_annual"])),
        "taxes_pre_tax_annual": dict(zip(e["rule"], e["pre_tax_annual"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    for _, r in a.iterrows():
        print(f"A  {r['rule']:45s} mean {r['mean_hpr']:+.2%}  win {r['win_rate']:.0%}  "
              f"losers at cap {r['share_losers_at_cap']:.0%}")
    for _, r in c["table"].iterrows():
        print(f"C  {r['rule']:45s} excess {r['excess_per_year']:+.2%}/yr  t {r['t']:+.2f}")
    print(f"D  sold winners {d['sold_winners_mean']:+.2%}  held losers {d['held_losers_mean']:+.2%}  "
          f"gap {d['diff_mean']:+.2%} (t {d['diff_t']:.2f})")
    for _, r in e.iterrows():
        print(f"E  {r['rule']:45s} after tax {r['after_tax_annual']:.2%}/yr  before {r['pre_tax_annual']:.2%}/yr")
    print(f"Report: {out / 'lab13_report.md'}")
    if sim:
        print("NOTE: simulated industry data for Parts C and D. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
