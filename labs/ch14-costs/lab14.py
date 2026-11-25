"""Lab 14: What Survives the Trip from Paper to Portfolio?

Chapter 14 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "A strategy that is profitable in a backtest is profitable to trade."

WHAT THE LAB DOES
  Part A  Gross performance of three classic long-short factors from the
          Kenneth R. French Data Library: value (HML), momentum (Mom) and
          short-term reversal (ST Rev).
  Part B  Turnover. Published estimates (Novy-Marx and Velikov 2016) beside
          turnover measured mechanically in a simulated stock universe
          whose portfolios are built the Fama-French way.
  Part C  Costs. Net returns across a range of one-way trading costs and a
          short-borrowing fee; break-even costs; rankings before and after
          costs.
  Part D  Capacity. Square-root market impact: how net returns shrink as
          assets under management grow.
  Part E  Mitigation, with known truth. Buy/hold spreads (banding) in the
          simulated universe: how much turnover they save and how much
          signal they give up.

DATA
  Fama/French 3 Factors, Momentum Factor (Mom) and Short-Term Reversal
  Factor (ST Rev), monthly, from the Kenneth R. French Data Library.

USAGE
  python lab14.py
  python lab14.py --source sim             # fake factor data, to test the code
  python lab14.py --cost 0.002             # one-way cost for the headline tables

OUTPUT
  report/lab14_report.md, five figures (PNG), CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.costs import (
    backtest_sort,
    break_even_cost,
    capacity_curve,
    monthly_cost,
    net_returns,
    simulate_universe,
    sort_weights,
)
from fintest.data import DataUnavailable, load_french_factors

ROOT = Path(__file__).resolve().parent
STRATEGIES = ["value", "momentum", "reversal"]
LABELS = {"value": "Value (HML)", "momentum": "Momentum (Mom)", "reversal": "Short-term reversal (ST Rev)"}
COLORS = {"value": plots.BLUE, "momentum": plots.ORANGE, "reversal": plots.VIOLET}

# One-sided monthly turnover, value-weighted portfolios, average of long and short sides:
# Novy-Marx and Velikov (2016), Table 3 (NBER working paper w20721 version).
PUBLISHED_TURNOVER = {"value": 0.0291, "momentum": 0.3452, "reversal": 0.9087}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _first_col(df: pd.DataFrame, *names) -> pd.Series:
    cols = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return df[cols[n.lower()]]
    return df.iloc[:, 0]


def simulated_factors(n_months: int = 1188, seed: int = 0, start: str = "1926-07") -> pd.DataFrame:
    """Fake monthly factor returns with plausible means and volatilities. FOR TESTING CODE ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start=start, periods=n_months, freq="M", name="date")
    t5 = lambda: rng.standard_t(5, n_months) / math.sqrt(5 / 3)  # noqa: E731
    df = pd.DataFrame({"value": 0.0035 + 0.032 * t5(), "momentum": 0.0065 + 0.045 * t5(),
                       "reversal": 0.0045 + 0.034 * t5(), "rf": 0.0028}, index=idx)
    df.attrs["simulated"] = True
    return df


def load_factors(source: str, ff3_file=None, mom_file=None, rev_file=None, start="1963-07", end=None,
                 sim_seed: int = 0) -> pd.DataFrame:
    if source == "sim":
        df = simulated_factors(seed=sim_seed)
    else:
        ff3 = load_french_factors("ff3_monthly", path=ff3_file)
        mom = load_french_factors("momentum_monthly", path=mom_file)
        rev = load_french_factors("st_reversal_monthly", path=rev_file)
        df = pd.DataFrame({"value": _first_col(ff3, "HML"), "momentum": _first_col(mom, "Mom", "UMD"),
                           "reversal": _first_col(rev, "ST_Rev", "ST_REV"), "rf": _first_col(ff3, "RF")}).dropna()
        df.attrs["simulated"] = False
    sim = df.attrs["simulated"]
    if start:
        df = df.loc[pd.Period(start, "M"):]
    if end:
        df = df.loc[:pd.Period(end, "M")]
    df.attrs["simulated"] = sim
    return df


# ---------------------------------------------------------------------------
# Part A: gross performance
# ---------------------------------------------------------------------------


def _stats(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    m, s = r.mean(), r.std(ddof=1)
    return {"mean": m, "t": m / (s / math.sqrt(len(r))), "sharpe": m / s * math.sqrt(12), "vol": s * math.sqrt(12)}


def part_a(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in STRATEGIES:
        s = _stats(df[k].to_numpy())
        rows.append({"strategy": LABELS[k], "key": k, "mean_monthly": s["mean"], "t_stat": s["t"],
                     "sharpe": s["sharpe"], "vol_annual": s["vol"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: turnover
# ---------------------------------------------------------------------------


def part_b(universe: dict) -> pd.DataFrame:
    rows = []
    for k in STRATEGIES:
        bt = backtest_sort(universe, k)
        rows.append({"strategy": LABELS[k], "key": k, "published": PUBLISHED_TURNOVER[k],
                     "simulated": float(np.nanmean(bt["turnover"])),
                     "simulated_gross": float(bt["gross"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part C: costs
# ---------------------------------------------------------------------------


def part_c(df: pd.DataFrame, turnover: dict, cost: float, borrow_fee: float,
           grid=np.linspace(0, 0.01, 41)) -> dict:
    rows, curves = [], {}
    for k in STRATEGIES:
        g = df[k].to_numpy()
        T = turnover[k]
        net = net_returns(g, T, cost, borrow_fee)
        sg, sn = _stats(g), _stats(net)
        rows.append({"strategy": LABELS[k], "key": k, "turnover": T, "gross_mean": sg["mean"],
                     "cost_per_month": monthly_cost(T, cost) + borrow_fee / 12, "net_mean": sn["mean"],
                     "net_t": sn["t"], "gross_sharpe": sg["sharpe"], "net_sharpe": sn["sharpe"],
                     "break_even_one_way": break_even_cost(sg["mean"], T, other_costs=borrow_fee / 12)})
        curves[k] = np.array([sg["mean"] - monthly_cost(T, c) - borrow_fee / 12 for c in grid])
    t = pd.DataFrame(rows)
    t["rank_gross"] = t["gross_sharpe"].rank(ascending=False).astype(int)
    t["rank_net"] = t["net_sharpe"].rank(ascending=False).astype(int)
    return {"table": t, "grid": grid, "curves": curves}


def cost_sensitivity(df: pd.DataFrame, turnover: dict, borrow_fee: float, costs=(0.0, 0.001, 0.002, 0.003, 0.005,
                                                                              0.0075, 0.01)) -> pd.DataFrame:
    rows = []
    for c in costs:
        row = {"one_way_cost": c}
        for k in STRATEGIES:
            row[LABELS[k]] = _stats(net_returns(df[k].to_numpy(), turnover[k], c, borrow_fee))["sharpe"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part D: capacity
# ---------------------------------------------------------------------------


def market_microstructure(universe: dict, total_cap: float = 50e12, daily_turnover: float = 0.004,
                          seed: int = 0) -> dict:
    """Assign dollar market caps, daily volume, daily volatility and half-spreads to the simulated stocks.

    Assumptions (all adjustable): market caps scaled so the universe totals
    `total_cap`; average daily dollar volume = daily_turnover x market cap
    (0.4% a day is roughly 100% a year); daily volatility from each stock's
    simulated returns; half-spreads from 2 basis points for the largest
    stocks to about 50 for the smallest, falling with log size.
    """
    me = universe["me"][-1]
    cap = me / me.sum() * total_cap
    adv = daily_turnover * cap
    vol = universe["returns"].std(axis=0) / math.sqrt(21)
    lr = np.log(cap)
    pct = (lr - lr.min()) / (lr.max() - lr.min())
    half_spread = 0.0002 + 0.0048 * (1 - pct) ** 2
    return {"cap": cap, "adv": adv, "daily_vol": vol, "half_spread": half_spread}


def part_d(universe: dict, gross: dict, turnover: dict, micro: dict, y: float = 1.0, days: int = 5,
           aum_grid=np.logspace(7, 12, 41)) -> dict:
    """Net return against AUM for each strategy, using its value-weighted holdings in the simulated universe."""
    from fintest.costs import signals

    sig = signals(universe)
    t = universe["returns"].shape[0] - 2
    out = {}
    for k in STRATEGIES:
        top, bot = sort_weights(sig[k][t], micro["cap"])
        wl = np.where(top, micro["cap"], 0) / np.where(top, micro["cap"], 0).sum()
        ws = np.where(bot, micro["cap"], 0) / np.where(bot, micro["cap"], 0).sum()
        w = 0.5 * (wl + ws)  # trades spread over both sides' holdings
        out[k] = capacity_curve(aum_grid, gross[k], turnover[k], w, micro["adv"], micro["daily_vol"],
                                micro["half_spread"], y, days)
    return out


# ---------------------------------------------------------------------------
# Part E: mitigation with known truth
# ---------------------------------------------------------------------------


def part_e(universe: dict, cost: float, high_cost: float = 0.005, hold_fracs=(None, 0.4, 0.5, 0.6)) -> pd.DataFrame:
    """Turnover and true expected returns of standard and banded sorts, net at two cost levels."""
    rows = []
    for k in STRATEGIES:
        for h in hold_fracs:
            bt = backtest_sort(universe, k, hold_frac=h)
            T = float(np.nanmean(bt["turnover"]))
            g = float(bt["expected"].mean())
            rows.append({"strategy": LABELS[k], "key": k, "rule": "Standard 30/70 sort" if h is None else
                         f"Buy top 30%, hold while in top {h:.0%}", "hold_frac": h if h else 0.3,
                         "turnover": T, "gross_mean": g, "net_mean": g - monthly_cost(T, cost),
                         "net_mean_high": g - monthly_cost(T, high_cost)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim: bool, text: str) -> str:
    return text + ("  ·  SIMULATED FACTOR DATA: tests the code only" if sim else "")


def fig_net_vs_cost(c, out, sim, cost, period):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    grid = c["grid"] * 1e4
    t = c["table"].set_index("key")
    ymin = min(cv.min() for cv in c["curves"].values())
    for k in STRATEGIES:
        ax.plot(grid, c["curves"][k] * 100, color=COLORS[k], linewidth=2)
        be = t.loc[k, "break_even_one_way"] * 1e4
        if be <= grid.max():
            ax.plot(be, 0, "o", color=COLORS[k], markersize=7, markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{be:.0f} bp", (be, 0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5,
                        color=COLORS[k])
        yend = c["curves"][k][-1] * 100
        ax.text(grid[-1] + 1.5, max(yend, ymin * 100), LABELS[k], va="center", fontsize=9, color=COLORS[k])
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.axvline(cost * 1e4, color=plots.INK_3, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(cost * 1e4, ax.get_ylim()[1], f" headline cost {cost * 1e4:.0f} bp", fontsize=8.5, color=plots.INK_2,
            va="top")
    ax.set_xlim(0, grid.max() * 1.45)
    ax.set_xlabel("One-way trading cost (basis points per dollar traded)")
    ax.set_ylabel("Average net return (% per month)")
    plots.titles(fig, "Lab Figure 14.1  How much trading cost can each strategy pay?",
                 _sub(sim, f"{period} · published turnover (Novy-Marx and Velikov 2016) · dots: break-even cost, "
                           "where the average net return reaches zero"))
    plots.finish(fig, out / "fig14_1_net_vs_cost.png")
    plt.close(fig)


def fig_cumulative(df, turnover, cost, borrow_fee, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    dates = df.index.to_timestamp()
    ends = []
    for k in STRATEGIES:
        g = df[k].to_numpy()
        n = net_returns(g, turnover[k], cost, borrow_fee)
        cg, cn = np.cumprod(1 + g), np.cumprod(1 + np.maximum(n, -0.99))
        ax.plot(dates, cg, color=COLORS[k], linewidth=1.6)
        ax.plot(dates, cn, color=COLORS[k], linewidth=1.6, linestyle=(0, (4, 2)))
        ends += [(cg[-1], f"{LABELS[k].split(' (')[0]}, gross", COLORS[k]),
                 (cn[-1], f"{LABELS[k].split(' (')[0]}, net", COLORS[k])]
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ys = np.log10([e[0] for e in ends])
    order = np.argsort(ys)
    placed = ys.copy()
    gap = (math.log10(hi) - math.log10(lo)) * 0.05
    for a, b in zip(order[:-1], order[1:]):
        if placed[b] - placed[a] < gap:
            placed[b] = placed[a] + gap
    for (v, lab, col), yv in zip(ends, placed):
        ax.text(dates[-1], 10 ** yv, "  " + lab, va="center", fontsize=8.5, color=col)
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=365 * 14))
    ax.set_ylabel("Growth of $1 (log scale)")
    plots.titles(fig, "Lab Figure 14.2  Paper returns and implementable returns",
                 _sub(sim, f"Solid: gross factor returns · dashed: after a one-way cost of {cost * 1e4:.0f} bp and a "
                           f"{borrow_fee:.2%} annual borrow fee"))
    plots.finish(fig, out / "fig14_2_cumulative.png")
    plt.close(fig)


def fig_turnover(b, out):
    fig, ax, plt = plots.new_figure(8.5, 4.4)
    y = np.arange(len(b))[::-1]
    ax.barh(y + 0.18, b["published"], height=0.34, color=plots.BLUE, label="Published (Novy-Marx and Velikov 2016)")
    ax.barh(y - 0.18, b["simulated"], height=0.34, color=plots.AQUA, label="Simulated universe (mechanical)")
    for yi, (_, r) in zip(y, b.iterrows()):
        ax.text(r.published + 0.01, yi + 0.18, f"{r.published:.0%}", va="center", fontsize=9, color=plots.INK)
        ax.text(r.simulated + 0.01, yi - 0.18, f"{r.simulated:.0%}", va="center", fontsize=9, color=plots.INK)
    ax.set_yticks(y, b["strategy"], fontsize=9, color=plots.INK)
    plots.percent_axis(ax, "x")
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", color=plots.GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_xlabel("One-sided monthly turnover (share of each side replaced per month)")
    plots.titles(fig, "Lab Figure 14.3  Turnover is set by how fast the signal changes",
                 "Value-weighted 30/70 sorts · value is rebalanced yearly; momentum and reversal monthly · "
                 "the simulated universe has no delistings or new listings")
    plots.finish(fig, out / "fig14_3_turnover.png")
    plt.close(fig)


def fig_capacity(d, out, sim, meta):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    for k in STRATEGIES:
        r = d[k]
        ax.plot(r["aum"], r["net"] * 100, color=COLORS[k], linewidth=2, label=LABELS[k])
        if np.isfinite(r["zero_net_aum"]):
            ax.plot(r["zero_net_aum"], 0, "o", color=COLORS[k], markersize=7, markeredgecolor=plots.SURFACE)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_xscale("log")
    lo = min(-0.5, min(np.nanmin(r["net"]) for r in d.values()) * 100)
    ax.set_ylim(max(lo, -1.5), None)
    ax.set_xlim(d["value"]["aum"][0], d["value"]["aum"][-1])
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ticks = [1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
    ax.set_xticks(ticks, ["$10M", "$100M", "$1B", "$10B", "$100B", "$1T"])
    ax.set_xlabel("Assets under management (long side)")
    ax.set_ylabel("Net return (% per month)")
    plots.titles(fig, "Lab Figure 14.4  Capacity: returns shrink as the strategy grows",
                 _sub(sim, f"Square-root impact (Y = {meta['y']}), trades spread over {meta['days']} days, market of "
                           f"${meta['total_cap'] / 1e12:.0f} trillion in a simulated universe · dots: zero net return"))
    plots.finish(fig, out / "fig14_4_capacity.png")
    plt.close(fig)


def fig_banding(e, out, cost, high_cost=0.005):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
    fig.patch.set_facecolor(plots.SURFACE)
    for ax, k in zip(axes, STRATEGIES):
        plots.style_axes(ax)
        g = e[e["key"] == k]
        x = np.arange(len(g))
        ax.bar(x - 0.27, g["gross_mean"] * 100, width=0.27, color=plots.INK_3, label="Expected gross")
        ax.bar(x, g["net_mean"] * 100, width=0.27, color=COLORS[k], alpha=0.55, label=f"Net, {cost * 1e4:.0f} bp")
        ax.bar(x + 0.27, g["net_mean_high"] * 100, width=0.27, color=COLORS[k], label=f"Net, {high_cost * 1e4:.0f} bp")
        ax.set_xticks(x, ["30/70"] + [f"hold {h:.0%}" for h in g["hold_frac"].iloc[1:]], fontsize=8.5)
        ax.axhline(0, color=plots.INK_3, linewidth=1)
        tv = g["turnover"].to_numpy()
        ax.set_title(f"{LABELS[k].split(' (')[0]}\nturnover {tv[0]:.0%} → {tv[-1]:.0%}", loc="left", fontsize=9.5,
                     color=plots.INK, pad=6)
    axes[0].set_ylabel("% per month")
    plots.titles(fig, "Lab Figure 14.5  Buy/hold spreads trade a little signal for a lot less trading",
                 f"Simulated universe, true expected returns · gray: gross · light: net at {cost * 1e4:.0f} bp · dark: "
                 f"net at {high_cost * 1e4:.0f} bp · 'hold 50%': buy in the top 30%, keep while in the top 50%")
    plots.finish(fig, out / "fig14_5_banding.png")
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


def _money(v):
    if not np.isfinite(v):
        return "beyond grid"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if v >= div:
            return f"${v / div:.1f}{unit}"
    return f"${v:,.0f}"


def write_report(out, df, a, b, c, sens, d, e, args, sim):
    period = f"{df.index[0]} to {df.index[-1]}"
    md = ["# Lab 14 results: What Survives the Trip from Paper to Portfolio?", ""]
    if sim:
        md += ["> **Simulated factor data.** Parts A, C and D test that the code runs; they say nothing about real "
               "markets. Parts B and E use a simulated stock universe by design.", ""]
    md += ["**Claim on trial:** a strategy that is profitable in a backtest is profitable to trade.", "",
           f"Sample {period}. Headline one-way cost {args.cost * 1e4:.0f} basis points; short-borrowing fee "
           f"{args.borrow_fee:.2%} a year; turnover source: {args.turnover}.", ""]
    md += ["## Part A: Gross performance", "",
           _md(a.drop(columns="key").rename(columns={"strategy": "Strategy", "mean_monthly": "Mean (monthly)",
                                                     "t_stat": "t", "sharpe": "Sharpe (annual)",
                                                     "vol_annual": "Volatility (annual)"}),
               {"Mean (monthly)": _num(".2%"), "t": _num(".2f"), "Sharpe (annual)": _num(".2f"),
                "Volatility (annual)": _num(".1%")}), ""]
    md += ["## Part B: Turnover", "",
           _md(b.drop(columns=["key", "simulated_gross"]).rename(columns={
               "strategy": "Strategy", "published": "Published (NMV 2016)", "simulated": "Simulated universe"}),
               {"Published (NMV 2016)": _num(".1%"), "Simulated universe": _num(".1%")}), "",
           "One-sided monthly turnover, average of the long and short sides. NMV = Novy-Marx and Velikov (2016), "
           "Table 3, value-weighted portfolios.", "", "![Lab Figure 14.3](fig14_3_turnover.png)", ""]
    t = c["table"]
    md += ["## Part C: Costs", "",
           _md(t[["strategy", "turnover", "gross_mean", "cost_per_month", "net_mean", "net_t", "gross_sharpe",
                  "net_sharpe", "break_even_one_way", "rank_gross", "rank_net"]].rename(columns={
               "strategy": "Strategy", "turnover": "Turnover", "gross_mean": "Gross mean", "cost_per_month": "Cost",
               "net_mean": "Net mean", "net_t": "Net t", "gross_sharpe": "Gross Sharpe", "net_sharpe": "Net Sharpe",
               "break_even_one_way": "Break-even one-way cost", "rank_gross": "Rank gross", "rank_net": "Rank net"}),
               {"Turnover": _num(".1%"), "Gross mean": _num(".2%"), "Cost": _num(".2%"), "Net mean": _num("+.2%"),
                "Net t": _num(".2f"), "Gross Sharpe": _num(".2f"), "Net Sharpe": _num("+.2f"),
                "Break-even one-way cost": lambda v: f"{v * 1e4:.0f} bp"}), "",
           "Means and costs are per month. Cost = 4 × turnover × one-way cost (two sides, buys and sells) plus the "
           "borrow fee. Break-even cost: the one-way cost at which the average net return is zero.", "",
           "**Net Sharpe ratio at different one-way costs**", "",
           _md(sens.assign(one_way_cost=sens["one_way_cost"] * 1e4).rename(columns={"one_way_cost": "One-way cost (bp)"}),
               {"One-way cost (bp)": _num(".0f")} | {LABELS[k]: _num("+.2f") for k in STRATEGIES}), "",
           "![Lab Figure 14.1](fig14_1_net_vs_cost.png)", "", "![Lab Figure 14.2](fig14_2_cumulative.png)", ""]
    rows = [{"Strategy": LABELS[k], "Net return at $100M": d[k]["net"][np.searchsorted(d[k]["aum"], 1e8)],
             "Net return at $10B": d[k]["net"][np.searchsorted(d[k]["aum"], 1e10)],
             "AUM where net return halves": _money(d[k]["half_net_aum"]),
             "AUM where net return reaches zero": _money(d[k]["zero_net_aum"])} for k in STRATEGIES]
    md += ["## Part D: Capacity", "",
           f"Square-root impact with Y = {args.impact_y}, trades spread over {args.trade_days} trading days, "
           f"half-spreads from 2 to 50 bp by size, daily volume {0.4:.1f}% of market cap, in a simulated universe "
           f"scaled to ${args.total_cap / 1e12:.0f} trillion. Gross returns from Part A.", "",
           _md(pd.DataFrame(rows), {"Net return at $100M": _num("+.2%"), "Net return at $10B": _num("+.2%")}), "",
           "Treat these numbers as an illustration of the mechanics, not estimates of real capacity: they rest on "
           "the stated assumptions.", "", "![Lab Figure 14.4](fig14_4_capacity.png)", ""]
    md += ["## Part E: Mitigation with known truth (simulated universe)", "",
           _md(e.drop(columns=["key", "hold_frac"]).rename(columns={
               "strategy": "Strategy", "rule": "Rule", "turnover": "Turnover", "gross_mean": "Expected gross",
               "net_mean": f"Net at {args.cost * 1e4:.0f} bp", "net_mean_high": "Net at 50 bp"}),
               {"Turnover": _num(".1%"), "Expected gross": _num(".2%"), f"Net at {args.cost * 1e4:.0f} bp": _num("+.2%"),
                "Net at 50 bp": _num("+.2%")}), "",
           "Expected gross: the true expected long-short return in the simulation (no sampling noise), per month.", "",
           "![Lab Figure 14.5](fig14_5_banding.png)", ""]
    md += ["## Your verdict", "",
           "1. Which strategies are profitable to trade at realistic costs, and which only on paper? Use the "
           "break-even costs and the cost sensitivity table.",
           "2. How does the ranking change after costs, and why?",
           "3. Part D rests on assumptions. Which assumption matters most for capacity, and how would you check it?",
           "4. Part E shows a cost-aware design. Is it fair to compare a cost-aware strategy with the published "
           "paper version? What would you need to report?", ""]
    (out / "lab14_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--ff3-file", default=None, help="Fama/French 3 Factors (monthly) zip or CSV")
    ap.add_argument("--mom-file", default=None, help="Momentum Factor (Mom) zip or CSV")
    ap.add_argument("--rev-file", default=None, help="Short-Term Reversal Factor (ST Rev) zip or CSV")
    ap.add_argument("--start", default="1963-07")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost", type=float, default=0.002, help="headline one-way cost (decimal; 0.002 = 20 bp)")
    ap.add_argument("--borrow-fee", type=float, default=0.0025, help="annual short-borrowing fee (decimal)")
    ap.add_argument("--turnover", choices=["published", "simulated"], default="published")
    ap.add_argument("--impact-y", type=float, default=1.0, help="square-root impact coefficient")
    ap.add_argument("--trade-days", type=int, default=5, help="days over which each month's trades are spread")
    ap.add_argument("--total-cap", type=float, default=50e12, help="total market cap of the simulated universe ($)")
    ap.add_argument("--n-stocks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026, help="seed for the simulated universe")
    ap.add_argument("--sim-seed", type=int, default=0, help="seed for simulated factor data (--source sim)")
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        df = load_factors(args.source, args.ff3_file, args.mom_file, args.rev_file, args.start, args.end,
                          args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = df.attrs.get("simulated", False)
    period = f"{df.index[0]} to {df.index[-1]}"

    universe = simulate_universe(args.n_stocks, 480, seed=args.seed)
    a = part_a(df)
    b = part_b(universe)
    turnover = PUBLISHED_TURNOVER if args.turnover == "published" else dict(zip(b["key"], b["simulated"]))
    c = part_c(df, turnover, args.cost, args.borrow_fee)
    sens = cost_sensitivity(df, turnover, args.borrow_fee)
    micro = market_microstructure(universe, args.total_cap)
    gross = dict(zip(a["key"], a["mean_monthly"] - args.borrow_fee / 12))
    d = part_d(universe, gross, turnover, micro, args.impact_y, args.trade_days)
    e = part_e(universe, args.cost)

    fig_net_vs_cost(c, out, sim, args.cost, period)
    fig_cumulative(df, turnover, args.cost, args.borrow_fee, out, sim)
    fig_turnover(b, out)
    fig_capacity(d, out, sim, {"y": args.impact_y, "days": args.trade_days, "total_cap": args.total_cap})
    fig_banding(e, out, args.cost)

    a.to_csv(out / "table_gross.csv", index=False)
    b.to_csv(out / "table_turnover.csv", index=False)
    c["table"].to_csv(out / "table_costs.csv", index=False)
    sens.to_csv(out / "table_cost_sensitivity.csv", index=False)
    pd.DataFrame({"aum": d["value"]["aum"], **{f"net_{k}": d[k]["net"] for k in STRATEGIES}}).to_csv(
        out / "table_capacity.csv", index=False)
    e.to_csv(out / "table_banding.csv", index=False)
    write_report(out, df, a, b, c, sens, d, e, args, sim)

    t = c["table"].set_index("key")
    summary = {
        "simulated": sim, "period": period, "cost": args.cost, "borrow_fee": args.borrow_fee,
        "turnover_source": args.turnover, "turnover": turnover,
        "gross_mean": t["gross_mean"].to_dict(), "net_mean": t["net_mean"].to_dict(),
        "break_even_one_way": t["break_even_one_way"].to_dict(),
        "rank_gross": t["rank_gross"].to_dict(), "rank_net": t["rank_net"].to_dict(),
        "simulated_turnover": dict(zip(b["key"], b["simulated"])),
        "capacity_zero_net_aum": {k: d[k]["zero_net_aum"] for k in STRATEGIES},
        "banding": e.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    for k in STRATEGIES:
        print(f"{LABELS[k]:30s} gross {t.loc[k, 'gross_mean']:+.2%}/mo  net {t.loc[k, 'net_mean']:+.2%}/mo  "
              f"break-even {t.loc[k, 'break_even_one_way'] * 1e4:.0f} bp  rank {t.loc[k, 'rank_gross']}"
              f"->{t.loc[k, 'rank_net']}  sim turnover {summary['simulated_turnover'][k]:.0%}")
    print(f"Report: {out / 'lab14_report.md'}")
    if sim:
        print("NOTE: simulated factor data for Parts A, C and D. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
