"""Lab 1: Brilliant by Chance.

Chapter 1 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "A strategy with an impressive backtest has skill."

WHAT THE LAB DOES
  Part A  Signal and noise. Measures the U.S. market's Sharpe ratio and how
          much it varies from one 10-year window to the next.
  Part B  Brilliant by chance. Builds 1,000 market-timing strategies that
          switch between stocks and Treasury bills AT RANDOM, so every one
          has zero skill by construction. Tests each in the first half of
          the sample (the "backtest"), picks the winners, and follows them
          in the second half.
  Part C  The cost of searching. Shows how the best backtest improves as
          more random strategies are tried.

DATA
  Monthly U.S. market excess returns and the one-month Treasury bill rate
  from the Kenneth R. French Data Library (downloaded automatically).

USAGE
  python lab01.py                              # real data (downloads once)
  python lab01.py --data-file F-F_Research_Data_Factors_CSV.zip
  python lab01.py --n-strategies 5000 --seed 7
  python lab01.py --source sim                 # fake data, to test the code

OUTPUT
  report/lab01_report.md, five figures (PNG) and CSV tables in report/.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, load_french_factors, simulated_market
from fintest.stats import (
    bonferroni_threshold,
    expected_max_normal,
    normal_sf,
    ols_newey_west,
    sharpe_ratio,
    sharpe_standard_error,
    years_to_detect,
)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_market(source: str = "french", data_file: str | None = None, seed: int = 0) -> pd.DataFrame:
    """Monthly market excess return ('mkt_rf') and risk-free rate ('rf'), decimals."""
    if source == "sim":
        df = simulated_market(seed=seed)
    else:
        df = load_french_factors("ff3_monthly", path=data_file)
    out = pd.DataFrame({"mkt_rf": df["Mkt-RF"], "rf": df["RF"]}).dropna()
    out.attrs["simulated"] = bool(df.attrs.get("simulated", False))
    return out


# ---------------------------------------------------------------------------
# Part A: signal and noise
# ---------------------------------------------------------------------------


def part_a(mkt: pd.DataFrame, window_years: int = 10) -> dict:
    ex = mkt["mkt_rf"].to_numpy()
    n_years = len(ex) / 12
    sr = float(sharpe_ratio(ex))
    window = window_years * 12
    rolling = (mkt["mkt_rf"].rolling(window).mean() / mkt["mkt_rf"].rolling(window).std() * math.sqrt(12)).dropna()
    detect = pd.DataFrame({"true_sharpe": [0.25, 0.5, 0.75, 1.0, 1.5]})
    detect["years_needed"] = detect["true_sharpe"].map(years_to_detect)
    return {
        "start": str(mkt.index[0]),
        "end": str(mkt.index[-1]),
        "years": n_years,
        "mean_excess_ann": float(ex.mean() * 12),
        "vol_ann": float(ex.std(ddof=1) * math.sqrt(12)),
        "sharpe": sr,
        "t_stat": sr * math.sqrt(n_years),
        "rolling": rolling,
        "rolling_min": float(rolling.min()),
        "rolling_max": float(rolling.max()),
        "rolling_share_negative": float((rolling < 0).mean()),
        "window_years": window_years,
        "se_window": sharpe_standard_error(window_years),
        "detect": detect,
    }


# ---------------------------------------------------------------------------
# Part B: 1,000 random strategies
# ---------------------------------------------------------------------------


def random_positions(n_months: int, n_strategies: int, rng: np.random.Generator,
                     min_hold: float = 3.0, max_hold: float = 24.0) -> np.ndarray:
    """Random in/out-of-market positions, shape (n_months, n_strategies).

    Each strategy is a two-state Markov chain: in the market (1) or in
    Treasury bills (0). Its average holding period is drawn uniformly
    between min_hold and max_hold months, so the strategies look like real
    timing rules (they hold positions for a while) but use no information
    at all. The position for month t is set before month t's return is
    known, so there is no look-ahead.
    """
    hold = rng.uniform(min_hold, max_hold, size=n_strategies)
    p_switch = 1.0 / hold
    pos = np.empty((n_months, n_strategies), dtype=np.int8)
    pos[0] = rng.integers(0, 2, size=n_strategies)
    switches = rng.random((n_months, n_strategies)) < p_switch
    for t in range(1, n_months):
        pos[t] = np.where(switches[t], 1 - pos[t - 1], pos[t - 1])
    return pos


@dataclass
class PartB:
    split: str
    n_strategies: int
    in_sample: dict
    out_sample: dict
    t_is: np.ndarray
    t_oos: np.ndarray
    alpha_is: np.ndarray
    alpha_oos: np.ndarray
    sharpe_is: np.ndarray
    market_sharpe_is: float
    market_sharpe_oos: float
    best: int
    top: list[int]
    pos: np.ndarray = field(repr=False)
    strat_excess: np.ndarray = field(repr=False)


def part_b(mkt: pd.DataFrame, n_strategies: int = 1000, split: str | None = None, seed: int = 0,
           lags: int | None = None) -> PartB:
    rng = np.random.default_rng(seed)
    n = len(mkt)
    split_idx = n // 2 if split is None else int(np.searchsorted(mkt.index.astype(str), split, side="right"))
    if not 60 <= split_idx <= n - 60:
        raise ValueError("each half of the sample needs at least 60 months; choose another --split")
    m = mkt["mkt_rf"].to_numpy()
    pos = random_positions(n, n_strategies, rng)
    strat_excess = pos * m[:, None]  # in the market: market excess return; in bills: 0

    is_slice, oos_slice = slice(0, split_idx), slice(split_idx, n)
    reg_is = ols_newey_west(strat_excess[is_slice], m[is_slice], lags=lags)
    reg_oos = ols_newey_west(strat_excess[oos_slice], m[oos_slice], lags=lags)
    t_is, t_oos = reg_is["t"][0], reg_oos["t"][0]
    alpha_is, alpha_oos = reg_is["coef"][0] * 12, reg_oos["coef"][0] * 12
    order = np.argsort(-t_is)

    def window(sl):
        idx = mkt.index[sl]
        return {"start": str(idx[0]), "end": str(idx[-1]), "months": len(idx)}

    return PartB(
        split=str(mkt.index[split_idx - 1]),
        n_strategies=n_strategies,
        in_sample=window(is_slice),
        out_sample=window(oos_slice),
        t_is=t_is, t_oos=t_oos, alpha_is=alpha_is, alpha_oos=alpha_oos,
        sharpe_is=sharpe_ratio(strat_excess[is_slice]),
        market_sharpe_is=float(sharpe_ratio(m[is_slice])),
        market_sharpe_oos=float(sharpe_ratio(m[oos_slice])),
        best=int(order[0]),
        top=[int(i) for i in order[:10]],
        pos=pos,
        strat_excess=strat_excess,
    )


def summarize_b(b: PartB) -> dict:
    n = b.n_strategies
    expected_one_sided = n * float(normal_sf(1.96))
    top10 = pd.DataFrame({
        "strategy": b.top,
        "alpha_in_sample_pct": b.alpha_is[b.top] * 100,
        "t_in_sample": b.t_is[b.top],
        "alpha_out_of_sample_pct": b.alpha_oos[b.top] * 100,
        "t_out_of_sample": b.t_oos[b.top],
    })
    sig_is = b.t_is > 1.96
    return {
        "n_strategies": n,
        "in_sample": b.in_sample,
        "out_of_sample": b.out_sample,
        "count_t_above_1_96": int(sig_is.sum()),
        "expected_count_t_above_1_96": expected_one_sided,
        "count_t_above_3": int((b.t_is > 3).sum()),
        "bonferroni_threshold": bonferroni_threshold(n),
        "count_above_bonferroni": int((b.t_is > bonferroni_threshold(n)).sum()),
        "best_t_in_sample": float(b.t_is[b.best]),
        "expected_best_t": expected_max_normal(n, n_sims=4000),
        "best_alpha_in_sample_pct": float(b.alpha_is[b.best] * 100),
        "best_sharpe_in_sample": float(b.sharpe_is[b.best]),
        "market_sharpe_in_sample": b.market_sharpe_is,
        "best_t_out_of_sample": float(b.t_oos[b.best]),
        "best_alpha_out_of_sample_pct": float(b.alpha_oos[b.best] * 100),
        "market_sharpe_out_of_sample": b.market_sharpe_oos,
        "winners_mean_t_out_of_sample": float(b.t_oos[sig_is].mean()) if sig_is.any() else float("nan"),
        "winners_share_positive_out_of_sample": float((b.t_oos[sig_is] > 0).mean()) if sig_is.any() else float("nan"),
        "correlation_t_in_vs_out": float(np.corrcoef(b.t_is, b.t_oos)[0, 1]),
        "top10": top10,
    }


# ---------------------------------------------------------------------------
# Part C: the cost of searching
# ---------------------------------------------------------------------------


def part_c(b: PartB, n_draws: int = 200, seed: int = 1) -> pd.DataFrame:
    """Median and 90% range of the best in-sample t-statistic when k strategies are tried."""
    rng = np.random.default_rng(seed)
    ks = [k for k in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if k <= b.n_strategies]
    rows = []
    for k in ks:
        best = np.array([b.t_is[rng.choice(b.n_strategies, size=k, replace=False)].max() for _ in range(n_draws)])
        rows.append({"strategies_tried": k, "best_t_median": float(np.median(best)),
                     "best_t_p05": float(np.percentile(best, 5)), "best_t_p95": float(np.percentile(best, 95)),
                     "approx_sqrt_2lnk": math.sqrt(2 * math.log(k)) if k > 1 else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(simulated: bool, text: str) -> str:
    return text + ("  ·  SIMULATED DATA: tests the code only" if simulated else "")


def fig_rolling(a: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    r = a["rolling"]
    x = r.index.to_timestamp()
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.axhline(a["sharpe"], color=plots.INK_3, linewidth=1, linestyle=(0, (3, 3)))
    ax.plot(x, r.values, color=plots.BLUE, linewidth=2)
    ax.text(x[0], a["sharpe"], f"full-sample average {a['sharpe']:.2f}", va="bottom", ha="left", fontsize=9,
            color=plots.INK_2, bbox={"facecolor": plots.SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_ylabel(f"Sharpe ratio over the previous {a['window_years']} years")
    plots.titles(fig, f"Lab Figure 1.1  The market's Sharpe ratio, measured {a['window_years']} years at a time",
                 _sub(simulated, f"U.S. market excess return, {a['start']} to {a['end']} · same market, very "
                                 "different decades"))
    plots.finish(fig, out / "fig1_1_rolling_sharpe.png")
    plt.close(fig)


def fig_histogram(b: PartB, s: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    t = b.t_is
    bins = np.linspace(min(-4.5, t.min() - 0.2), max(4.5, t.max() + 0.2), 46)
    width = bins[1] - bins[0]
    ax.hist(t, bins=bins, color=plots.BLUE, edgecolor=plots.SURFACE, linewidth=1)
    grid = np.linspace(bins[0], bins[-1], 400)
    ax.plot(grid, len(t) * width * np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi), color=plots.INK_2,
            linewidth=1.5)
    for thr, label in ((1.96, "t = 1.96"), (3.0, "t = 3")):
        ax.axvline(thr, color=plots.ORANGE if thr < 3 else plots.RED, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.text(thr, ax.get_ylim()[1] * 0.97, f" {label}", fontsize=9, color=plots.INK, va="top")
    ax.set_xlabel("t-statistic of alpha versus buy-and-hold (in-sample, Newey–West)")
    ax.set_ylabel("Number of strategies")
    plots.titles(fig, f"Lab Figure 1.2  {b.n_strategies:,} strategies with zero skill",
                 _sub(simulated, f"{s['count_t_above_1_96']} pass t > 1.96 (about {s['expected_count_t_above_1_96']:.0f} "
                                 f"expected by chance) · grey curve: what pure luck predicts"))
    plots.finish(fig, out / "fig1_2_t_histogram.png")
    plt.close(fig)


def fig_best(mkt: pd.DataFrame, b: PartB, s: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    x = mkt.index.to_timestamp()
    rf = mkt["rf"].to_numpy()
    market_total = mkt["mkt_rf"].to_numpy() + rf
    best_total = b.strat_excess[:, b.best] + rf
    split_ts = pd.Period(b.split, freq="M").to_timestamp(how="end")
    ax.axvspan(split_ts, x[-1], color=plots.SHADE, zorder=0)
    ax.plot(x, np.cumprod(1 + market_total), color=plots.INK_3, linewidth=1.8)
    ax.plot(x, np.cumprod(1 + best_total), color=plots.ORANGE, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.text(x[len(x) // 4], ax.get_ylim()[1] * 0.6, "BACKTEST\n(strategy chosen here)", fontsize=9,
            color=plots.INK_2, ha="center", va="top")
    ax.text(split_ts + (x[-1] - split_ts) / 2, ax.get_ylim()[1] * 0.6, "AFTER SELECTION\n(what an investor gets)",
            fontsize=9, color=plots.INK_2, ha="center", va="top")
    # End labels, nudged apart in log space so they never overlap.
    ends = sorted([[math.log10(np.cumprod(1 + best_total)[-1]), "  best random strategy"],
                   [math.log10(np.cumprod(1 + market_total)[-1]), "  market"]])
    if ends[1][0] - ends[0][0] < 0.18:
        mid = (ends[0][0] + ends[1][0]) / 2
        ends[0][0], ends[1][0] = mid - 0.09, mid + 0.09
    for y_log, label in ends:
        ax.text(x[-1], 10 ** y_log, label, fontsize=9, color=plots.INK, va="center")
    ax.set_xlim(x[0], x[-1] + (x[-1] - x[0]) * 0.12)
    plots.titles(fig, "Lab Figure 1.3  The best of 1,000 random strategies",
                 _sub(simulated, f"Backtest alpha {s['best_alpha_in_sample_pct']:+.1f}% a year (t = {s['best_t_in_sample']:.2f}); "
                                 f"after selection {s['best_alpha_out_of_sample_pct']:+.1f}% (t = {s['best_t_out_of_sample']:.2f})"))
    plots.finish(fig, out / "fig1_3_best_strategy.png")
    plt.close(fig)


def fig_scatter(b: PartB, s: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure(7.2, 5.4)
    winners = b.t_is > 1.96
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.axvline(1.96, color=plots.ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.scatter(b.t_is[~winners], b.t_oos[~winners], s=12, color=plots.INK_3, alpha=0.45, linewidths=0)
    ax.scatter(b.t_is[winners], b.t_oos[winners], s=26, color=plots.ORANGE, edgecolors=plots.SURFACE,
               linewidths=0.8, zorder=3)
    ax.set_xlabel("t-statistic of alpha in the backtest")
    ax.set_ylabel("t-statistic of alpha after selection")
    ax.text(1.96, ax.get_ylim()[1], "  backtest 'winners'", fontsize=9, color=plots.INK, va="top")
    ax.grid(axis="x", color=plots.GRID, linewidth=0.8)
    plots.titles(fig, "Lab Figure 1.4  Backtest results do not predict later results",
                 _sub(simulated, f"Correlation {s['correlation_t_in_vs_out']:+.2f} · winners' average t afterward "
                                 f"{s['winners_mean_t_out_of_sample']:+.2f}"))
    plots.finish(fig, out / "fig1_4_in_vs_out.png", top=0.88)
    plt.close(fig)


def fig_search(c: pd.DataFrame, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    k = c["strategies_tried"].to_numpy()
    ax.fill_between(k, c["best_t_p05"], c["best_t_p95"], color=plots.BLUE, alpha=0.15, linewidth=0)
    ax.plot(k, c["best_t_median"], color=plots.BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=plots.SURFACE, markeredgewidth=2)
    for thr, label in ((1.96, "t = 1.96"), (3.0, "t = 3")):
        ax.axhline(thr, color=plots.INK_3, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(k[0], thr, f" {label}", fontsize=9, color=plots.INK_2, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(k, [f"{v:,}" for v in k])
    ax.set_xlabel("Number of random strategies tried")
    ax.set_ylabel("Best backtest t-statistic")
    plots.titles(fig, "Lab Figure 1.5  Try more strategies, get a better-looking winner",
                 _sub(simulated, "Line: median best t across 200 random draws · band: 5th to 95th percentile"))
    plots.finish(fig, out / "fig1_5_search_cost.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(df: pd.DataFrame, fmt: dict[str, str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            f = fmt.get(c, "")
            cells.append(format(int(r[c]) if f.endswith("d") else r[c], f))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, a: dict, s: dict, c: pd.DataFrame, simulated: bool, args) -> None:
    md = ["# Lab 1 results: Brilliant by Chance", ""]
    if simulated:
        md += ["> **Simulated data.** These numbers test that the lab runs. They are produced from random draws "
               "and say nothing about real markets. Run without `--source sim` for real results.", ""]
    md += [
        "**Claim on trial:** a strategy with an impressive backtest has skill.", "",
        "## Part A: Signal and noise", "",
        f"- Sample: {a['start']} to {a['end']} ({a['years']:.0f} years).",
        f"- Market excess return {a['mean_excess_ann']:.1%} a year, volatility {a['vol_ann']:.1%}, "
        f"Sharpe ratio {a['sharpe']:.2f} (t = {a['t_stat']:.2f}).",
        f"- Over rolling {a['window_years']}-year windows, the Sharpe ratio ranged from {a['rolling_min']:.2f} to "
        f"{a['rolling_max']:.2f}, and was negative in {a['rolling_share_negative']:.0%} of windows. "
        f"The rough standard error for a {a['window_years']}-year estimate is {a['se_window']:.2f}.", "",
        "**Years of data needed before a true Sharpe ratio reaches t = 1.96**", "",
        _md(a["detect"].rename(columns={"true_sharpe": "True Sharpe ratio", "years_needed": "Years needed"}),
            {"True Sharpe ratio": ".2f", "Years needed": ".0f"}), "",
        "![Lab Figure 1.1](fig1_1_rolling_sharpe.png)", "",
        "## Part B: 1,000 strategies with zero skill", "",
        f"- {s['n_strategies']:,} random market-timing strategies; seed {args.seed}.",
        f"- Backtest (in-sample): {s['in_sample']['start']} to {s['in_sample']['end']}. "
        f"After selection (out-of-sample): {s['out_of_sample']['start']} to {s['out_of_sample']['end']}.",
        "- Skill is measured as alpha against buy-and-hold, with Newey–West t-statistics.", "",
        "| Measure | Result |", "| --- | --- |",
        f"| Strategies with backtest t > 1.96 | {s['count_t_above_1_96']} (about {s['expected_count_t_above_1_96']:.0f} expected by luck) |",
        f"| Strategies with backtest t > 3 | {s['count_t_above_3']} |",
        f"| Bonferroni threshold for {s['n_strategies']:,} tests | t > {s['bonferroni_threshold']:.2f}; {s['count_above_bonferroni']} pass |",
        f"| Best backtest t-statistic | {s['best_t_in_sample']:.2f} (about {s['expected_best_t']:.2f} expected for the best of {s['n_strategies']:,} by luck) |",
        f"| Best strategy's backtest alpha | {s['best_alpha_in_sample_pct']:+.2f}% a year |",
        f"| Best strategy's backtest Sharpe ratio | {s['best_sharpe_in_sample']:.2f} (market {s['market_sharpe_in_sample']:.2f}) |",
        f"| Best strategy after selection | alpha {s['best_alpha_out_of_sample_pct']:+.2f}% a year, t = {s['best_t_out_of_sample']:.2f} |",
        f"| Backtest winners (t > 1.96) after selection | average t = {s['winners_mean_t_out_of_sample']:+.2f}; {s['winners_share_positive_out_of_sample']:.0%} positive |",
        f"| Correlation of backtest t and later t | {s['correlation_t_in_vs_out']:+.2f} |", "",
        "**The ten best backtests, and what happened next**", "",
        _md(s["top10"].rename(columns={
            "strategy": "Strategy", "alpha_in_sample_pct": "Backtest alpha (%/yr)", "t_in_sample": "Backtest t",
            "alpha_out_of_sample_pct": "Later alpha (%/yr)", "t_out_of_sample": "Later t"}),
            {"Strategy": "d", "Backtest alpha (%/yr)": "+.2f", "Backtest t": ".2f", "Later alpha (%/yr)": "+.2f",
             "Later t": "+.2f"}), "",
        "![Lab Figure 1.2](fig1_2_t_histogram.png)", "",
        "![Lab Figure 1.3](fig1_3_best_strategy.png)", "",
        "![Lab Figure 1.4](fig1_4_in_vs_out.png)", "",
        "## Part C: The cost of searching", "",
        _md(c.rename(columns={"strategies_tried": "Strategies tried", "best_t_median": "Best t (median)",
                              "best_t_p05": "5th pct", "best_t_p95": "95th pct",
                              "approx_sqrt_2lnk": "Approximation sqrt(2 ln k)"}),
            {"Strategies tried": ",d", "Best t (median)": ".2f", "5th pct": ".2f", "95th pct": ".2f",
             "Approximation sqrt(2 ln k)": ".2f"}), "",
        "![Lab Figure 1.5](fig1_5_search_cost.png)", "",
        "## Your verdict", "",
        "Write two or three sentences: does an impressive backtest, on its own, show skill? "
        "Use the numbers above. Then name one piece of information you would ask for before believing a backtest.",
        "",
    ]
    (out / "lab01_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french",
                    help="french: Kenneth French Data Library (default); sim: fake data for testing")
    ap.add_argument("--data-file", default=None, help="a French library zip or CSV you downloaded yourself")
    ap.add_argument("--n-strategies", type=int, default=1000)
    ap.add_argument("--split", default=None,
                    help="last month of the backtest, e.g. 1975-12 (default: middle of the sample)")
    ap.add_argument("--window-years", type=int, default=10)
    ap.add_argument("--nw-lags", type=int, default=None, help="Newey-West lags (default: rule of thumb)")
    ap.add_argument("--seed", type=int, default=2026, help="seed for the random strategies")
    ap.add_argument("--sim-seed", type=int, default=0, help="seed for simulated market data (--source sim)")
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        mkt = load_market(args.source, args.data_file, seed=args.sim_seed)
    except DataUnavailable as e:
        raise SystemExit(str(e)) from None
    simulated = mkt.attrs.get("simulated", False)

    a = part_a(mkt, args.window_years)
    b = part_b(mkt, args.n_strategies, args.split, args.seed, args.nw_lags)
    s = summarize_b(b)
    c = part_c(b)

    fig_rolling(a, out, simulated)
    fig_histogram(b, s, out, simulated)
    fig_best(mkt, b, s, out, simulated)
    fig_scatter(b, s, out, simulated)
    fig_search(c, out, simulated)

    a["detect"].to_csv(out / "table_years_to_detect.csv", index=False)
    s["top10"].to_csv(out / "table_top10.csv", index=False)
    c.to_csv(out / "table_search_cost.csv", index=False)
    pd.DataFrame({"t_in_sample": b.t_is, "t_out_of_sample": b.t_oos, "alpha_in_sample": b.alpha_is,
                  "alpha_out_of_sample": b.alpha_oos}).to_csv(out / "table_all_strategies.csv",
                                                              index_label="strategy")
    summary = {k: v for k, v in s.items() if k != "top10"}
    summary.update({f"market_{k}": v for k, v in a.items() if k not in ("rolling", "detect")})
    summary["simulated"] = simulated
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(out, a, s, c, simulated, args)

    print(f"Sample {a['start']} to {a['end']}: market Sharpe {a['sharpe']:.2f}")
    print(f"{s['count_t_above_1_96']} of {s['n_strategies']:,} zero-skill strategies pass t > 1.96 in the backtest "
          f"(about {s['expected_count_t_above_1_96']:.0f} expected by luck)")
    print(f"Best backtest t = {s['best_t_in_sample']:.2f}; after selection t = {s['best_t_out_of_sample']:.2f}")
    print(f"Report: {out / 'lab01_report.md'}")
    if simulated:
        print("NOTE: simulated data. These results test the code; they say nothing about real markets.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
