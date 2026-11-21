"""Lab 2: Four Data Traps.

Chapter 2 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "The data I am testing on represents what an investor could actually
  have done."

WHAT THE LAB DOES
  Part A  Averages. Arithmetic versus geometric returns for the market and
          for small and large stocks (real data).
  Part B  Survivorship. A simulated stock market in which the truth is
          known: compares a full, investable dataset with one missing
          delisting returns and one holding only today's survivors.
  Part C  Look-ahead. A 200-day moving-average timing rule on daily market
          data, traded correctly and with look-ahead; then the same rule
          with its window chosen with hindsight versus walk-forward (real data).
  Part D  Micro-caps and weighting. The size premium in value- and
          equal-weighted portfolios, with and without the smallest stocks,
          before and after its 1981 publication (real data).

DATA (Kenneth R. French Data Library, downloaded automatically)
  Fama/French 3 Factors, monthly and daily; Portfolios Formed on Size.
  Part B needs no data: free sources do not include delisted stocks, so the
  lab builds a market where the right answer is known.

USAGE
  python lab02.py                     # real data (downloads three files once)
  python lab02.py --source sim        # fake data, to test the code
  python lab02.py --monthly-file F-F_Research_Data_Factors_CSV.zip \\
                  --daily-file F-F_Research_Data_Factors_daily_CSV.zip \\
                  --size-file Portfolios_Formed_on_ME_CSV.zip

OUTPUT
  report/lab02_report.md, five figures (PNG), CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import (
    DataUnavailable,
    find_table,
    load_french_factors,
    load_french_tables,
    simulated_daily_market,
    simulated_market,
    simulated_size_deciles,
)
from fintest.stats import geometric_mean, max_drawdown, ols_newey_west

ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_all(source: str, monthly_file=None, daily_file=None, size_file=None, sim_seed: int = 0) -> dict:
    """Monthly market, daily market and size-decile tables (decimal returns)."""
    if source == "sim":
        monthly = simulated_market(seed=sim_seed)
        daily = simulated_daily_market(seed=sim_seed)
        size = simulated_size_deciles(seed=sim_seed)
        simulated = True
    else:
        monthly = load_french_factors("ff3_monthly", path=monthly_file)
        daily = load_french_factors("ff3_daily", path=daily_file)
        size = load_french_tables("size_portfolios", path=size_file)
        simulated = False
    vw = find_table(size, "value", "weight")
    ew = find_table(size, "equal", "weight")
    return {"monthly": monthly, "daily": daily, "size_vw": decile_columns(vw), "size_ew": decile_columns(ew),
            "simulated": simulated}


def decile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The ten size-decile columns, renamed D1 (smallest) to D10 (largest).

    The French size file has columns for several breakpoints (<= 0, terciles,
    quintiles, deciles); the deciles are the last ten columns.
    """
    dec = df.iloc[:, -10:].copy()
    dec.columns = [f"D{i}" for i in range(1, 11)]
    return dec.dropna(how="all")


# ---------------------------------------------------------------------------
# Part A: arithmetic versus geometric averages
# ---------------------------------------------------------------------------


def part_a(data: dict) -> pd.DataFrame:
    m = data["monthly"]
    series = {
        "Treasury bills": m["RF"],
        "Stock market": m["Mkt-RF"] + m["RF"],
        "Largest 10% of stocks": data["size_vw"]["D10"],
        "Smallest 10% of stocks": data["size_vw"]["D1"],
    }
    rows = []
    for name, r in series.items():
        r = r.dropna().to_numpy()
        # Both averages are annualized by compounding the monthly figure, so the
        # comparison is like for like: (1 + monthly mean)^12 - 1 versus the
        # compound growth rate actually achieved.
        arith = (1 + r.mean()) ** 12 - 1
        vol = r.std(ddof=1) * math.sqrt(12)
        geo = geometric_mean(r)
        rows.append({"series": name, "arithmetic": arith, "geometric": geo, "gap": arith - geo,
                     "volatility": vol, "approx_gap": vol ** 2 / 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: survivorship in a simulated market with a known truth
# ---------------------------------------------------------------------------


def simulate_universe(n_firms: int = 2000, n_months: int = 360, seed: int = 0,
                      delist_threshold: float = 0.2, delist_return: float = -0.30,
                      merger_prob: float = 0.003, merger_premium: float = 0.20) -> dict:
    """Simulate a stock market where every firm has the same expected return.

    Each firm's monthly return is the market return plus idiosyncratic noise
    with a firm-specific volatility (5% to 20% a month). No firm has an
    expected-return advantage. A firm is delisted for performance when its
    price falls below `delist_threshold` of its listing price, with a final
    delisting return of `delist_return` (Shumway 1997 suggests about -30%).
    Firms are also acquired at random, at a premium. Every exit is replaced
    by a new listing, so the market keeps about n_firms stocks.

    Returns the full return matrix (months x firms, NaN when not listed) and
    the information needed to build the three datasets.
    """
    rng = np.random.default_rng(seed)
    cap = n_firms * 6
    R = np.full((n_months, cap), np.nan)
    price = np.ones(cap)
    vol = np.zeros(cap)
    alive = np.zeros(cap, dtype=bool)
    exit_month = np.full(cap, -1)
    exit_type = np.zeros(cap, dtype=np.int8)  # 0 none, 1 performance delisting, 2 merger
    alive[:n_firms] = True
    vol[:n_firms] = rng.uniform(0.05, 0.20, n_firms)
    n_used = n_firms
    mkt = 0.008 + 0.045 * rng.standard_normal(n_months)

    for t in range(n_months):
        idx = np.flatnonzero(alive)
        r = np.maximum(mkt[t] + vol[idx] * rng.standard_normal(idx.size), -0.95)
        price[idx] *= 1 + r
        perf = price[idx] < delist_threshold
        merge = (~perf) & (rng.random(idx.size) < merger_prob)
        out = r.copy()
        out[perf] = (1 + r[perf]) * (1 + delist_return) - 1
        out[merge] = (1 + r[merge]) * (1 + merger_premium) - 1
        R[t, idx] = out
        exited = idx[perf | merge]
        alive[exited] = False
        exit_month[exited] = t
        exit_type[idx[perf]] = 1
        exit_type[idx[merge]] = 2
        k = exited.size
        if k and n_used + k <= cap:
            new = np.arange(n_used, n_used + k)
            alive[new] = True
            price[new] = 1.0
            vol[new] = rng.uniform(0.05, 0.20, k)
            n_used += k
    R = R[:, :n_used]
    return {"R": R, "exit_month": exit_month[:n_used], "exit_type": exit_type[:n_used],
            "alive_at_end": alive[:n_used], "n_firms": n_firms, "n_months": n_months}


def survivorship_datasets(u: dict) -> dict[str, np.ndarray]:
    R = u["R"]
    missing = R.copy()
    perf = np.flatnonzero(u["exit_type"] == 1)
    missing[u["exit_month"][perf], perf] = np.nan  # the final, delisting month is absent
    survivors = R[:, u["alive_at_end"]]
    return {"Full data (investable)": R, "Missing delisting returns": missing, "Survivors only": survivors}


def _equal_weight(R: np.ndarray) -> np.ndarray:
    return np.nanmean(R, axis=1)


def _vol_spread(R: np.ndarray, lookback: int = 12, quintile: float = 0.2) -> np.ndarray:
    """Monthly return of most-volatile minus least-volatile quintile (equal-weighted).

    Volatility is each stock's trailing `lookback`-month standard deviation,
    computed only from data available before the month being traded.
    """
    n_months = R.shape[0]
    out = np.full(n_months, np.nan)
    for t in range(lookback, n_months):
        hist = R[t - lookback:t]
        valid = (~np.isnan(hist)).sum(axis=0) == lookback
        valid &= ~np.isnan(R[t])
        if valid.sum() < 50:
            continue
        sd = hist[:, valid].std(axis=0, ddof=1)
        cols = np.flatnonzero(valid)
        order = np.argsort(sd)
        k = max(1, int(len(order) * quintile))
        out[t] = np.nanmean(R[t, cols[order[-k:]]]) - np.nanmean(R[t, cols[order[:k]]])
    return out


def part_b(seed: int = 0, n_firms: int = 2000, n_months: int = 360, delist_return: float = -0.30) -> dict:
    u = simulate_universe(n_firms=n_firms, n_months=n_months, seed=seed, delist_return=delist_return)
    sets = survivorship_datasets(u)
    rows = []
    for name, R in sets.items():
        ew = _equal_weight(R)
        spread = _vol_spread(R)
        s = spread[~np.isnan(spread)]
        t_stat = s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))
        rows.append({"dataset": name, "stocks_in_dataset": R.shape[1],
                     "ew_return": geometric_mean(ew), "ew_arith": ew.mean() * 12,
                     "vol_spread": s.mean() * 12, "vol_spread_t": t_stat})
    n_total = u["R"].shape[1]
    stats = {
        "stocks_ever_listed": int(n_total),
        "performance_delistings": int((u["exit_type"] == 1).sum()),
        "mergers": int((u["exit_type"] == 2).sum()),
        "survivors": int(u["alive_at_end"].sum()),
        "years": n_months / 12,
    }
    return {"table": pd.DataFrame(rows), "stats": stats}


# ---------------------------------------------------------------------------
# Part C: look-ahead bias with a moving-average rule on daily data
# ---------------------------------------------------------------------------


def _perf(ret: np.ndarray, rf: np.ndarray) -> dict:
    ex = ret - rf
    return {"cagr": float(np.exp(np.log1p(ret).mean() * TRADING_DAYS) - 1),
            "volatility": float(ret.std(ddof=1) * math.sqrt(TRADING_DAYS)),
            "sharpe": float(ex.mean() / ex.std(ddof=1) * math.sqrt(TRADING_DAYS)),
            "max_drawdown": max_drawdown(ret)}


def ma_signal(price: pd.Series, window: int) -> pd.Series:
    """1 when the price closes above its trailing moving average, else 0 (NaN during warm-up)."""
    ma = price.rolling(window).mean()
    return (price > ma).astype(float).where(ma.notna())


def part_c(daily: pd.DataFrame, window: int = 200,
           grid=(20, 30, 50, 75, 100, 150, 200, 250, 300), warmup_years: int = 10) -> dict:
    total = daily["Mkt-RF"] + daily["RF"]
    rf = daily["RF"]
    price = (1 + total).cumprod()

    sig = ma_signal(price, window)
    correct_pos = sig.shift(1)  # decided at yesterday's close, held today
    lookahead_pos = sig  # decided at today's close, but credited with today's return
    keep = correct_pos.notna() & lookahead_pos.notna()
    strat = {}
    for name, pos in (("Buy and hold", pd.Series(1.0, index=total.index)),
                      (f"{window}-day rule, traded correctly", correct_pos),
                      (f"{window}-day rule, with look-ahead", lookahead_pos)):
        r = (pos * total + (1 - pos) * rf)[keep]
        strat[name] = r
    perf = pd.DataFrame({k: _perf(v.to_numpy(), rf[keep].to_numpy()) for k, v in strat.items()}).T
    days_switched = float((correct_pos[keep].diff().abs() > 0).mean() * TRADING_DAYS)

    # Hindsight versus walk-forward choice of the window, both correctly lagged.
    rets = {}
    for w in grid:
        pos = ma_signal(price, w).shift(1)
        rets[w] = pos * total + (1 - pos) * rf
    rets = pd.DataFrame(rets)
    start = rets.dropna().index[0] + pd.DateOffset(years=warmup_years)
    years = sorted({d.year for d in rets.index if d >= start})
    eval_mask = rets.index >= pd.Timestamp(year=years[0], month=1, day=1)
    ex = rets.sub(rf, axis=0)
    full_sharpe = (ex[eval_mask].mean() / ex[eval_mask].std() * math.sqrt(TRADING_DAYS))
    hindsight_w = int(full_sharpe.idxmax())

    wf = pd.Series(np.nan, index=rets.index)
    chosen = []
    for y in years:
        past = ex[(ex.index < pd.Timestamp(year=y, month=1, day=1))].dropna()
        sr = past.mean() / past.std()
        w = int(sr.idxmax())
        chosen.append({"year": y, "window": w})
        this = rets.index.year == y
        wf[this] = rets.loc[this, w]
    wf_ex = (wf - rf)[eval_mask].dropna()
    walk_forward_sharpe = float(wf_ex.mean() / wf_ex.std() * math.sqrt(TRADING_DAYS))
    return {
        "window": window, "performance": perf, "series": strat, "switches_per_year": days_switched,
        "grid_sharpe": full_sharpe, "hindsight_window": hindsight_w,
        "hindsight_sharpe": float(full_sharpe.max()), "walk_forward_sharpe": walk_forward_sharpe,
        "chosen_windows": pd.DataFrame(chosen),
        "eval_start": str(rets.index[eval_mask][0].date()), "eval_end": str(rets.index[-1].date()),
        "sample_start": str(total.index[0].date()),
    }


# ---------------------------------------------------------------------------
# Part D: micro-caps, weighting and the size premium
# ---------------------------------------------------------------------------


def _premium(series: pd.Series) -> tuple[float, float]:
    s = series.dropna().to_numpy()
    res = ols_newey_west(s, np.zeros((len(s), 0)))
    return float(s.mean() * 12), float(res["t"][0, 0])


def part_d(vw: pd.DataFrame, ew: pd.DataFrame, split: str = "1980-12") -> dict:
    split_p = pd.Period(split, freq="M")
    periods = {
        f"{vw.index[0]} to {split}": lambda idx: idx <= split_p,
        f"{split_p + 1} to {vw.index[-1]}": lambda idx: idx > split_p,
        "Full sample": lambda idx: np.ones(len(idx), dtype=bool),
    }
    rows = []
    for weighting, df in (("Value-weighted", vw), ("Equal-weighted", ew)):
        for label, cols in (("Smallest minus largest (D1 - D10)", ("D1", "D10")),
                            ("Without the smallest decile (D2 - D10)", ("D2", "D10"))):
            spread = df[cols[0]] - df[cols[1]]
            row = {"weighting": weighting, "comparison": label}
            for p_label, sel in periods.items():
                mean, t = _premium(spread[sel(spread.index)])
                row[f"{p_label} mean"] = mean
                row[f"{p_label} t"] = t
            rows.append(row)
    deciles = pd.DataFrame({"decile": [f"D{i}" for i in range(1, 11)],
                            "value_weighted": vw.mean().to_numpy() * 12,
                            "equal_weighted": ew.mean().to_numpy() * 12})
    return {"table": pd.DataFrame(rows), "deciles": deciles, "periods": list(periods), "split": split}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(simulated: bool, text: str) -> str:
    return text + ("  ·  SIMULATED DATA: tests the code only" if simulated else "")


def fig_averages(a: pd.DataFrame, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure(9, 4.2)
    y = np.arange(len(a))[::-1]
    for yi, (_, r) in zip(y, a.iterrows()):
        ax.plot([r.geometric, r.arithmetic], [yi, yi], color=plots.INK_3, linewidth=2, zorder=1)
        ax.plot(r.geometric, yi, "o", color=plots.BLUE, markersize=9, markeredgecolor=plots.SURFACE,
                markeredgewidth=2, zorder=2)
        ax.plot(r.arithmetic, yi, "o", color=plots.ORANGE, markersize=9, markeredgecolor=plots.SURFACE,
                markeredgewidth=2, zorder=2)
        ax.text(r.arithmetic + 0.006, yi, f"gap {_pct(r.gap)}", va="center", fontsize=9, color=plots.INK)
    ax.set_yticks(y, a["series"], fontsize=9, color=plots.INK)
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Annual return")
    ax.grid(axis="x", color=plots.GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=plots.BLUE, label="geometric (what a buy-and-hold investor earned)")
    ax.plot([], [], "o", color=plots.ORANGE, label="arithmetic (average of yearly returns)")
    ax.legend(frameon=False, fontsize=9, labelcolor=plots.INK_2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, borderaxespad=0.3, handletextpad=0.3)
    ax.set_xlim(0, max(a["arithmetic"]) * 1.25)
    plots.titles(fig, "Lab Figure 2.1  The more volatile the asset, the more the average overstates growth",
                 _sub(simulated, "Annualized arithmetic and geometric average returns, value-weighted portfolios"))
    plots.finish(fig, out / "fig2_1_averages.png", top=0.84)
    plt.close(fig)


def fig_survivorship(b: dict, out: Path, simulated_note: str):
    t = b["table"]
    fig, axes_ = plots.new_figure(10, 4.4)[0], None
    import matplotlib.pyplot as plt
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fig.patch.set_facecolor(plots.SURFACE)
    colors = [plots.BLUE, plots.YELLOW, plots.RED]
    for ax, col, title, fmt in ((axes[0], "ew_return", "Equal-weighted market, compound return", "{:.1%}"),
                                (axes[1], "vol_spread", "Most minus least volatile stocks", "{:+.1%}")):
        plots.style_axes(ax)
        vals = t[col].to_numpy()
        x = np.arange(len(t))
        ax.bar(x, vals, color=colors, width=0.62)
        ax.axhline(0, color=plots.INK_3, linewidth=1)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.004 if v >= 0 else -0.004), fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=plots.INK)
        ax.set_xticks(x, [s.replace(" (", "\n(") for s in t["dataset"]], fontsize=8.5, color=plots.INK)
        plots.percent_axis(ax)
        ax.set_title(title + " (per year)", loc="left", fontsize=10, color=plots.INK, pad=8)
        lo, hi = min(0, vals.min()), max(0, vals.max())
        pad = (hi - lo) * 0.18 or 0.01
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    st = b["stats"]
    plots.titles(fig, "Lab Figure 2.2  Survivorship bias in a simulated market where the truth is known",
                 f"{st['stocks_ever_listed']:,} stocks over {st['years']:.0f} years; {st['performance_delistings']:,} delisted "
                 f"for poor performance · no stock has an expected-return advantage{simulated_note}")
    plots.finish(fig, out / "fig2_2_survivorship.png")
    plt.close(fig)


def fig_lookahead(c: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    colors = [plots.INK_3, plots.BLUE, plots.RED]
    ends = []
    for (name, r), color in zip(c["series"].items(), colors):
        wealth = (1 + r).cumprod()
        ax.plot(wealth.index, wealth.values, color=color, linewidth=1.8 if color != plots.RED else 2)
        ends.append([math.log10(wealth.iloc[-1]), name, wealth.index[-1]])
    ax.set_yscale("log")
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.25)
    for y_log, name, x_end in ends:
        ax.text(x_end, 10 ** y_log, "  " + name, fontsize=9, color=plots.INK, va="center")
    first = next(iter(c["series"].values())).index
    ax.set_xlim(first[0], first[-1] + (first[-1] - first[0]) * 0.27)
    ax.set_ylabel("Growth of $1 (log scale)")
    p = c["performance"]
    la = [k for k in p.index if "look-ahead" in k][0]
    ok = [k for k in p.index if "correctly" in k][0]
    plots.titles(fig, "Lab Figure 2.3  One day of look-ahead turns a modest rule into a spectacular one",
                 _sub(simulated, f"Compound return: traded correctly {p.loc[ok, 'cagr']:.1%} a year, with look-ahead "
                                 f"{p.loc[la, 'cagr']:.1%} · daily U.S. market data from {c['sample_start']}"))
    plots.finish(fig, out / "fig2_3_lookahead.png")
    plt.close(fig)


def fig_hindsight(c: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    g = c["grid_sharpe"]
    x = np.arange(len(g))
    colors = [plots.ORANGE if w == c["hindsight_window"] else plots.BLUE for w in g.index]
    ax.bar(x, g.values, color=colors, width=0.62)
    ax.axhline(c["walk_forward_sharpe"], color=plots.INK, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(len(g) - 0.5, c["walk_forward_sharpe"],
            f"walk-forward choice: {c['walk_forward_sharpe']:.2f}", ha="right", va="bottom", fontsize=9,
            color=plots.INK, bbox={"facecolor": plots.SURFACE, "edgecolor": "none", "pad": 1})
    best = int(np.argmax(g.values))
    ax.text(best, g.values[best], f"hindsight pick\n{g.values[best]:.2f}", ha="center", va="bottom", fontsize=9,
            color=plots.INK)
    ax.set_xticks(x, [f"{w}" for w in g.index])
    ax.set_xlabel("Moving-average window (trading days)")
    ax.set_ylabel("Sharpe ratio over the evaluation period")
    ax.set_ylim(min(0, g.min()) * 1.2, max(g.max(), c["walk_forward_sharpe"]) * 1.3)
    plots.titles(fig, "Lab Figure 2.4  Choosing a rule's settings with hindsight is look-ahead too",
                 _sub(simulated, f"Evaluation period {c['eval_start']} to {c['eval_end']} · orange bar: best window "
                                 "chosen after seeing the results · dashed line: window chosen each year from past data only"))
    plots.finish(fig, out / "fig2_4_hindsight.png")
    plt.close(fig)


def fig_size(d: dict, out: Path, simulated: bool):
    fig, ax, plt = plots.new_figure()
    dec = d["deciles"]
    x = np.arange(10)
    ax.bar(x - 0.19, dec["value_weighted"], width=0.36, color=plots.BLUE, label="value-weighted")
    ax.bar(x + 0.19, dec["equal_weighted"], width=0.36, color=plots.ORANGE, label="equal-weighted")
    ax.set_xticks(x, ["D1\nsmallest"] + [f"D{i}" for i in range(2, 10)] + ["D10\nlargest"])
    plots.percent_axis(ax)
    ax.set_ylabel("Average annual return (arithmetic)")
    ax.legend(frameon=False, fontsize=9, labelcolor=plots.INK_2, loc="upper right")
    plots.titles(fig, "Lab Figure 2.5  Where the size premium lives",
                 _sub(simulated, "Average return by size decile, full sample · the gap is largest in the smallest "
                                 "decile and in equal-weighted portfolios"))
    plots.finish(fig, out / "fig2_5_size_deciles.png")
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


def _pct(v):
    return f"{v:.1%}".replace("-0.0%", "0.0%")


def _spct(v):
    return f"{v:+.1%}"


def _t(v):
    return f"{v:.2f}"


def write_report(out: Path, a, b, c, d, simulated: bool, args):
    md = ["# Lab 2 results: Four Data Traps", ""]
    if simulated:
        md += ["> **Simulated data.** Parts A, C and D used simulated returns, so their numbers test the code "
               "only. Run without `--source sim` for real results. (Part B is always a simulation, by design.)", ""]
    md += ["**Claim on trial:** the data I am testing on represents what an investor could actually have done.", ""]

    md += ["## Part A: Arithmetic versus geometric averages", "",
           _md(a.rename(columns={"series": "Series", "arithmetic": "Arithmetic", "geometric": "Geometric",
                                 "gap": "Gap", "volatility": "Volatility", "approx_gap": "Approx. gap (σ²/2)"}),
               {"Arithmetic": _pct, "Geometric": _pct, "Gap": _pct, "Volatility": _pct,
                "Approx. gap (σ²/2)": _pct}), "",
           "![Lab Figure 2.1](fig2_1_averages.png)", ""]

    st = b["stats"]
    md += ["## Part B: Survivorship (simulated market, truth known)", "",
           f"- {st['stocks_ever_listed']:,} stocks listed over {st['years']:.0f} years; "
           f"{st['performance_delistings']:,} delisted for poor performance (delisting return {args.delist_return:.0%}), "
           f"{st['mergers']:,} acquired, {st['survivors']:,} still listed at the end.",
           "- Every stock has the same expected return, so the investable dataset is the benchmark for truth. "
           "Any premium for volatile stocks there is produced by delisting losses, not by skill.", "",
           _md(b["table"].rename(columns={"dataset": "Dataset", "stocks_in_dataset": "Stocks",
                                          "ew_return": "Equal-weighted compound return", "ew_arith": "Arithmetic",
                                          "vol_spread": "Volatile minus calm (per year)", "vol_spread_t": "t-statistic"}),
               {"Equal-weighted compound return": _pct, "Arithmetic": _pct,
                "Volatile minus calm (per year)": _spct, "t-statistic": _t, "Stocks": lambda v: f"{int(v):,}"}), "",
           "![Lab Figure 2.2](fig2_2_survivorship.png)", ""]

    p = c["performance"].reset_index().rename(columns={"index": "Strategy"})
    md += ["## Part C: Look-ahead bias", "",
           f"**C1. Trading on the same day's close.** {c['window']}-day moving-average rule on the U.S. market, "
           f"from {c['sample_start']}. The correct version switches about {c['switches_per_year']:.0f} times a year.", "",
           _md(p.rename(columns={"cagr": "Compound return", "volatility": "Volatility", "sharpe": "Sharpe ratio",
                                 "max_drawdown": "Maximum drawdown"}),
               {"Compound return": _pct, "Volatility": _pct, "Sharpe ratio": _t, "Maximum drawdown": _pct}), "",
           "![Lab Figure 2.3](fig2_3_lookahead.png)", "",
           f"**C2. Choosing the window with hindsight.** Evaluation period {c['eval_start']} to {c['eval_end']}.", "",
           f"- Window chosen with hindsight: {c['hindsight_window']} days, Sharpe ratio {c['hindsight_sharpe']:.2f}.",
           f"- Window chosen each January from past data only (walk-forward): Sharpe ratio {c['walk_forward_sharpe']:.2f}.",
           f"- The walk-forward rule changed its window {int((c['chosen_windows']['window'].diff() != 0).sum() - 1)} "
           "times; see table_chosen_windows.csv.", "",
           "![Lab Figure 2.4](fig2_4_hindsight.png)", ""]

    t = d["table"].copy()
    cols = ["weighting", "comparison"] + [col for col in t.columns if col not in ("weighting", "comparison")]
    fmt = {col: (_spct if col.endswith("mean") else _t) for col in cols[2:]}
    md += ["## Part D: Micro-caps, weighting and the size premium", "",
           f"Annualized average return of the spread, with Newey–West t-statistics. Banz published the size effect in 1981.", "",
           _md(t[cols].rename(columns={"weighting": "Weighting", "comparison": "Comparison"}), fmt), "",
           "![Lab Figure 2.5](fig2_5_size_deciles.png)", ""]

    md += ["## Your verdict", "",
           "For each trap, write one sentence: how much did it change the result, and in which direction? "
           "Then answer the claim: does a typical free dataset represent what an investor could have done? "
           "What would you check first in someone else's backtest?", ""]
    (out / "lab02_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--monthly-file", default=None, help="Fama/French 3 Factors (monthly) zip or CSV")
    ap.add_argument("--daily-file", default=None, help="Fama/French 3 Factors [Daily] zip or CSV")
    ap.add_argument("--size-file", default=None, help="Portfolios Formed on Size zip or CSV")
    ap.add_argument("--ma-window", type=int, default=200)
    ap.add_argument("--publication-split", default="1980-12", help="last month before the size effect was published")
    ap.add_argument("--sim-firms", type=int, default=2000, help="Part B: stocks in the simulated market")
    ap.add_argument("--sim-years", type=int, default=30, help="Part B: years in the simulated market")
    ap.add_argument("--delist-return", type=float, default=-0.30,
                    help="Part B: final return of a stock delisted for poor performance (default -0.30)")
    ap.add_argument("--seed", type=int, default=2026, help="seed for the Part B simulation")
    ap.add_argument("--sim-seed", type=int, default=0, help="seed for simulated market data (--source sim)")
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        data = load_all(args.source, args.monthly_file, args.daily_file, args.size_file, args.sim_seed)
    except DataUnavailable as e:
        raise SystemExit(str(e)) from None
    sim = data["simulated"]

    a = part_a(data)
    b = part_b(seed=args.seed, n_firms=args.sim_firms, n_months=args.sim_years * 12,
               delist_return=args.delist_return)
    c = part_c(data["daily"], window=args.ma_window)
    d = part_d(data["size_vw"], data["size_ew"], split=args.publication_split)

    fig_averages(a, out, sim)
    fig_survivorship(b, out, "")
    fig_lookahead(c, out, sim)
    fig_hindsight(c, out, sim)
    fig_size(d, out, sim)

    a.to_csv(out / "table_averages.csv", index=False)
    b["table"].to_csv(out / "table_survivorship.csv", index=False)
    c["performance"].to_csv(out / "table_lookahead.csv")
    c["grid_sharpe"].rename("sharpe").to_csv(out / "table_window_sharpe.csv", index_label="window")
    c["chosen_windows"].to_csv(out / "table_chosen_windows.csv", index=False)
    d["table"].to_csv(out / "table_size_premium.csv", index=False)
    d["deciles"].to_csv(out / "table_size_deciles.csv", index=False)
    write_report(out, a, b, c, d, sim, args)

    perf = c["performance"]
    summary = {
        "simulated": sim,
        "averages": a.set_index("series")[["arithmetic", "geometric", "gap"]].to_dict(orient="index"),
        "survivorship": b["table"].set_index("dataset").to_dict(orient="index"),
        "survivorship_stats": b["stats"],
        "lookahead": perf.to_dict(orient="index"),
        "hindsight_window": c["hindsight_window"], "hindsight_sharpe": c["hindsight_sharpe"],
        "walk_forward_sharpe": c["walk_forward_sharpe"],
        "size_premium": d["table"].to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    la = [k for k in perf.index if "look-ahead" in k][0]
    ok = [k for k in perf.index if "correctly" in k][0]
    bt = b["table"].set_index("dataset")
    print(f"A  Smallest-decile gap between arithmetic and geometric averages: "
          f"{a.set_index('series').loc['Smallest 10% of stocks', 'gap']:.1%} a year")
    print(f"B  Volatile-minus-calm: investable {bt['vol_spread'].iloc[0]:+.1%}, survivors only "
          f"{bt['vol_spread'].iloc[2]:+.1%} a year")
    print(f"C  {args.ma_window}-day rule: correct {perf.loc[ok, 'cagr']:.1%}, look-ahead {perf.loc[la, 'cagr']:.1%} a year; "
          f"hindsight Sharpe {c['hindsight_sharpe']:.2f} vs walk-forward {c['walk_forward_sharpe']:.2f}")
    print(f"D  Size premium table written to {out / 'table_size_premium.csv'}")
    print(f"Report: {out / 'lab02_report.md'}")
    if sim:
        print("NOTE: simulated data for Parts A, C and D. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
