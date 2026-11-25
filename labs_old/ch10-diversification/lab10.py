"""Lab 10: Does Diversification Fail When You Need It Most?

Chapter 10 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "Correlations rise in crises, so diversification fails exactly when
   investors need it most."

WHAT THE LAB DOES
  Part A  Correlations of US stocks with international stocks, Treasury bonds
          and oil, and the average correlation among ten industries: in calm
          and high-volatility months, raw and corrected for the volatility
          bias (Forbes-Rigobon), and exceedance correlations against what a
          normal distribution would produce (Longin-Solnik, Ang-Chen).
  Part B  Tail dependence: how often assets fall into their worst decile
          together, against a normal benchmark.
  Part C  The stock-bond correlation over time and its link to inflation;
          the 2022 sell-off.
  Part D  The investor's view: what 60/40 and a multi-asset mix lost in the
          worst stock months, and in 2008 and 2022.

DATA (free)
  Kenneth R. French Data Library: Fama/French 3 Factors (US market and bill
  rate), Developed ex US 3 Factors (international market), 10 Industry
  Portfolios. FRED: GS10 (10-year Treasury yield), CPIAUCSL (consumer
  prices), WTISPLC (spot crude oil price).

USAGE
  python lab10.py
  python lab10.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.corr import (avg_pairwise_corr, bond_returns_from_yields, diversification_ratio, exceedance_curve,
                          forbes_rigobon, normal_exceedance, normal_tail_dependence, rolling_corr, simulated_assets,
                          tail_dependence)
from fintest.credit import monthly_fred
from fintest.data import DataUnavailable, find_table, load_french_factors, load_french_tables

ROOT = Path(__file__).resolve().parent
OTHERS = ["Intl stocks", "Bonds", "Oil"]
COLORS = {"Intl stocks": plots.BLUE, "Bonds": plots.GREEN, "Oil": plots.ORANGE, "Industries (avg)": plots.VIOLET}
QS = np.r_[np.linspace(0.05, 0.45, 9), np.linspace(0.55, 0.95, 9)]


def _col(df, name):
    return df[[c for c in df.columns if c.strip().upper() == name][0]]


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        return simulated_assets(seed=seed)
    ff = load_french_factors("ff3_monthly", path=files.get("ff3"))
    rf = _col(ff, "RF")
    assets = pd.DataFrame({"US stocks": _col(ff, "MKT-RF")})
    missing = []
    try:
        intl = load_french_factors("developed_ex_us_ff3", path=files.get("intl"))
        assets["Intl stocks"] = _col(intl, "MKT-RF")
    except (DataUnavailable, IndexError) as e:
        missing.append(f"Intl stocks ({str(e).splitlines()[0]})")
    infl = None
    for key, series in (("gs10", "GS10"), ("cpi", "CPIAUCSL"), ("oil", "WTISPLC")):
        try:
            s = monthly_fred(series, files.get(key))
        except DataUnavailable as e:
            missing.append(f"{series} ({str(e).splitlines()[0]})")
            continue
        if series == "GS10":
            assets["Bonds"] = bond_returns_from_yields(s, 10.0) - rf.reindex(s.index)
        elif series == "CPIAUCSL":
            infl = (s / s.shift(12) - 1).rename("inflation")
        else:
            assets["Oil"] = s.pct_change() - rf.reindex(s.index)
    try:
        t = load_french_tables("industry_10", path=files.get("ind10"))
        ind = find_table(t, "value", "weight").sub(rf, axis=0).dropna()
    except (DataUnavailable, KeyError) as e:
        missing.append(f"10 industries ({str(e).splitlines()[0]})")
        ind = None
    assets = assets.dropna(subset=["US stocks"])
    return {"assets": assets, "industries": ind, "inflation": infl, "simulated": False, "missing": missing}


def high_vol_months(us: pd.Series, window: int = 12, top: float = 0.2) -> pd.Series:
    """Months whose trailing realized volatility of US stocks (including the month) is in the top `top` share."""
    rv = us.rolling(window, min_periods=window).std()
    return (rv >= rv.quantile(1 - top)).reindex(us.index).fillna(False)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part_a(d: dict) -> dict:
    A = d["assets"]
    us = A["US stocks"]
    hv = high_vol_months(us)
    crash = us <= us.quantile(0.10)
    rows = []
    pairs = [(o, A[o]) for o in OTHERS if o in A]
    for name, x in pairs:
        df = pd.concat([us, x], axis=1).dropna()
        u, v = df.iloc[:, 0], df.iloc[:, 1]
        h, c = hv.reindex(df.index), crash.reindex(df.index)
        rho_all = u.corr(v)
        rho_calm, rho_high = u[~h].corr(v[~h]), u[h].corr(v[h])
        adj = float(forbes_rigobon(rho_high, u[h].var(), u[~h].var()))
        rows.append({"pair": f"US stocks vs {name}", "start": str(df.index[0]), "months": len(df), "all": rho_all,
                     "calm": rho_calm, "high_vol": rho_high, "high_vol_adjusted": adj,
                     "worst_10pct_us_months": u[c].corr(v[c]), "n_high_vol": int(h.sum())})
    ind = d.get("industries")
    if ind is not None:
        I = ind.dropna()
        h = hv.reindex(I.index).fillna(False)
        mk = us.reindex(I.index)
        calm, high = avg_pairwise_corr(I[~h]), avg_pairwise_corr(I[h])
        adj = float(forbes_rigobon(high, mk[h].var(), mk[~h].var()))
        rows.append({"pair": "Industries (avg pairwise)", "start": str(I.index[0]), "months": len(I),
                     "all": avg_pairwise_corr(I), "calm": calm, "high_vol": high, "high_vol_adjusted": adj,
                     "worst_10pct_us_months": avg_pairwise_corr(I[(mk <= mk.quantile(0.1)).to_numpy()]),
                     "n_high_vol": int(h.sum())})
    table = pd.DataFrame(rows)
    curves = {}
    for name, x in pairs:
        df = pd.concat([us, x], axis=1).dropna()
        rho = df.iloc[:, 0].corr(df.iloc[:, 1])
        curves[name] = {"emp": exceedance_curve(df.iloc[:, 0], df.iloc[:, 1], QS),
                        "normal": normal_exceedance(rho, QS), "rho": rho}
    return {"table": table, "curves": curves, "high_vol": hv}


def part_b(d: dict, ps=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    A = d["assets"]
    rows = []
    for o in [c for c in OTHERS if c in A]:
        df = A[["US stocks", o]].dropna()
        rho = df.corr().iat[0, 1]
        for p in ps:
            rows.append({"pair": f"US stocks vs {o}", "tail": p, "empirical": tail_dependence(df.iloc[:, 0], df.iloc[:, 1], p),
                         "normal_same_corr": normal_tail_dependence(rho, p), "independent": p})
    t = pd.DataFrame(rows)
    t["excess"] = t["empirical"] - t["normal_same_corr"]
    return t


def part_c(d: dict) -> dict:
    A = d["assets"]
    if "Bonds" not in A:
        return {}
    df = A[["US stocks", "Bonds"]].dropna()
    rc = rolling_corr(df["US stocks"], df["Bonds"], 36).rename("corr36")
    out = {"rolling": rc}
    infl = d.get("inflation")
    if infl is not None:
        inf = infl.reindex(df.index)
        z = pd.concat([rc, inf.rolling(36, min_periods=36).mean().rename("infl36")], axis=1).dropna()
        x = np.c_[np.ones(len(z)), z["infl36"]]
        b = np.linalg.lstsq(x, z["corr36"].to_numpy(), rcond=None)[0]
        regimes = []
        for label, lo, hi in (("Inflation below 3%", -1, 0.03), ("Inflation 3% or more", 0.03, 1)):
            m = (inf.shift(1) >= lo) & (inf.shift(1) < hi)
            sub = df[m.reindex(df.index).fillna(False)]
            regimes.append({"regime": label + " (prior 12 months)", "months": len(sub),
                            "stock_bond_corr": sub.corr().iat[0, 1],
                            "bond_return_in_worst_10pct_stock_months":
                                sub["Bonds"][sub["US stocks"] <= sub["US stocks"].quantile(0.1)].mean()})
        out["regimes"] = pd.DataFrame(regimes)
        out["infl_fit"] = {"intercept": float(b[0]), "slope_per_1pt_inflation": float(b[1] / 100),
                           "months": len(z), "note": "overlapping 36-month windows; descriptive only"}
        out["inflation"] = inf
    decades = []
    for start in range(int(str(df.index[0])[:4]) // 10 * 10, 2030, 10):
        sub = df[(df.index.year >= start) & (df.index.year < start + 10)]
        if len(sub) >= 24:
            decades.append({"decade": f"{start}s", "stock_bond_corr": sub.corr().iat[0, 1],
                            "both_negative_share": float(((sub["US stocks"] < 0) & (sub["Bonds"] < 0)).mean())})
    out["decades"] = pd.DataFrame(decades)
    y22 = df[df.index.year == 2022]
    if len(y22):
        out["y2022"] = {"stocks": float(np.prod(1 + y22["US stocks"]) - 1), "bonds": float(np.prod(1 + y22["Bonds"]) - 1),
                        "months_both_negative": int(((y22["US stocks"] < 0) & (y22["Bonds"] < 0)).sum()),
                        "corr": float(y22.corr().iat[0, 1])}
    return out


def portfolios(A: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame(index=A.index)
    p["100% US stocks"] = A["US stocks"]
    if "Bonds" in A:
        p["60/40 stocks/bonds"] = 0.6 * A["US stocks"] + 0.4 * A["Bonds"]
    cols = [c for c in ["US stocks", "Intl stocks", "Bonds", "Oil"] if c in A]
    if len(cols) > 2:
        p["Equal-weight multi-asset"] = A[cols].mean(axis=1, skipna=False)
    return p


def part_d(d: dict) -> dict:
    A = d["assets"]
    P = portfolios(A)
    rows = []
    for name in P.columns:
        df = pd.concat([P[name], A["US stocks"]], axis=1, keys=["r", "us"]).dropna()
        r, us = df["r"], df["us"]
        wealth = (1 + r).cumprod()
        worst, best = us <= us.quantile(0.05), us >= us.quantile(0.95)
        rows.append({"portfolio": name, "start": str(df.index[0]), "mean_annual": 12 * r.mean(),
                     "vol_annual": r.std() * math.sqrt(12), "sharpe": r.mean() / r.std() * math.sqrt(12),
                     "worst_5pct_stock_months_avg": r[worst].mean(), "stocks_in_same_months": us[worst].mean(),
                     "best_5pct_stock_months_avg": r[best].mean(),
                     "max_drawdown": float((wealth / wealth.cummax() - 1).min())})
    crisis = []
    for label, a, b in (("1973–74", "1973-01", "1974-09"), ("2000–02", "2000-09", "2002-09"),
                        ("2008 (Nov 2007–Feb 2009)", "2007-11", "2009-02"), ("2020 (Feb–Mar)", "2020-02", "2020-03"),
                        ("2022", "2022-01", "2022-12")):
        row = {"period": label}
        for name in P.columns:
            x = P[name].loc[pd.Period(a, "M"):pd.Period(b, "M")]
            row[name] = float(np.prod(1 + x) - 1) if len(x) and x.notna().all() else float("nan")
        crisis.append(row)
    dr = {}
    if "Bonds" in A:
        df = A[["US stocks", "Bonds"]].dropna()
        hv = high_vol_months(A["US stocks"]).reindex(df.index).fillna(False)
        dr = {"calm": diversification_ratio(df[~hv], [0.6, 0.4]), "high_vol": diversification_ratio(df[hv], [0.6, 0.4])}
    return {"table": pd.DataFrame(rows), "crisis": pd.DataFrame(crisis), "div_ratio_6040": dr, "portfolios": P}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_exceedance(a, out, sim):
    import matplotlib.pyplot as plt

    names = list(a["curves"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names) + 1, 4.4), sharey=True)
    fig.patch.set_facecolor(plots.SURFACE)
    for ax, n in zip(np.atleast_1d(axes), names):
        plots.style_axes(ax)
        c = a["curves"][n]
        lo, hi = QS < 0.5, QS > 0.5
        for m in (lo, hi):
            ax.plot(QS[m], c["emp"][m], marker="o", color=COLORS[n], linewidth=1.6, markersize=4)
            ax.plot(QS[m], c["normal"][m], color=plots.INK_3, linestyle="--", linewidth=1.2)
        ax.axvline(0.5, color=plots.GRID, linewidth=0.8)
        ax.set_title(f"US stocks vs {n}", loc="left", fontsize=9.5, color=plots.INK)
        ax.set_xlabel("Quantile threshold (both below ← → both above)")
    np.atleast_1d(axes)[0].set_ylabel("Exceedance correlation")
    plots.titles(fig, "Lab Figure 10.1  Correlation in the tails",
                 _sub(sim, "Solid: correlation over months when both returns are beyond the threshold · dashed: what a "
                           "normal distribution with the same overall correlation gives"))
    plots.finish(fig, out / "fig10_1_exceedance.png")
    plt.close(fig)


def fig_conditional(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = a["table"]
    x = np.arange(len(t))
    w = 0.26
    ax.bar(x - w, t["calm"], w, color=plots.INK_3, label="Calm months")
    ax.bar(x, t["high_vol"], w, color=plots.RED, label="High-volatility months (raw)")
    ax.bar(x + w, t["high_vol_adjusted"], w, color=plots.BLUE, label="High-volatility, Forbes–Rigobon adjusted")
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_xticks(x, [p.replace("US stocks vs ", "vs ") for p in t["pair"]], fontsize=9)
    ax.set_ylabel("Correlation with US stocks")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(min(0, float(t[["calm", "high_vol", "high_vol_adjusted"]].min().min()) - 0.05),
                float(t[["calm", "high_vol", "high_vol_adjusted"]].max().max()) + 0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    plots.titles(fig, "Lab Figure 10.2  Correlations in calm and turbulent markets",
                 _sub(sim, "High volatility: top 20% of months by trailing 12-month US stock volatility · industries: "
                           "average pairwise correlation"))
    plots.finish(fig, out / "fig10_2_conditional.png")
    plt.close(fig)


def fig_stock_bond(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    rc = c["rolling"].dropna()
    dates = rc.index.to_timestamp()
    ax.plot(dates, rc, color=plots.GREEN, linewidth=1.5, label="Stock–bond correlation (36 months)")
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    if "inflation" in c:
        ax2 = ax.twinx()
        inf = c["inflation"].reindex(rc.index)
        ax2.plot(dates, inf, color=plots.ORANGE, linewidth=1.0, alpha=0.8, label="Inflation, 12 months (right)")
        ax2.set_ylabel("Inflation")
        plots.percent_axis(ax2)
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower left")
    else:
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    plots.titles(fig, "Lab Figure 10.3  The stock–bond correlation over time",
                 _sub(sim, "US stock and 10-year Treasury excess returns, rolling 36-month correlation"))
    plots.finish(fig, out / "fig10_3_stock_bond.png")
    plt.close(fig)


def fig_worst(dd, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    t = dd["table"]
    x = np.arange(len(t))
    ax.bar(x - 0.18, t["worst_5pct_stock_months_avg"], 0.36, color=plots.RED, label="Worst 5% of stock months")
    ax.bar(x + 0.18, t["best_5pct_stock_months_avg"], 0.36, color=plots.BLUE, label="Best 5% of stock months")
    for i, row in t.iterrows():
        ax.text(i - 0.18, row.worst_5pct_stock_months_avg - 0.004, f"{row.worst_5pct_stock_months_avg:.1%}", ha="center",
                va="top", fontsize=8.5, color=plots.INK_2)
        ax.text(i + 0.18, row.best_5pct_stock_months_avg + 0.003, f"{row.best_5pct_stock_months_avg:.1%}", ha="center",
                va="bottom", fontsize=8.5, color=plots.INK_2)
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_xticks(x, t["portfolio"], fontsize=9)
    plots.percent_axis(ax)
    lo = t["worst_5pct_stock_months_avg"].min()
    ax.set_ylim(lo * 1.3, t["best_5pct_stock_months_avg"].max() * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Average monthly excess return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    plots.titles(fig, "Lab Figure 10.4  Does diversification still cushion the worst months?",
                 _sub(sim, "Months ranked by US stock returns within each portfolio's own sample · first month: "
                           + ", ".join(f"{p} {s}" for p, s in zip(t["portfolio"], t["start"]))))
    plots.finish(fig, out / "fig10_4_worst_months.png")
    plt.close(fig)


def fig_crises(dd, out, sim):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    P = dd["portfolios"]
    periods = [("2007-10", "2009-12", "2007–09"), ("2021-12", "2023-12", "2022")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    fig.patch.set_facecolor(plots.SURFACE)
    cols = {"100% US stocks": plots.INK, "60/40 stocks/bonds": plots.GREEN, "Equal-weight multi-asset": plots.VIOLET}
    for ax, (a, b, label) in zip(axes, periods):
        plots.style_axes(ax)
        sub = P.loc[pd.Period(a, "M"):pd.Period(b, "M")]
        for n in P.columns:
            if sub[n].notna().all() and len(sub):
                w = (1 + sub[n]).cumprod() - 1
                ax.plot(sub.index.to_timestamp(), w, color=cols.get(n, plots.BLUE), linewidth=1.6, label=n)
        ax.axhline(0, color=plots.INK_3, linewidth=0.8)
        ax.set_title(label, loc="left", fontsize=9.5, color=plots.INK)
        plots.percent_axis(ax)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Cumulative excess return")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3)

    plots.titles(fig, "Lab Figure 10.5  Two crises, two answers?",
                 _sub(sim, "Cumulative excess return over Treasury bills, monthly rebalanced"))
    plots.finish(fig, out / "fig10_5_crises.png", bottom=0.07)
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


def write_report(out, a, b, c, dd, sim, missing):
    f2, pc = _num("+.2f"), _num("+.1%")
    md = ["# Lab 10 results: Does Diversification Fail When You Need It Most?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    if missing:
        md += ["Not loaded: " + "; ".join(missing), ""]
    md += ["**Claim on trial:** correlations rise in crises, so diversification fails when investors need it most.", "",
           "## Part A: Conditional correlations", "",
           "High volatility: top 20% of months by trailing 12-month US stock volatility. Adjusted: Forbes–Rigobon "
           "correction for the higher variance of US stocks in those months.", "",
           _md(a["table"], {k: f2 for k in ("all", "calm", "high_vol", "high_vol_adjusted", "worst_10pct_us_months")}), "",
           "![Lab Figure 10.1](fig10_1_exceedance.png)", "", "![Lab Figure 10.2](fig10_2_conditional.png)", "",
           "## Part B: Tail dependence", "",
           "P(other asset in its worst p-tail | US stocks in their worst p-tail), against a normal distribution with the "
           "same correlation.", "",
           _md(b, {"tail": _num(".0%"), "empirical": _num(".2f"), "normal_same_corr": _num(".2f"),
                   "independent": _num(".2f"), "excess": f2}), ""]
    if c:
        md += ["## Part C: Stocks and bonds", "", "![Lab Figure 10.3](fig10_3_stock_bond.png)", "",
               _md(c["decades"], {"stock_bond_corr": f2, "both_negative_share": _num(".0%")}), ""]
        if "regimes" in c:
            md += [_md(c["regimes"], {"stock_bond_corr": f2, "bond_return_in_worst_10pct_stock_months": pc}), "",
                   f"36-month correlation on 36-month average inflation: intercept {c['infl_fit']['intercept']:+.2f}, "
                   f"slope {c['infl_fit']['slope_per_1pt_inflation']:+.3f} per percentage point (descriptive; windows "
                   "overlap).", ""]
        if "y2022" in c:
            y = c["y2022"]
            md += [f"2022: stocks {y['stocks']:+.1%}, bonds {y['bonds']:+.1%} (excess of bills); both fell in "
                   f"{y['months_both_negative']} of 12 months; correlation {y['corr']:+.2f}.", ""]
    md += ["## Part D: The investor's view", "",
           _md(dd["table"], {"mean_annual": pc, "vol_annual": _num(".1%"), "sharpe": _num(".2f"),
                             "worst_5pct_stock_months_avg": pc, "stocks_in_same_months": pc,
                             "best_5pct_stock_months_avg": pc,
                             "max_drawdown": _num(".0%")}), "", "![Lab Figure 10.4](fig10_4_worst_months.png)", "",
           "**Crisis periods (cumulative excess return)**", "",
           _md(dd["crisis"], {k: pc for k in dd["crisis"].columns if k != "period"}), "",
           "![Lab Figure 10.5](fig10_5_crises.png)", ""]
    if dd["div_ratio_6040"]:
        md += [f"Diversification ratio of 60/40: calm months {dd['div_ratio_6040']['calm']:.2f}, high-volatility months "
               f"{dd['div_ratio_6040']['high_vol']:.2f} (1.00 = no diversification).", ""]
    md += ["## Your verdict", "", "1. Do correlations really rise in turbulent markets once the volatility bias is removed?",
           "2. Which diversifiers held up in the worst months, and which did not?",
           "3. Is the failure of bonds in 2022 the rule or the exception?", ""]
    (out / "lab10_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["download", "sim"], default="download")
    for f in ("ff3", "intl", "ind10", "gs10", "cpi", "oil"):
        ap.add_argument(f"--{f}-file", dest=f, default=None)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, {k: getattr(args, k) for k in ("ff3", "intl", "ind10", "gs10", "cpi", "oil")}, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    a, b, c, dd = part_a(d), part_b(d), part_c(d), part_d(d)
    fig_exceedance(a, out, sim)
    fig_conditional(a, out, sim)
    if c:
        fig_stock_bond(c, out, sim)
    fig_worst(dd, out, sim)
    fig_crises(dd, out, sim)
    a["table"].to_csv(out / "table_conditional.csv", index=False)
    b.to_csv(out / "table_tail.csv", index=False)
    if c:
        c["decades"].to_csv(out / "table_stock_bond_decades.csv", index=False)
        if "regimes" in c:
            c["regimes"].to_csv(out / "table_inflation_regimes.csv", index=False)
    dd["table"].to_csv(out / "table_portfolios.csv", index=False)
    dd["crisis"].to_csv(out / "table_crises.csv", index=False)
    write_report(out, a, b, c, dd, sim, d.get("missing", []))
    summary = {"simulated": sim, "conditional": a["table"].to_dict(orient="records"),
               "y2022": c.get("y2022") if c else None, "infl_fit": c.get("infl_fit") if c else None,
               "portfolios": dd["table"].to_dict(orient="records"), "div_ratio_6040": dd["div_ratio_6040"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a["table"][["pair", "calm", "high_vol", "high_vol_adjusted", "worst_10pct_us_months"]].round(2).to_string(index=False))
    print(f"Report: {out / 'lab10_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
