"""Lab 4: Momentum and Its Crashes.

Chapter 4 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  Whether momentum has earned a premium throughout a century of U.S. data,
  when its crashes occur, and whether volatility scaling can make it
  holdable.

WHAT THE LAB DOES
  Part A  The premium. The momentum factor (Mom) from 1927 by decade and
          before and after Jegadeesh and Titman (1993); average returns of
          the ten momentum-sorted portfolios.
  Part B  Crashes. The worst months, the worst drawdowns, and when they
          happen: momentum's market beta in bear markets and rebounds
          (Daniel and Moskowitz 2016).
  Part C  Volatility scaling. Momentum scaled to constant risk using its own
          recent daily volatility (Barroso and Santa-Clara 2015).
  Part D  Time-series momentum on ten U.S. industry portfolios and the
          market (Moskowitz, Ooi and Pedersen 2012, equity-only version).
  Part E  Costs, and optionally other countries (Jensen-Kelly-Pedersen).

DATA
  Kenneth R. French Data Library: Fama/French 3 Factors (monthly), Momentum
  Factor (monthly and daily), 10 Portfolios Formed on Prior (12-2) Return,
  10 Industry Portfolios.

USAGE
  python lab04.py
  python lab04.py --source sim            # fake data, to test the code

OUTPUT
  report/lab04_report.md, five figures, CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, find_table, load_french_factors, load_french_tables
from fintest.inference import moments
from fintest.stats import ols_newey_west

ROOT = Path(__file__).resolve().parent
NMV_MOM_TURNOVER = 0.3452  # Novy-Marx and Velikov (2016), Table 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _col(df, *names):
    cols = {c.strip().lower().replace(" ", "").replace("-", ""): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "").replace("-", "")
        if k in cols:
            return df[cols[k]]
    return df.iloc[:, 0]


def load(source: str, files: dict, sim_seed: int = 0) -> dict:
    if source == "sim":
        return simulated(seed=sim_seed)
    ff3 = load_french_factors("ff3_monthly", path=files.get("ff3"))
    mom = load_french_factors("momentum_monthly", path=files.get("mom"))
    momd = load_french_factors("momentum_daily", path=files.get("mom_daily"))
    dec = find_table(load_french_tables("mom_portfolios_10", path=files.get("deciles")), "value", "weight")
    ind = find_table(load_french_tables("industry_10", path=files.get("industries")), "value", "weight")
    return {"mom": _col(mom, "Mom", "UMD"), "mkt": _col(ff3, "Mkt-RF"), "rf": _col(ff3, "RF"),
            "mom_daily": _col(momd, "Mom", "UMD"), "deciles": dec, "industries": ind, "simulated": False}


def simulated(n_months: int = 1182, seed: int = 0, start: str = "1927-01") -> dict:
    """Fake data with a momentum premium, crashes after bear markets, and daily momentum returns. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    mkt = 0.0065 + 0.053 * rng.standard_t(5, n_months) / math.sqrt(5 / 3)
    past = pd.Series(mkt).rolling(24).sum().shift(1).fillna(0).to_numpy()
    bear = past < 0
    vol = np.where(bear, 0.07, 0.035)
    beta = np.where(bear & (mkt > 0), -1.2, 0.0)  # crashes: momentum is short the rebound
    mom = 0.011 + beta * mkt + vol * rng.standard_normal(n_months)
    days = []
    for t, p in enumerate(idx):
        n = 21
        d = mom[t] / n + vol[t] / math.sqrt(n) * rng.standard_normal(n) * 0.9
        dates = pd.bdate_range(p.start_time, periods=n)
        days.append(pd.Series(d, index=dates))
    mom_daily = pd.concat(days)
    dec_mu = np.linspace(0.002, 0.012, 10)
    common = mkt[:, None]
    dec = pd.DataFrame(0.003 + dec_mu + common + 0.03 * rng.standard_normal((n_months, 10)), index=idx,
                       columns=["Lo PRIOR"] + [f"PRIOR {i}" for i in range(2, 10)] + ["Hi PRIOR"])
    ind = pd.DataFrame(0.003 + 0.9 * common + 0.025 * rng.standard_normal((n_months, 10)), index=idx,
                       columns=["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"])
    return {"mom": pd.Series(mom, index=idx), "mkt": pd.Series(mkt, index=idx), "rf": pd.Series(0.003, index=idx),
            "mom_daily": mom_daily, "deciles": dec, "industries": ind, "simulated": True}


def load_jkp(path, factor: str = "ret_12_1") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"location", "name", "date", "ret"} <= set(df.columns):
        raise DataUnavailable(f"JKP file needs columns location, name, date, ret; found {list(df.columns)}")
    df = df[df["name"].str.lower() == factor].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df.pivot_table(index="date", columns="location", values="ret", aggfunc="mean").sort_index()


def _stats(r: pd.Series) -> dict:
    x = r.dropna()
    v = x.to_numpy()
    m, s = v.mean(), v.std(ddof=1)
    mo = moments(v)
    wealth = (1 + x).cumprod()
    return {"mean_annual": 12 * m, "t": m / (s / math.sqrt(len(v))), "sharpe": m / s * math.sqrt(12),
            "vol_annual": s * math.sqrt(12), "skewness": mo["skewness"], "kurtosis": mo["kurtosis"],
            "worst_month": float(v.min()), "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
            "months": len(v), "start": str(x.index[0]), "end": str(x.index[-1])}


def _slice(s, a=None, b=None):
    return s.loc[pd.Period(a, "M") if a else None:pd.Period(b, "M") if b else None]


# ---------------------------------------------------------------------------
# Part A: the momentum premium
# ---------------------------------------------------------------------------


def part_a(d: dict) -> dict:
    mom = d["mom"].dropna()
    rows = [{"period": "Full sample", **_stats(mom)},
            {"period": "Before Jegadeesh–Titman sample (to 1964)", **_stats(_slice(mom, None, "1964-12"))},
            {"period": "Jegadeesh–Titman sample (1965–1989)", **_stats(_slice(mom, "1965-01", "1989-12"))},
            {"period": "After publication (1993–today)", **_stats(_slice(mom, "1993-04", None))}]
    decades = []
    for start in range(int(mom.index[0].year) // 10 * 10, int(mom.index[-1].year) + 1, 10):
        s = _slice(mom, f"{max(start, mom.index[0].year)}-01", f"{start + 9}-12")
        if len(s) >= 24:
            decades.append({"decade": f"{start}s", **_stats(s)})
    dec = d["deciles"].dropna()
    dec_means = (dec.mean() * 12).rename("mean_annual")
    spread = dec.iloc[:, -1] - dec.iloc[:, 0]
    return {"table": pd.DataFrame(rows), "decades": pd.DataFrame(decades), "decile_means": dec_means,
            "decile_spread": _stats(spread)}


# ---------------------------------------------------------------------------
# Part B: crashes
# ---------------------------------------------------------------------------


def part_b(d: dict, n_worst: int = 10) -> dict:
    df = pd.DataFrame({"mom": d["mom"], "mkt": d["mkt"], "rf": d["rf"]}).dropna()
    df["past24"] = (np.log1p(df["mkt"] + df["rf"])).rolling(24).sum().shift(1)
    df["bear"] = (df["past24"] < 0).astype(float)
    df["up"] = (df["mkt"] > 0).astype(float)
    worst = df.nsmallest(n_worst, "mom")[["mom", "mkt", "past24", "bear"]].copy()
    worst.index = worst.index.astype(str)
    x = df.dropna()
    X = np.column_stack([x["mkt"], x["bear"] * x["mkt"], x["bear"] * x["up"] * x["mkt"], x["bear"]])
    reg = ols_newey_west(x["mom"].to_numpy(), X)
    names = ["alpha", "beta (all months)", "extra beta in bear markets", "extra beta in bear-market rebounds",
             "bear-market intercept shift"]
    coef = pd.DataFrame({"term": names, "estimate": reg["coef"][:, 0], "t": reg["t"][:, 0]})
    states = []
    for label, m in (("Bull market (past 24-month market return ≥ 0)", x["bear"] == 0),
                     ("Bear market, market falls this month", (x["bear"] == 1) & (x["up"] == 0)),
                     ("Bear market, market rises this month", (x["bear"] == 1) & (x["up"] == 1))):
        s = x.loc[m, "mom"]
        states.append({"state": label, "months": int(m.sum()), "mean_monthly": float(s.mean()),
                       "worst_month": float(s.min()) if len(s) else float("nan")})
    return {"worst": worst, "regression": coef, "states": pd.DataFrame(states), "data": x}


# ---------------------------------------------------------------------------
# Part C: volatility scaling
# ---------------------------------------------------------------------------


def vol_scaled(mom: pd.Series, mom_daily: pd.Series, target: float = 0.12, window: int = 126,
               cap: float | None = None) -> pd.DataFrame:
    """Barroso and Santa-Clara (2015): weight_t = target / forecast volatility from the prior `window` days."""
    dd = mom_daily.dropna()
    rv = (dd ** 2).rolling(window, min_periods=window // 2).mean() * 252
    month_end = rv.groupby(rv.index.to_period("M")).last()  # known at the end of each month
    sigma = np.sqrt(month_end).shift(1)  # forecast for month t uses data to the end of month t-1
    out = pd.DataFrame({"mom": mom}).join(sigma.rename("sigma_hat"), how="inner").dropna()
    w = target / out["sigma_hat"]
    if cap:
        w = w.clip(upper=cap)
    out["weight"] = w
    out["scaled"] = out["weight"] * out["mom"]
    return out


def part_c(d: dict, target: float = 0.12, cap: float | None = None) -> dict:
    vs = vol_scaled(d["mom"], d["mom_daily"], target, cap=cap)
    raw = vs["mom"] * (vs["scaled"].std() / vs["mom"].std())  # raw momentum at the same average volatility
    rows = [{"strategy": "Momentum (raw)", **_stats(vs["mom"])},
            {"strategy": "Momentum, raw, scaled to the same overall volatility", **_stats(raw)},
            {"strategy": f"Volatility-scaled momentum (target {target:.0%})", **_stats(vs["scaled"])}]
    return {"table": pd.DataFrame(rows), "series": vs, "raw_matched": raw,
            "avg_weight": float(vs["weight"].mean()), "max_weight": float(vs["weight"].max()),
            "weight_turnover": float(vs["weight"].diff().abs().mean())}


# ---------------------------------------------------------------------------
# Part D: time-series momentum on industries
# ---------------------------------------------------------------------------


def part_d(d: dict, lookback: int = 12, target_vol: float = 0.40, vol_window: int = 36) -> dict:
    ind = d["industries"].dropna()
    rf = d["rf"].reindex(ind.index)
    assets = ind.sub(rf, axis=0)
    assets["Market"] = d["mkt"].reindex(ind.index)
    assets = assets.dropna()
    past = np.log1p(assets).rolling(lookback).sum()
    sig = np.sign(past).shift(1)  # signal known at the end of the previous month
    vol = assets.rolling(vol_window, min_periods=12).std().shift(1) * math.sqrt(12)
    pos = sig * target_vol / vol
    pnl = (pos * assets).mean(axis=1).dropna()
    long_only = assets.mean(axis=1).reindex(pnl.index)
    reg = ols_newey_west(pnl.to_numpy(), long_only.to_numpy())
    rows = [{"strategy": "Time-series momentum (11 equity portfolios)", **_stats(pnl)},
            {"strategy": "Equal-weight long-only, same assets", **_stats(long_only)}]
    return {"table": pd.DataFrame(rows), "pnl": pnl, "long_only": long_only,
            "alpha_annual": 12 * float(reg["coef"][0, 0]), "alpha_t": float(reg["t"][0, 0]),
            "beta": float(reg["coef"][1, 0])}


# ---------------------------------------------------------------------------
# Part E: costs and other countries
# ---------------------------------------------------------------------------


def part_e(d: dict, c_out: dict, costs=(0.002, 0.005), jkp: pd.DataFrame | None = None) -> dict:
    mom = d["mom"].dropna()
    rows = []
    for c in costs:
        net = mom - 4 * NMV_MOM_TURNOVER * c
        vs = c_out["series"]
        # The scaled strategy trades its momentum book (scaled by the weight) plus the change in weight.
        extra = vs["weight"].diff().abs().fillna(0) * 2 * c
        net_vs = vs["scaled"] - 4 * NMV_MOM_TURNOVER * c * vs["weight"] - extra
        rows.append({"one_way_cost": c, "raw_net_mean_annual": 12 * net.mean(), "raw_net_sharpe": _stats(net)["sharpe"],
                     "scaled_net_mean_annual": 12 * net_vs.mean(), "scaled_net_sharpe": _stats(net_vs)["sharpe"]})
    out = {"costs": pd.DataFrame(rows)}
    if jkp is not None:
        jr = []
        for country in jkp.columns:
            s = jkp[country].dropna()
            if len(s) >= 120:
                st = _stats(s)
                jr.append({"country": country, "mean_annual": st["mean_annual"], "t": st["t"],
                           "skewness": st["skewness"], "months": st["months"]})
        out["jkp"] = pd.DataFrame(jr).sort_values("t", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_cumulative(d, b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    mom = d["mom"].dropna()
    w = (1 + mom).cumprod()
    dates = mom.index.to_timestamp()
    ax.plot(dates, w, color=plots.ORANGE, linewidth=1.8)
    worst5 = sorted(b["worst"].index[:5])
    for k, p in enumerate(worst5):
        per = pd.Period(p, "M")
        if per in w.index:
            ax.plot(per.to_timestamp(), w.loc[per], "o", color=plots.RED, markersize=6)
            ax.annotate(f"{p}: {b['worst'].loc[p, 'mom']:.0%}", (per.to_timestamp(), w.loc[per]),
                        textcoords="offset points", xytext=(-10, -18 - 13 * k), ha="right", fontsize=8,
                        color=plots.RED, arrowprops={"arrowstyle": "-", "color": plots.RED, "lw": 0.5})
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 in Mom (log scale)")
    plots.titles(fig, "Lab Figure 4.1  Momentum: a steep climb with sudden falls",
                 _sub(sim, f"Fama–French momentum factor (UMD), {mom.index[0]} to {mom.index[-1]} · red: the five "
                           "worst months"))
    plots.finish(fig, out / "fig4_1_cumulative.png")
    plt.close(fig)


def fig_deciles_decades(a, out, sim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(plots.SURFACE)
    ax = axes[0]
    plots.style_axes(ax)
    m = a["decile_means"]
    ax.bar(range(1, 11), m.to_numpy(), color=[plots.BLUE] * 9 + [plots.ORANGE])
    ax.set_xticks(range(1, 11), ["Losers"] + [""] * 8 + ["Winners"], fontsize=8.5)
    plots.percent_axis(ax)
    ax.set_title("Average return by past-return decile", loc="left", fontsize=10, color=plots.INK)
    ax = axes[1]
    plots.style_axes(ax)
    dd = a["decades"]
    cols = [plots.ORANGE if v > 0 else plots.RED for v in dd["mean_annual"]]
    ax.bar(dd["decade"], dd["mean_annual"], color=cols)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.set_title("Momentum factor, average return by decade", loc="left", fontsize=10, color=plots.INK)
    plots.titles(fig, "Lab Figure 4.2  Winners beat losers in most decades, not all",
                 _sub(sim, "Annualized average returns · deciles: value-weighted portfolios sorted on months −12 to −2"))
    plots.finish(fig, out / "fig4_2_deciles_decades.png")
    plt.close(fig)


def fig_crash(b, out, sim):
    fig, ax, plt = plots.new_figure(8.6, 5.0)
    x = b["data"]
    bull, bear = x[x.bear == 0], x[x.bear == 1]
    ax.scatter(bull["mkt"], bull["mom"], s=8, color=plots.INK_3, alpha=0.5, label="Bull markets", linewidths=0)
    ax.scatter(bear["mkt"], bear["mom"], s=14, color=plots.RED, alpha=0.7, label="Bear markets (past 24 months < 0)",
               linewidths=0)
    for g, col in ((bull, plots.INK_2), (bear[bear.mkt > 0], plots.RED)):
        if len(g) > 5:
            bfit = np.polyfit(g["mkt"], g["mom"], 1)
            xs = np.linspace(g["mkt"].min(), g["mkt"].max(), 10)
            ax.plot(xs, np.polyval(bfit, xs), color=col, linewidth=1.5)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.axvline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax, "x")
    plots.percent_axis(ax)
    ax.set_xlabel("Market excess return this month")
    ax.set_ylabel("Momentum return this month")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    plots.titles(fig, "Lab Figure 4.3  Momentum crashes when beaten-down markets rebound",
                 _sub(sim, "Lines: fit in bull markets and in bear-market up months · after a market fall, the loser "
                           "portfolio is full of high-beta stocks"))
    plots.finish(fig, out / "fig4_3_crashes.png")
    plt.close(fig)


def fig_scaled(c, out, sim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(plots.SURFACE)
    vs = c["series"]
    dates = vs.index.to_timestamp()
    for s, col, lab in ((c["raw_matched"], plots.INK_3, "Raw momentum (same average volatility)"),
                        (vs["scaled"], plots.ORANGE, "Volatility-scaled momentum")):
        w = (1 + s).cumprod()
        axes[0].plot(dates, w, color=col, linewidth=1.6, label=lab)
        axes[1].plot(dates, w / w.cummax() - 1, color=col, linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[1].set_ylabel("Drawdown")
    plots.percent_axis(axes[1])
    for ax in axes:
        plots.style_axes(ax)
    plots.titles(fig, "Lab Figure 4.4  Scaling momentum by its own volatility tames the crashes",
                 _sub(sim, "Barroso–Santa-Clara: weight = 12% / volatility of daily momentum returns over the prior "
                           "six months, known in advance"))
    plots.finish(fig, out / "fig4_4_vol_scaled.png")
    plt.close(fig)


def fig_tsmom(dd, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    dates = dd["pnl"].index.to_timestamp()
    for s, col, lab in ((dd["pnl"], plots.VIOLET, "Time-series momentum"),
                        (dd["long_only"], plots.INK_3, "Equal-weight long-only")):
        ax.plot(dates, (1 + s).cumprod(), color=col, linewidth=1.6, label=lab)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylabel("Growth of $1 (log scale)")
    plots.titles(fig, "Lab Figure 4.5  Trend following within U.S. equities",
                 _sub(sim, f"Long or short each of 10 industries and the market by the sign of its past 12-month excess "
                           f"return, scaled to 40% volatility · alpha vs long-only {dd['alpha_annual']:+.1%} a year "
                           f"(t = {dd['alpha_t']:.1f})"))
    plots.finish(fig, out / "fig4_5_tsmom.png")
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


STATFMT = {"Mean (annual)": _num("+.1%"), "t": _num(".2f"), "Sharpe": _num(".2f"), "Volatility": _num(".1%"),
           "Skewness": _num("+.2f"), "Worst month": _num(".1%"), "Max drawdown": _num(".0%")}


def _stat_cols(df, first):
    return df[[first, "mean_annual", "t", "sharpe", "vol_annual", "skewness", "worst_month", "max_drawdown",
               "months"]].rename(columns={"mean_annual": "Mean (annual)", "sharpe": "Sharpe", "vol_annual": "Volatility",
                                          "skewness": "Skewness", "worst_month": "Worst month",
                                          "max_drawdown": "Max drawdown", "months": "Months"})


def write_report(out, d, a, b, c, dd, e, sim):
    md = ["# Lab 4 results: Momentum and Its Crashes", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** whether momentum has earned a premium throughout a century of U.S. data, when its crashes occur, and whether volatility scaling can make it holdable.", "",
           "## Part A: The momentum premium", "", _md(_stat_cols(a["table"], "period"), STATFMT), "",
           "**By decade**", "", _md(_stat_cols(a["decades"], "decade"), STATFMT), "",
           f"Decile spread (winners minus losers, value-weighted deciles): {a['decile_spread']['mean_annual']:+.1%} a "
           f"year, t = {a['decile_spread']['t']:.2f}.", "",
           "![Lab Figure 4.1](fig4_1_cumulative.png)", "", "![Lab Figure 4.2](fig4_2_deciles_decades.png)", ""]
    w = b["worst"].reset_index().rename(columns={"index": "month", "date": "Month", "mom": "Momentum", "mkt": "Market",
                                                 "past24": "Market, past 24 months (log)", "bear": "Bear market"})
    md += ["## Part B: Crashes", "", "**The worst months**", "",
           _md(w, {"Momentum": _num(".1%"), "Market": _num("+.1%"), "Market, past 24 months (log)": _num("+.0%"),
                   "Bear market": lambda v: "yes" if v == 1 else "no"}), "",
           "**Momentum's market exposure by state** (Newey–West t)", "",
           _md(b["regression"], {"estimate": _num("+.3f"), "t": _num(".2f")}), "",
           _md(b["states"], {"mean_monthly": _num("+.2%"), "worst_month": _num(".1%")}), "",
           "![Lab Figure 4.3](fig4_3_crashes.png)", ""]
    md += ["## Part C: Volatility scaling", "", _md(_stat_cols(c["table"], "strategy"), STATFMT), "",
           f"Average weight {c['avg_weight']:.2f}, maximum {c['max_weight']:.2f}; average monthly change in weight "
           f"{c['weight_turnover']:.2f}.", "", "![Lab Figure 4.4](fig4_4_vol_scaled.png)", ""]
    md += ["## Part D: Time-series momentum (equity-only version)", "", _md(_stat_cols(dd["table"], "strategy"), STATFMT),
           "", f"Alpha against the equal-weight long-only portfolio: {dd['alpha_annual']:+.1%} a year (t = "
           f"{dd['alpha_t']:.2f}), beta {dd['beta']:+.2f}.", "",
           "Moskowitz, Ooi and Pedersen (2012) study 58 futures across equities, bonds, currencies and commodities; this "
           "version uses only U.S. equity portfolios, where assets are highly correlated, so it is a weaker test.", "",
           "![Lab Figure 4.5](fig4_5_tsmom.png)", ""]
    md += ["## Part E: Costs and other countries", "",
           _md(e["costs"].assign(one_way_cost=e["costs"]["one_way_cost"] * 1e4).rename(columns={
               "one_way_cost": "One-way cost (bp)", "raw_net_mean_annual": "Raw, net mean", "raw_net_sharpe": "Raw, net Sharpe",
               "scaled_net_mean_annual": "Scaled, net mean", "scaled_net_sharpe": "Scaled, net Sharpe"}),
               {"One-way cost (bp)": _num(".0f"), "Raw, net mean": _num("+.1%"), "Raw, net Sharpe": _num(".2f"),
                "Scaled, net mean": _num("+.1%"), "Scaled, net Sharpe": _num(".2f")}), "",
           "Turnover 34.5% a month (Novy-Marx and Velikov 2016); cost = 4 × turnover × one-way cost; the scaled strategy "
           "also pays for changes in its weight.", ""]
    if "jkp" in e:
        md += ["**Jensen–Kelly–Pedersen momentum factor (ret_12_1) by country**", "",
               _md(e["jkp"], {"mean_annual": _num("+.1%"), "t": _num(".2f"), "skewness": _num("+.2f")}), ""]
    md += ["## Questions to think about", "",
           "1. Do past winners keep beating past losers across the century, after publication and after costs?",
           "2. How severe are momentum's crashes, and can they be anticipated from the state of the market?",
           "3. Does volatility scaling change the answer, and would you trust it out of sample?", ""]
    (out / "lab04_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    for f in ("ff3", "mom", "mom-daily", "deciles", "industries"):
        ap.add_argument(f"--{f}-file", dest=f"{f.replace('-', '_')}_file")
    ap.add_argument("--jkp-file", default=None)
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--weight-cap", type=float, default=None)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"ff3": args.ff3_file, "mom": args.mom_file, "mom_daily": args.mom_daily_file, "deciles": args.deciles_file,
             "industries": args.industries_file}
    try:
        d = load(args.source, files, args.sim_seed)
        jkp = load_jkp(args.jkp_file) if args.jkp_file else None
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    a, b = part_a(d), part_b(d)
    c = part_c(d, args.target_vol, args.weight_cap)
    dd = part_d(d)
    e = part_e(d, c, jkp=jkp)
    fig_cumulative(d, b, out, sim)
    fig_deciles_decades(a, out, sim)
    fig_crash(b, out, sim)
    fig_scaled(c, out, sim)
    fig_tsmom(dd, out, sim)
    a["table"].to_csv(out / "table_periods.csv", index=False)
    a["decades"].to_csv(out / "table_decades.csv", index=False)
    b["worst"].to_csv(out / "table_worst_months.csv")
    b["regression"].to_csv(out / "table_crash_regression.csv", index=False)
    c["table"].to_csv(out / "table_vol_scaled.csv", index=False)
    dd["table"].to_csv(out / "table_tsmom.csv", index=False)
    e["costs"].to_csv(out / "table_costs.csv", index=False)
    write_report(out, d, a, b, c, dd, e, sim)
    summary = {"simulated": sim, "periods": a["table"].to_dict(orient="records"),
               "worst_month": float(b["worst"]["mom"].min()), "crash_regression": b["regression"].to_dict(orient="records"),
               "vol_scaled": c["table"].to_dict(orient="records"), "tsmom": dd["table"].to_dict(orient="records"),
               "costs": e["costs"].to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for _, r in a["table"].iterrows():
        print(f"{r.period:45s} {r.mean_annual:+.1%}/yr t={r.t:5.2f} skew={r.skewness:+.2f}")
    ct = c["table"]
    print(f"Sharpe raw {ct.iloc[0].sharpe:.2f} -> scaled {ct.iloc[2].sharpe:.2f}; max drawdown "
          f"{ct.iloc[1].max_drawdown:.0%} -> {ct.iloc[2].max_drawdown:.0%}")
    print(f"Report: {out / 'lab04_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
