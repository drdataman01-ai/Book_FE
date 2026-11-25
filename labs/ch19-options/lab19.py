"""Lab 19: Does Selling Options Earn a Reliable Premium?

Chapter 19 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "Implied volatility exceeds the volatility that follows, on average, so
   option sellers, for example covered-call and put-writing strategies, earn
   a reliable premium."

WHAT THE LAB DOES
  Part A  The variance risk premium: the VIX against the market's realized
          volatility over the following month, since 1990.
  Part B  Option-writing strategies against the market: one-month
          at-the-money put-writing and buy-write portfolios rebuilt from the
          VIX and daily market returns, and (optionally) Cboe's PUT and BXM
          indices. Returns, volatility, skewness, drawdowns, alphas.
  Part C  Stress tests: crash months, the payoff shape, and performance
          measures that cannot be gamed by selling crash insurance.
  Part D  Equity exposure versus the volatility premium: how much of the
          return is beta, and what a pure short-variance position earns.

DATA (free)
  Kenneth R. French Data Library: Fama/French 3 Factors [Daily].
  FRED: VIXCLS (from 1990).
  Optional: Cboe index histories for PUT and BXM (https://www.cboe.com/us/indices/).

USAGE
  python lab19.py
  python lab19.py --put-file data/PUT_History.csv --bxm-file data/BXM_History.csv
  python lab19.py --source sim
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
from fintest.options import (
    capture_ratios,
    monthly_option_writes,
    monthly_returns_from_levels,
    mppm,
    read_cboe_index,
    up_down_betas,
    variance_swap_pnl,
)
from fintest.stats import ols_newey_west
from fintest.vol import garch_forecast_sum, load_fred

ROOT = Path(__file__).resolve().parent
PW, BW = "Put-write, ATM (rebuilt)", "Buy-write, 2% OTM (rebuilt)"
COLORS = {"Market": plots.INK_3, PW: plots.BLUE, BW: plots.AQUA,
          "Cboe PUT": plots.VIOLET, "Cboe BXM": plots.ORANGE, "Beta-matched market": plots.INK_2}
CRISES = [("Oct 1987", "1987-10", "1987-10"), ("Aug 1998", "1998-08", "1998-08"),
          ("Sep 2008–Feb 2009", "2008-09", "2009-02"), ("Feb–Mar 2020", "2020-02", "2020-03"),
          ("2022", "2022-01", "2022-12")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        return simulated(seed)
    ff = load_french_factors("ff3_daily", path=files.get("ff3"))
    cols = {c.strip().lower(): c for c in ff.columns}
    rf = ff[cols["rf"]]
    ret = (ff[cols["mkt-rf"]] + rf).rename("ret")
    vix = load_fred("VIXCLS", files.get("vix"))
    cboe = {}
    for key, name in (("put", "Cboe PUT"), ("bxm", "Cboe BXM")):
        if files.get(key):
            cboe[name] = monthly_returns_from_levels(read_cboe_index(files[key]))
    return {"ret": ret, "rf": rf, "vix": vix, "cboe": cboe, "simulated": False}


def simulated(seed: int = 0, n: int = 9500) -> dict:
    """GJR-GARCH daily returns with t shocks, a VIX priced 12% above expected volatility, and Cboe-like indices.

    FOR TESTING ONLY.
    """
    rng = np.random.default_rng(seed)
    om, a, g, b = 1.2e-6, 0.03, 0.10, 0.905
    p = {"long_run_var": om / (1 - a - b - g / 2), "persistence": a + b + g / 2}
    r, s2 = np.zeros(n), np.zeros(n)
    s2[0] = p["long_run_var"]
    z = rng.standard_t(6, n) / math.sqrt(1.5)
    for t in range(1, n):
        s2[t] = om + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = 0.0004 + math.sqrt(s2[t]) * z[t]
    idx = pd.bdate_range("1988-01-04", periods=n)
    nxt = om + (a + g * (r < 0)) * r ** 2 + b * s2
    exp_var21 = garch_forecast_sum(nxt, p, 21)
    vix = 100 * np.sqrt(exp_var21 / 21 * 252) * 1.12 * np.exp(0.04 * rng.standard_normal(n))
    ret = pd.Series(r, idx, name="ret")
    rf = pd.Series(0.00012, idx)
    vix_s = pd.Series(vix, idx)
    vix_s = vix_s[vix_s.index >= "1990-01-02"]
    full_iv = pd.Series(vix, idx)
    syn = monthly_option_writes(ret, rf, full_iv, iv_shift=-2.0)
    cboe = {"Cboe PUT": syn["put_write"] + 0.001 * rng.standard_normal(len(syn)),
            "Cboe BXM": syn["buy_write"] + 0.001 * rng.standard_normal(len(syn))}
    return {"ret": ret, "rf": rf, "vix": vix_s, "cboe": cboe, "simulated": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def monthly(ret: pd.Series) -> pd.Series:
    return (1 + ret).groupby(ret.index.to_period("M")).prod() - 1


def _stats(r: pd.Series, rf: pd.Series, mkt_ex: pd.Series) -> dict:
    df = pd.concat({"r": r, "rf": rf, "m": mkt_ex}, axis=1).dropna()
    ex = df["r"] - df["rf"]
    v = ex.to_numpy()
    w = (1 + df["r"]).cumprod()
    reg = ols_newey_west(v, df["m"].to_numpy())
    ud = up_down_betas(v, df["m"].to_numpy())
    cap = capture_ratios(v, df["m"].to_numpy())
    return {"mean_annual": 12 * df["r"].mean(), "excess_annual": 12 * v.mean(), "vol_annual": v.std(ddof=1) * math.sqrt(12),
            "sharpe": v.mean() / v.std(ddof=1) * math.sqrt(12), "skewness": moments(v)["skewness"],
            "worst_month": float(df["r"].min()), "max_drawdown": float((w / w.cummax() - 1).min()),
            "beta": float(reg["coef"][1, 0]), "alpha_annual": 12 * float(reg["coef"][0, 0]),
            "alpha_t": float(reg["t"][0, 0]), **ud, **cap,
            "mppm_3": mppm(df["r"].to_numpy(), df["rf"].to_numpy(), 3.0),
            "months": len(df), "start": str(df.index[0]), "end": str(df.index[-1])}


# ---------------------------------------------------------------------------
# Part A: variance risk premium
# ---------------------------------------------------------------------------


def part_a(d: dict) -> dict:
    vs = variance_swap_pnl(d["ret"], d["vix"])
    vs = vs[vs.index >= d["vix"].index[0].to_period("M")]
    iv = np.sqrt(vs["implied"] * 252 / 21)
    rv = np.sqrt(vs["realized"] * 252 / 21)
    gap = iv - rv
    reg = ols_newey_west(gap.to_numpy(), np.zeros((len(gap), 0))) if False else None  # noqa: F841
    m = gap.mean()
    se = gap.std(ddof=1) / math.sqrt(len(gap))
    # Newey-West standard error of the mean
    x = gap.to_numpy() - m
    L = 6
    s = (x @ x) / len(x) + 2 * sum((1 - k / (L + 1)) * (x[k:] @ x[:-k]) / len(x) for k in range(1, L + 1))
    se_nw = math.sqrt(s / len(x))
    var_prem = (vs["implied"] - vs["realized"]) * 252 / 21
    summary = {"months": len(gap), "start": str(gap.index[0]), "end": str(gap.index[-1]),
               "mean_implied_vol": float(iv.mean()), "mean_realized_vol": float(rv.mean()),
               "mean_gap_vol_points": 100 * float(m), "t_nw": float(m / se_nw), "t_iid": float(m / se),
               "share_months_positive": float((gap > 0).mean()), "median_gap_vol_points": 100 * float(gap.median()),
               "worst_gap_vol_points": 100 * float(gap.min()), "worst_month": str(gap.idxmin()),
               "mean_variance_premium": float(var_prem.mean()),
               "implied_over_realized_variance": float(vs["implied"].mean() / vs["realized"].mean())}
    by_year = gap.groupby(gap.index.year).mean() * 100
    return {"summary": summary, "iv": iv, "rv": rv, "gap": gap, "by_year": by_year, "vs": vs}


# ---------------------------------------------------------------------------
# Part B: strategies
# ---------------------------------------------------------------------------


def part_b(d: dict, iv_shift: float = 0.0, call_strike: float = 1.02) -> dict:
    syn = monthly_option_writes(d["ret"], d["rf"], d["vix"], call_strike=call_strike, iv_shift=iv_shift)
    mret = monthly(d["ret"])
    mrf = monthly(d["rf"])
    mkt_ex = mret - mrf
    series = {"Market": mret, PW: syn["put_write"], BW: syn["buy_write"]}
    series.update(d.get("cboe", {}))
    rows = []
    common = syn.index
    for name, s in series.items():
        rows.append({"strategy": name, "sample": "Own history", **_stats(s, mrf, mkt_ex)})
    for name, s in series.items():
        rows.append({"strategy": name, "sample": "Common period", **_stats(s.reindex(common), mrf, mkt_ex)})
    for r in rows:  # the market's alpha against itself is zero by construction
        if r["strategy"] == "Market":
            r["alpha_annual"], r["alpha_t"] = 0.0, float("nan")
    return {"series": series, "syn": syn, "rf": mrf, "mkt_ex": mkt_ex, "table": pd.DataFrame(rows), "common": common}


# ---------------------------------------------------------------------------
# Part C: stress tests
# ---------------------------------------------------------------------------


def part_c(b: dict) -> dict:
    rows = []
    for label, s, e in CRISES:
        row = {"period": label}
        for name, ser in b["series"].items():
            x = ser.loc[pd.Period(s, "M"):pd.Period(e, "M")]
            n_exp = (pd.Period(e, "M") - pd.Period(s, "M")).n + 1
            row[name] = float((1 + x).prod() - 1) if len(x) == n_exp else float("nan")
        rows.append(row)
    crisis = pd.DataFrame(rows)
    # worst five market months in the common period, and what each strategy did
    m = b["series"]["Market"].reindex(b["common"])
    worst = m.nsmallest(5).index
    w = pd.DataFrame({name: ser.reindex(worst) for name, ser in b["series"].items()})
    w.index = w.index.astype(str)
    t = b["table"][b["table"]["sample"] == "Common period"]
    measures = t[["strategy", "sharpe", "skewness", "mppm_3", "beta_down", "beta_up", "downside_capture",
                  "upside_capture"]].reset_index(drop=True)
    return {"crisis": crisis, "worst_months": w, "measures": measures}


# ---------------------------------------------------------------------------
# Part D: equity exposure versus volatility premium
# ---------------------------------------------------------------------------


def part_d(d: dict, b: dict, a: dict) -> dict:
    rows = []
    mrf, mkt_ex = b["rf"], b["mkt_ex"]
    for name in [n for n in b["series"] if n != "Market"]:
        s = b["series"][name].reindex(b["common"])
        df = pd.concat({"r": s, "rf": mrf, "m": mkt_ex}, axis=1).dropna()
        ex = df["r"] - df["rf"]
        reg = ols_newey_west(ex.to_numpy(), df["m"].to_numpy())
        beta, alpha = float(reg["coef"][1, 0]), float(reg["coef"][0, 0])
        bm = df["rf"] + beta * df["m"]  # the market, scaled to the same beta
        st_bm = _stats(bm, df["rf"], df["m"])
        rows.append({"strategy": name, "excess_annual": 12 * ex.mean(), "from_beta": 12 * beta * df["m"].mean(),
                     "alpha_annual": 12 * alpha, "alpha_t": float(reg["t"][0, 0]), "beta": beta,
                     "share_from_beta": beta * df["m"].mean() / ex.mean() if ex.mean() != 0 else float("nan"),
                     "sharpe": ex.mean() / ex.std(ddof=1) * math.sqrt(12), "sharpe_beta_matched": st_bm["sharpe"],
                     "skew": moments(ex.to_numpy())["skewness"], "skew_beta_matched": st_bm["skewness"]})
    decomposition = pd.DataFrame(rows)
    # pure variance premium: short a one-month variance swap, sized so the position has 10% annual volatility
    pnl = a["vs"]["pnl"]
    scale = 0.10 / (pnl.std(ddof=1) * math.sqrt(12))
    vsr = pnl * scale
    v = vsr.to_numpy()
    w = (1 + vsr).cumprod()
    var_swap = {"mean_annual": 12 * v.mean(), "vol_annual": 0.10, "sharpe": v.mean() / v.std(ddof=1) * math.sqrt(12),
                "skewness": moments(v)["skewness"], "worst_month": float(v.min()), "worst_month_date": str(vsr.idxmin()),
                "max_drawdown": float((w / w.cummax() - 1).min()), "share_months_positive": float((v > 0).mean()),
                "corr_with_market": float(np.corrcoef(v, mkt_ex.reindex(vsr.index).fillna(0).to_numpy())[0, 1])}
    return {"decomposition": decomposition, "var_swap": var_swap, "var_swap_returns": vsr}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_vix(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    dates = a["iv"].index.to_timestamp()
    ax.plot(dates, a["iv"], color=plots.RED, linewidth=1.3, label="VIX at month end (implied, next 30 days)")
    ax.plot(dates, a["rv"], color=plots.INK_3, linewidth=1.1, label="Realized volatility over the next 21 trading days")
    plots.percent_axis(ax)
    ax.set_ylabel("Annualized volatility")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    s = a["summary"]
    plots.titles(fig, "Lab Figure 19.1  Implied volatility usually exceeds what follows",
                 _sub(sim, f"Implied above realized in {s['share_months_positive']:.0%} of months · average gap "
                           f"{s['mean_gap_vol_points']:.1f} vol points (Newey–West t = {s['t_nw']:.1f})"))
    plots.finish(fig, out / "fig19_1_vix_realized.png")
    plt.close(fig)


def fig_gap(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    g = a["gap"] * 100
    dates = g.index.to_timestamp()
    ax.bar(dates, g.clip(lower=0), width=25, color=plots.BLUE, label="Implied above realized (seller gains)")
    ax.bar(dates, g.clip(upper=0), width=25, color=plots.RED, label="Realized above implied (seller loses)")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_ylabel("VIX minus realized volatility (vol points)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    plots.titles(fig, "Lab Figure 19.2  Small gains most months, large losses in a few",
                 _sub(sim, f"Median gap {a['summary']['median_gap_vol_points']:+.1f} points · worst "
                           f"{a['summary']['worst_gap_vol_points']:+.1f} points ({a['summary']['worst_month']})"))
    plots.finish(fig, out / "fig19_2_gap.png")
    plt.close(fig)


def fig_cumulative(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    for name, s in b["series"].items():
        x = s.reindex(b["common"]).dropna()
        ax.plot(x.index.to_timestamp(), (1 + x).cumprod(), color=COLORS.get(name, plots.INK_2), linewidth=1.5,
                label=name, linestyle="--" if name.startswith("Cboe") else "-")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 19.3  Option writing against the market",
                 _sub(sim, f"Monthly returns, {b['common'][0]} to {b['common'][-1]} · rebuilt strategies sell "
                           "one-month options priced at the VIX"))
    plots.finish(fig, out / "fig19_3_cumulative.png")
    plt.close(fig)


def fig_payoff(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    m = b["mkt_ex"].reindex(b["common"])
    for name in (PW, BW):
        y = b["series"][name].reindex(b["common"]) - b["rf"].reindex(b["common"])
        ax.scatter(m, y, s=10, alpha=0.6, color=COLORS[name], label=name, edgecolors="none")
    lim = [float(m.min()) * 1.05, float(m.max()) * 1.05]
    ax.plot(lim, lim, color=plots.INK_3, linestyle="--", linewidth=1, label="Market (45° line)")
    ax.axhline(0, color=plots.GRID, linewidth=1)
    ax.axvline(0, color=plots.GRID, linewidth=1)
    plots.percent_axis(ax)
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Market excess return in the month")
    ax.set_ylabel("Strategy excess return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 19.4  The option seller's payoff: capped gains, full losses",
                 _sub(sim, "Each dot is a month; points below the dashed line underperformed the market"))
    plots.finish(fig, out / "fig19_4_payoff.png")
    plt.close(fig)


def fig_decomposition(dd, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    t = dd["decomposition"]
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["from_beta"], color=plots.INK_3, label="From market exposure (beta × market premium)")
    ax.barh(y, t["alpha_annual"], left=np.where(t["alpha_annual"] >= 0, t["from_beta"], 0), color=plots.BLUE,
            label="Alpha (what is left)")
    for yy, (_, row) in zip(y, t.iterrows()):
        ax.text(max(row.from_beta + max(row.alpha_annual, 0), row.from_beta) + 0.002, yy,
                f"β = {row.beta:.2f}, α t = {row.alpha_t:.1f}", va="center", fontsize=8.5, color=plots.INK_2)
    ax.set_yticks(y, t["strategy"], fontsize=9)
    ax.axvline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, float(t["alpha_annual"].min()) * 1.3), float((t["from_beta"] + t["alpha_annual"].clip(lower=0)).max()) * 1.6)
    ax.set_xlabel("Average excess return, annual")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    vs = dd["var_swap"]
    plots.titles(fig, "Lab Figure 19.5  How much of the premium is just equity?",
                 _sub(sim, f"For comparison, a pure short variance swap (10% vol): Sharpe {vs['sharpe']:.2f}, skewness "
                           f"{vs['skewness']:.1f}, worst month {vs['worst_month']:.0%}"))
    plots.finish(fig, out / "fig19_5_decomposition.png")
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


P1, F2 = _num("+.1%"), _num(".2f")


def write_report(out, a, b, c, dd, sim, iv_shift):
    s = a["summary"]
    md = ["# Lab 19 results: Does Selling Options Earn a Reliable Premium?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**Claim on trial:** implied volatility exceeds later realized volatility on average, so option sellers earn "
           "a reliable premium.", "", "## Part A: The variance risk premium", "",
           f"Months: {s['months']} ({s['start']} to {s['end']}). Average VIX {s['mean_implied_vol']:.1%}; average "
           f"realized volatility over the next 21 trading days {s['mean_realized_vol']:.1%}.", "",
           f"- Average gap: {s['mean_gap_vol_points']:.2f} vol points (Newey–West t = {s['t_nw']:.2f}); median "
           f"{s['median_gap_vol_points']:.2f}.",
           f"- Implied above realized in {s['share_months_positive']:.0%} of months.",
           f"- Worst month: {s['worst_gap_vol_points']:.1f} points ({s['worst_month']}).",
           f"- Implied variance / realized variance: {s['implied_over_realized_variance']:.2f}.", "",
           "Realized volatility uses the CRSP value-weighted market (French), a close proxy for the S&P 500 on which "
           "the VIX is computed.", "", "![Lab Figure 19.1](fig19_1_vix_realized.png)", "", "![Lab Figure 19.2](fig19_2_gap.png)",
           "", "## Part B: Option-writing strategies", "",
           f"Rebuilt strategies price one-month options by Black–Scholes at the month-end VIX "
           f"{'' if iv_shift == 0 else f'shifted by {iv_shift:+.1f} points '}and hold them to the next month end. "
           "The at-the-money put-write has the same returns as an at-the-money buy-write (put–call parity); the "
           "buy-write shown sells a call 2% out of the money, like Cboe's BXY.", ""]
    t = b["table"]
    cols = ["strategy", "sample", "start", "excess_annual", "vol_annual", "sharpe", "skewness", "worst_month",
            "max_drawdown", "beta", "alpha_annual", "alpha_t"]
    md += [_md(t[cols].rename(columns={"strategy": "Strategy", "sample": "Sample", "start": "From",
                                       "excess_annual": "Excess return", "vol_annual": "Volatility", "sharpe": "Sharpe",
                                       "skewness": "Skewness", "worst_month": "Worst month",
                                       "max_drawdown": "Max drawdown", "beta": "Beta", "alpha_annual": "CAPM α",
                                       "alpha_t": "t"}),
               {"Excess return": P1, "Volatility": _num(".1%"), "Sharpe": F2, "Skewness": _num("+.2f"),
                "Worst month": _num(".1%"), "Max drawdown": _num(".0%"), "Beta": F2, "CAPM α": P1, "t": F2}), "",
           "![Lab Figure 19.3](fig19_3_cumulative.png)", "", "## Part C: Stress tests", "", "**Crisis periods (cumulative return)**",
           "", _md(c["crisis"], {k: _num("+.1%") for k in c["crisis"].columns if k != "period"}), "",
           "**The five worst market months in the common period**", "",
           _md(c["worst_months"].reset_index().rename(columns={"index": "Month", "date": "Month"}),
               {k: _num("+.1%") for k in c["worst_months"].columns}), "",
           "**Performance measures that account for skewness**", "",
           _md(c["measures"].rename(columns={"strategy": "Strategy", "sharpe": "Sharpe", "skewness": "Skewness",
                                             "mppm_3": "MPPM (ρ = 3)", "beta_down": "Down beta", "beta_up": "Up beta",
                                             "downside_capture": "Down capture", "upside_capture": "Up capture"}),
               {"Sharpe": F2, "Skewness": _num("+.2f"), "MPPM (ρ = 3)": P1, "Down beta": F2, "Up beta": F2,
                "Down capture": F2, "Up capture": F2}), "",
           "MPPM is the certainty-equivalent excess return of an investor with relative risk aversion 3 (Goetzmann et al. "
           "2007); it penalizes negative skewness that the Sharpe ratio ignores.", "", "![Lab Figure 19.4](fig19_4_payoff.png)",
           "", "## Part D: Equity exposure versus the volatility premium", "",
           _md(dd["decomposition"].rename(columns={"strategy": "Strategy", "excess_annual": "Excess return",
                                                   "from_beta": "From beta", "alpha_annual": "Alpha", "alpha_t": "t",
                                                   "beta": "Beta", "share_from_beta": "Share from beta",
                                                   "sharpe": "Sharpe", "sharpe_beta_matched": "Sharpe, beta-matched market",
                                                   "skew": "Skewness", "skew_beta_matched": "Skewness, beta-matched"}),
               {"Excess return": P1, "From beta": P1, "Alpha": P1, "t": F2, "Beta": F2, "Share from beta": _num(".0%"),
                "Sharpe": F2, "Sharpe, beta-matched market": F2, "Skewness": _num("+.2f"),
                "Skewness, beta-matched": _num("+.2f")}), ""]
    vs = dd["var_swap"]
    md += [f"**Pure variance premium.** Selling a one-month variance swap struck at the VIX each month, sized to 10% annual "
           f"volatility: mean {vs['mean_annual']:+.1%} a year, Sharpe {vs['sharpe']:.2f}, skewness {vs['skewness']:.1f}, "
           f"worst month {vs['worst_month']:.0%} ({vs['worst_month_date']}), maximum drawdown {vs['max_drawdown']:.0%}, "
           f"positive in {vs['share_months_positive']:.0%} of months, correlation with the market "
           f"{vs['corr_with_market']:+.2f}. VIX² approximates the variance swap rate; trading costs are ignored.", "",
           "![Lab Figure 19.5](fig19_5_decomposition.png)", "", "## Your verdict", "",
           "1. Is implied volatility reliably above realized volatility? How often, and by how much?",
           "2. Do put-writing and covered calls beat the market once you account for beta and skewness?",
           "3. How much of an option-income strategy's return is equity, and how much is a volatility premium?", ""]
    (out / "lab19_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--ff3-daily-file", default=None)
    ap.add_argument("--vix-file", default=None, help="FRED VIXCLS CSV")
    ap.add_argument("--put-file", default=None, help="Cboe PUT index history CSV")
    ap.add_argument("--bxm-file", default=None, help="Cboe BXM index history CSV")
    ap.add_argument("--iv-shift", type=float, default=0.0,
                    help="vol points added to the VIX when pricing the rebuilt options (e.g. -2 to remove skew)")
    ap.add_argument("--call-strike", type=float, default=1.02, help="buy-write call strike / index level")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"ff3": args.ff3_daily_file, "vix": args.vix_file, "put": args.put_file, "bxm": args.bxm_file}
    try:
        d = load(args.source, files, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    a = part_a(d)
    b = part_b(d, args.iv_shift, args.call_strike)
    c = part_c(b)
    dd = part_d(d, b, a)
    fig_vix(a, out, sim)
    fig_gap(a, out, sim)
    fig_cumulative(b, out, sim)
    fig_payoff(b, out, sim)
    fig_decomposition(dd, out, sim)
    b["table"].to_csv(out / "table_strategies.csv", index=False)
    b["syn"].to_csv(out / "table_rebuilt_monthly.csv")
    c["crisis"].to_csv(out / "table_crises.csv", index=False)
    c["measures"].to_csv(out / "table_measures.csv", index=False)
    dd["decomposition"].to_csv(out / "table_decomposition.csv", index=False)
    a["by_year"].to_csv(out / "table_vrp_by_year.csv")
    write_report(out, a, b, c, dd, sim, args.iv_shift)
    summary = {"simulated": sim, "vrp": a["summary"], "strategies": b["table"].to_dict(orient="records"),
               "decomposition": dd["decomposition"].to_dict(orient="records"), "var_swap": dd["var_swap"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    s = a["summary"]
    print(f"VRP: VIX {s['mean_implied_vol']:.1%} vs realized {s['mean_realized_vol']:.1%}; gap "
          f"{s['mean_gap_vol_points']:.2f} pts (t {s['t_nw']:.1f}); positive {s['share_months_positive']:.0%} of months")
    tc = b["table"][b["table"]["sample"] == "Common period"]
    print(tc[["strategy", "excess_annual", "sharpe", "skewness", "max_drawdown", "beta", "alpha_annual", "alpha_t",
              "mppm_3"]].round(3).to_string(index=False))
    print(f"Report: {out / 'lab19_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
