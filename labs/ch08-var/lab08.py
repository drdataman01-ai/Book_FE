"""Lab 8: Value at Risk and Its Backtests.

Chapter 8 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  How often losses exceed a 99% VaR, in calm markets and in crises, for six
  common models.

WHAT THE LAB DOES
  Part A  Backtest six VaR models on daily U.S. market returns from 1930:
          normal (250-day volatility), historical simulation, EWMA
          (RiskMetrics), GARCH with normal shocks, GJR-GARCH with Student-t
          shocks, and filtered historical simulation. Kupiec and
          Christoffersen tests; the Basel traffic light by year.
  Part B  Clustering: breaches after breaches, breaches per year.
  Part C  Stress tests: crises (1930-32, October 1987, 2008-09, 2020),
          confidence levels from 95% to 99.9%, and the historical window.
  Part D  Expected shortfall at 97.5%, the Basel measure since the
          Fundamental Review of the Trading Book: backtest on breach days.

DATA
  Kenneth R. French Data Library: Fama/French 3 Factors [Daily] (from 1926).
  The market return is Mkt-RF + RF.

USAGE
  python lab08.py
  python lab08.py --source sim
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
from fintest.risk import (
    christoffersen,
    es_backtest,
    es_normal,
    es_t,
    hits,
    historical_var_es,
    kupiec,
    var_normal,
    var_t,
    yearly_zones,
)
from fintest.vol import ewma_variance, fit_garch, garch_variance

ROOT = Path(__file__).resolve().parent
MODELS = ["Normal, 250-day", "Historical simulation", "EWMA (RiskMetrics)", "GARCH, normal", "GJR-GARCH, Student-t",
          "Filtered historical simulation"]
SHORT = dict(zip(MODELS, ["Normal", "HS", "EWMA", "GARCH", "GJR-t", "FHS"]))
COLORS = dict(zip(MODELS, [plots.INK_3, plots.ORANGE, plots.AQUA, plots.BLUE, plots.VIOLET, plots.MAGENTA]))
ALPHAS = [0.05, 0.025, 0.01, 0.005, 0.001]
CRISES = [("1930–32 Depression", "1930-01-01", "1932-12-31"), ("October 1987", "1987-10-01", "1987-10-31"),
          ("2008–09 financial crisis", "2008-09-01", "2009-03-31"), ("2020 pandemic", "2020-02-15", "2020-04-30")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(source: str, ff3_file=None, seed: int = 0) -> dict:
    if source == "sim":
        return simulated(seed)
    ff = load_french_factors("ff3_daily", path=ff3_file)
    cols = {c.strip().lower(): c for c in ff.columns}
    return {"ret": (ff[cols["mkt-rf"]] + ff[cols["rf"]]).dropna().rename("ret"), "simulated": False}


def simulated(seed: int = 0, n: int = 26000) -> dict:
    """GJR-GARCH daily returns with Student-t shocks (6 degrees of freedom). FOR TESTING ONLY."""
    rng = np.random.default_rng(seed)
    om, a, g, b = 1.2e-6, 0.03, 0.10, 0.905
    r, s2 = np.zeros(n), np.zeros(n)
    s2[0] = om / (1 - a - b - g / 2)
    z = rng.standard_t(6, n) / math.sqrt(1.5)
    for t in range(1, n):
        s2[t] = om + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * z[t]
    idx = pd.bdate_range("1926-07-01", periods=n)
    return {"ret": pd.Series(r, index=idx, name="ret"), "simulated": True}


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------


def _rolling_tail(z: np.ndarray, window: int, alphas) -> dict:
    """Rolling empirical quantile and tail mean of z over the last `window` values (inclusive), per alpha."""
    n = len(z)
    out = {a: (np.full(n, np.nan), np.full(n, np.nan)) for a in alphas}
    from numpy.lib.stride_tricks import sliding_window_view

    zz = np.where(np.isfinite(z), z, np.nan)
    for s in range(window - 1, n, 4000):
        e = min(s + 4000, n)
        W = sliding_window_view(zz[s - window + 1:e], window)
        W = np.sort(W, axis=1)  # NaNs sort to the end
        cnt = np.isfinite(W).sum(axis=1)
        for a in alphas:
            k = np.maximum(1, np.floor(a * cnt).astype(int))
            pos = np.clip(a * (cnt - 1), 0, None)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, cnt - 1)
            rows = np.arange(len(W))
            q = W[rows, lo] + (pos - lo) * (W[rows, hi] - W[rows, lo])
            cs = np.cumsum(np.nan_to_num(W), axis=1)
            tm = cs[rows, k - 1] / k
            good = cnt >= window // 2
            out[a][0][s:e] = np.where(good, q, np.nan)
            out[a][1][s:e] = np.where(good, tm, np.nan)
    return out


def forecasts(d: dict, start: str = "1930-01-01", fit_years: int = 10, refit_every: int = 252,
              hs_window: int = 250, fhs_window: int = 1000, alphas=ALPHAS) -> dict:
    """VaR (and ES) forecasts made at the end of each day for the next day, for every model and tail probability."""
    ret = d["ret"]
    r = ret.to_numpy(float)
    n = len(r)
    dates = ret.index
    first = int(np.searchsorted(dates, pd.Timestamp(start)))
    first = max(first, 750)
    # volatility forecasts for day t+1 made at the end of day t
    sd250 = pd.Series(r).rolling(250).apply(lambda x: math.sqrt(np.mean(x ** 2)), raw=True).to_numpy()
    ew = ewma_variance(r)
    ew_next = np.sqrt(0.94 * ew + 0.06 * r ** 2)
    sg, sj, nu = (np.full(n, np.nan) for _ in range(3))
    zj = np.full(n, np.nan)  # GJR standardized returns r_t / sigma_t (sigma_t forecast made at t-1)
    params = []
    win = 252 * fit_years
    t = first
    while t < n:
        end = min(t + refit_every, n)
        lo = max(0, t - win)
        pg = fit_garch(r[lo:t])
        pj = fit_garch(r[lo:t], gjr=True, dist="t")
        params.append({"refit_date": str(dates[t].date()), "garch_persistence": pg["persistence"],
                       "gjr_alpha": pj["alpha"], "gjr_gamma": pj["gamma"], "gjr_beta": pj["beta"], "gjr_nu": pj["nu"]})
        seg = r[lo:end]
        vg = garch_variance(seg, pg["omega"], pg["alpha"], pg["beta"])
        vj = garch_variance(seg, pj["omega"], pj["alpha"], pj["beta"], pj["gamma"])
        g_next = pg["omega"] + pg["alpha"] * seg ** 2 + pg["beta"] * vg
        j_next = pj["omega"] + (pj["alpha"] + pj["gamma"] * (seg < 0)) * seg ** 2 + pj["beta"] * vj
        i0 = t - lo
        sg[t:end] = np.sqrt(g_next[i0:])
        sj[t:end] = np.sqrt(j_next[i0:])
        nu[t:end] = pj["nu"]
        zj_seg = seg / np.sqrt(vj)
        if t == first:
            zj[lo:end] = zj_seg  # in-sample residuals seed the first FHS window
        else:
            zj[t:end] = zj_seg[i0:]
        t = end
    tails = _rolling_tail(zj, fhs_window, alphas)
    var, es = {}, {}
    for a in alphas:
        hs = historical_var_es(ret, hs_window, a)
        var[a] = {"Normal, 250-day": var_normal(sd250, a), "Historical simulation": hs["var"].to_numpy(),
                  "EWMA (RiskMetrics)": var_normal(ew_next, a), "GARCH, normal": var_normal(sg, a),
                  "GJR-GARCH, Student-t": _vt(sj, nu, a),
                  "Filtered historical simulation": -tails[a][0] * sj}
        es[a] = {"Normal, 250-day": es_normal(sd250, a), "Historical simulation": hs["es"].to_numpy(),
                 "EWMA (RiskMetrics)": es_normal(ew_next, a), "GARCH, normal": es_normal(sg, a),
                 "GJR-GARCH, Student-t": _et(sj, nu, a), "Filtered historical simulation": -tails[a][1] * sj}
        for m in MODELS:
            var[a][m] = np.asarray(var[a][m], float).copy()
            es[a][m] = np.asarray(es[a][m], float).copy()
            var[a][m][:first] = np.nan
            es[a][m][:first] = np.nan
    sig = {"Normal, 250-day": sd250, "Historical simulation": sd250, "EWMA (RiskMetrics)": ew_next,
           "GARCH, normal": sg, "GJR-GARCH, Student-t": sj, "Filtered historical simulation": sj}
    return {"var": var, "es": es, "sigma": sig, "ret": r, "dates": dates, "first": first,
            "params": pd.DataFrame(params)}


def _vt(s, nu, a):
    out = np.full(len(s), np.nan)
    ok = np.isfinite(s) & np.isfinite(nu)
    for v in np.unique(nu[ok]):
        m = ok & (nu == v)
        out[m] = var_t(s[m], v, a)
    return out


def _et(s, nu, a):
    out = np.full(len(s), np.nan)
    ok = np.isfinite(s) & np.isfinite(nu)
    for v in np.unique(nu[ok]):
        m = ok & (nu == v)
        out[m] = es_t(s[m], v, a)
    return out


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


def backtest(fc: dict, alpha: float = 0.01, mask: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    r = fc["ret"]
    for m in MODELS:
        h = hits(r, fc["var"][alpha][m])
        if mask is not None:
            h = np.where(mask, h, np.nan)
        if np.isfinite(h).sum() < (50 if mask is None else 10):
            continue
        uc, cc = kupiec(h, alpha), christoffersen(h, alpha)
        v = fc["var"][alpha][m]
        loss = -r[1:]
        brk = (h[1:] == 1)
        rows.append({"model": m, "days": uc["n"], "breaches": uc["breaches"], "expected": uc["expected"],
                     "rate": uc["rate"], "kupiec_p": uc["p"], "p_breach_after_breach": cc["p11"],
                     "p_breach_after_none": cc["p01"], "independence_p": cc["p_ind"], "cc_p": cc["p_cc"],
                     "avg_var": float(np.nanmean(np.where(np.isfinite(h), v, np.nan)[:-1])) if mask is None else
                     float(np.nanmean(v[:-1][np.isfinite(h[1:])])),
                     "worst_loss_over_var": float(np.nanmax(loss[brk] / v[:-1][brk])) if brk.any() else float("nan")})
    return pd.DataFrame(rows)


def part_a(fc: dict) -> dict:
    bt = backtest(fc, 0.01)
    zones = {}
    for m in MODELS:
        h = pd.Series(hits(fc["ret"], fc["var"][0.01][m]), index=fc["dates"])
        zones[m] = yearly_zones(h)
    bt["share_years_yellow"] = [float((zones[m].zone == "yellow").mean()) for m in bt.model]
    bt["share_years_red"] = [float((zones[m].zone == "red").mean()) for m in bt.model]
    return {"table": bt, "zones": zones}


def part_b(fc: dict) -> dict:
    per_year = {}
    for m in MODELS:
        h = pd.Series(hits(fc["ret"], fc["var"][0.01][m]), index=fc["dates"]).dropna()
        per_year[m] = h.groupby(h.index.year).sum()
    py = pd.DataFrame(per_year)
    py.index = py.index.astype(int)
    worst = py.sum(axis=1).sort_values(ascending=False).head(8).index
    return {"per_year": py, "worst_years": py.loc[sorted(worst)]}


def part_c(fc: dict, d: dict, hs_windows=(250, 500, 1000)) -> dict:
    crows = []
    for label, s, e in CRISES:
        m = (fc["dates"] >= s) & (fc["dates"] <= e)
        bt = backtest(fc, 0.01, m)
        if bt.empty:
            continue
        for _, row in bt.iterrows():
            crows.append({"period": label, "model": row.model, "days": row.days, "breaches": row.breaches,
                          "expected": row.expected})
    levels = []
    for a in ALPHAS:
        bt = backtest(fc, a)
        for _, row in bt.iterrows():
            levels.append({"confidence": 1 - a, "model": row.model, "rate": row.rate, "ratio": row.rate / a,
                           "kupiec_p": row.kupiec_p})
    wrows = []
    r = fc["ret"]
    for w in hs_windows:
        v = historical_var_es(d["ret"], w, 0.01)["var"].to_numpy().copy()
        v[:fc["first"]] = np.nan
        h = hits(r, v)
        uc, cc = kupiec(h, 0.01), christoffersen(h, 0.01)
        wrows.append({"window_days": w, "breaches": uc["breaches"], "rate": uc["rate"], "kupiec_p": uc["p"],
                      "p_breach_after_breach": cc["p11"], "independence_p": cc["p_ind"]})
    return {"crises": pd.DataFrame(crows), "levels": pd.DataFrame(levels), "windows": pd.DataFrame(wrows)}


def part_d(fc: dict, alpha: float = 0.025) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        e = es_backtest(fc["ret"], fc["var"][alpha][m], fc["es"][alpha][m], fc["sigma"][m])
        e_raw = es_backtest(fc["ret"], fc["var"][alpha][m], fc["es"][alpha][m])
        rows.append({"model": m, "breach_days": e["n"], "tail_loss_over_es": e_raw["ratio"],
                     "mean_std_exceedance": e["mean_resid"], "t": e["t"], "p": e["p"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_path(fc, out, sim, start="2007-01-01", end="2010-12-31"):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    dates = fc["dates"]
    m = (dates >= start) & (dates <= end)
    if m.sum() < 100:
        m = np.zeros(len(dates), bool)
        m[-1000:] = True
    idx = np.flatnonzero(m)
    loss = -fc["ret"]
    ax.bar(dates[idx], loss[idx], width=1.5, color=plots.INK_3, alpha=0.6, label="Daily loss (gain below zero)")
    for mname in ("Historical simulation", "GJR-GARCH, Student-t"):
        v = fc["var"][0.01][mname]
        prev = v[np.maximum(idx - 1, 0)]  # forecast for each day, made the day before
        ax.plot(dates[idx], prev, color=COLORS[mname], linewidth=1.4, label=f"99% VaR, {mname}")
        brk = loss[idx] > prev
        ax.plot(dates[idx][brk], loss[idx][brk], "o", color=COLORS[mname], markersize=3.5)
    plots.percent_axis(ax)
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 8.1  Two VaR models through 2007–2010",
                 _sub(sim, "Dots mark days when the loss exceeded the VaR forecast made the day before"))
    plots.finish(fig, out / "fig8_1_path.png")
    plt.close(fig)


def fig_rates(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = a["table"]
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["rate"], color=[COLORS[m] for m in t.model])
    ax.axvline(0.01, color=plots.INK, linestyle="--", linewidth=1.2)
    ax.text(0.0102, y.max() + 0.45, "Target: 1%", fontsize=8.5, color=plots.INK)
    for yy, (_, row) in zip(y, t.iterrows()):
        ax.text(row.rate + 0.0003, yy, f"{row.rate:.2%}  (Kupiec p = {row.kupiec_p:.2f})", va="center", fontsize=8.5,
                color=plots.INK_2)
    ax.set_yticks(y, t["model"], fontsize=9)
    ax.set_xlim(0, max(0.03, t["rate"].max() * 1.6))
    plots.percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of days on which the loss exceeded the 99% VaR")
    plots.titles(fig, "Lab Figure 8.2  How often is a 99% VaR breached?",
                 _sub(sim, f"Daily U.S. market returns, {fc_period(a)}"))
    plots.finish(fig, out / "fig8_2_rates.png")
    plt.close(fig)


def fc_period(a):
    return a.get("period", "")


def fig_per_year(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    py = b["per_year"]
    for mname in ("Normal, 250-day", "Historical simulation", "GJR-GARCH, Student-t"):
        ax.plot(py.index, py[mname], color=COLORS[mname], linewidth=1.3, label=mname,
                marker="o" if mname != "Normal, 250-day" else None, markersize=2.5)
    ax.axhspan(4.5, 9.5, color=plots.YELLOW, alpha=0.12, linewidth=0)
    ax.axhspan(9.5, max(10.5, py.max().max() + 1), color=plots.RED, alpha=0.08, linewidth=0)
    from matplotlib.patches import Patch

    ax.set_ylabel("Breaches of the 99% VaR per year")
    ax.set_ylim(0, max(12, py.max().max() + 1))
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=plots.YELLOW, alpha=0.25), Patch(color=plots.RED, alpha=0.18)]
    labels += ["Basel yellow zone (5–9)", "Basel red zone (10+)"]
    ax.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper right")
    plots.titles(fig, "Lab Figure 8.3  Breaches come in bunches",
                 _sub(sim, "About 2.5 breaches a year are expected; green zone 0–4, yellow 5–9, red 10 or more"))
    plots.finish(fig, out / "fig8_3_per_year.png")
    plt.close(fig)


def fig_levels(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    lv = c["levels"]
    confs = sorted(lv.confidence.unique())
    x = np.arange(len(confs))
    for mname in MODELS:
        g = lv[lv.model == mname].set_index("confidence").reindex(confs)
        ax.plot(x, g["ratio"], marker="o", color=COLORS[mname], linewidth=1.4, label=mname, markersize=4)
    ax.axhline(1, color=plots.INK, linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"{c:.1%}".replace(".0%", "%") for c in confs])
    ax.set_yscale("log")
    import matplotlib.ticker as mt

    lo, hi = np.nanmin(lv["ratio"]), np.nanmax(lv["ratio"])
    ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20, 30) if lo * 0.8 <= t <= hi * 1.25] or [1]
    ax.yaxis.set_major_locator(mt.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("VaR confidence level")
    ax.set_ylabel("Actual / expected breaches (log)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    plots.titles(fig, "Lab Figure 8.4  The further into the tail, the worse the models do",
                 _sub(sim, "1× means the breach rate matched the tail probability"))
    plots.finish(fig, out / "fig8_4_levels.png")
    plt.close(fig)


def fig_es(dd, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    y = np.arange(len(dd))[::-1]
    ax.barh(y, dd["tail_loss_over_es"], color=[COLORS[m] for m in dd.model])
    ax.axvline(1, color=plots.INK, linestyle="--", linewidth=1.2)
    for yy, (_, row) in zip(y, dd.iterrows()):
        ax.text(max(row.tail_loss_over_es, 1.0) + 0.02, yy, f"{row.tail_loss_over_es:.2f}  (t = {row.t:.1f})", va="center",
                fontsize=8.5, color=plots.INK_2)
    ax.set_yticks(y, dd["model"], fontsize=9)
    ax.set_xlim(0, max(1.6, dd["tail_loss_over_es"].max() * 1.25))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average loss on breach days ÷ average predicted ES (1 = accurate)")
    plots.titles(fig, "Lab Figure 8.5  Expected shortfall at 97.5%: how bad are the bad days?",
                 _sub(sim, "t tests whether exceedances beyond the ES, scaled by volatility, average zero"))
    plots.finish(fig, out / "fig8_5_es.png")
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


def write_report(out, fc, a, b, c, dd, sim):
    p = a["period"]
    md = ["# Lab 8 results: Value at Risk and Its Backtests", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** how often losses exceed a 99% VaR, in calm markets and in crises, for six common models.", "",
           f"Backtest period: {p}. Forecasts use only data available the day before.", "",
           "## Part A: 99% VaR backtests", "",
           _md(a["table"][["model", "days", "breaches", "expected", "rate", "kupiec_p", "independence_p", "cc_p",
                           "avg_var", "share_years_yellow", "share_years_red", "worst_loss_over_var"]].rename(columns={
               "model": "Model", "days": "Days", "breaches": "Breaches", "expected": "Expected", "rate": "Rate",
               "kupiec_p": "Kupiec p", "independence_p": "Independence p", "cc_p": "Cond. coverage p",
               "avg_var": "Average VaR", "share_years_yellow": "Years yellow", "share_years_red": "Years red",
               "worst_loss_over_var": "Worst loss / VaR"}),
               {"Expected": _num(".0f"), "Rate": _num(".2%"), "Kupiec p": _num(".3f"), "Independence p": _num(".3f"),
                "Cond. coverage p": _num(".3f"), "Average VaR": _num(".2%"), "Years yellow": _num(".0%"),
                "Years red": _num(".0%"), "Worst loss / VaR": _num(".1f")}), "",
           "Years yellow/red: share of calendar years in the Basel yellow (5–9 breaches per 250 days) or red (10 or more) "
           "zone.", "", "![Lab Figure 8.1](fig8_1_path.png)", "", "![Lab Figure 8.2](fig8_2_rates.png)", "",
           "## Part B: Clustering", "",
           _md(a["table"][["model", "p_breach_after_none", "p_breach_after_breach"]].rename(columns={
               "model": "Model", "p_breach_after_none": "P(breach | no breach yesterday)",
               "p_breach_after_breach": "P(breach | breach yesterday)"}),
               {"P(breach | no breach yesterday)": _num(".2%"), "P(breach | breach yesterday)": _num(".1%")}), "",
           "**Years with the most breaches**", "",
           _md(b["worst_years"].rename(columns=SHORT).rename_axis("Year").reset_index().astype({"Year": str}),
               {k: _num(".0f") for k in SHORT.values()}), "", "![Lab Figure 8.3](fig8_3_per_year.png)", "",
           "## Part C: Stress tests", "", "**Crises (99% VaR)**", "",
           _md(c["crises"].pivot_table(index="period", columns="model", values="breaches", sort=False)[MODELS].rename(
               columns=SHORT).reset_index().merge(
               c["crises"].groupby("period", sort=False)[["days", "expected"]].first().reset_index(), on="period").rename(
               columns={"period": "Period", "days": "Days", "expected": "Expected"}),
               {**{k: _num(".0f") for k in SHORT.values()}, "Expected": _num(".1f")}), "",
           "**Confidence levels: actual / expected breaches**", "",
           _md(c["levels"].pivot_table(index="confidence", columns="model", values="ratio")[MODELS].rename(
               columns=SHORT).reset_index().rename(columns={"confidence": "Confidence"}),
               {**{k: _num(".2f") for k in SHORT.values()}, "Confidence": _num(".1%")}), "",
           "**Historical-simulation window (99% VaR)**", "",
           _md(c["windows"].rename(columns={"window_days": "Window (days)", "breaches": "Breaches", "rate": "Rate",
                                            "kupiec_p": "Kupiec p", "p_breach_after_breach": "P(breach | breach)",
                                            "independence_p": "Independence p"}),
               {"Window (days)": _num(".0f"), "Breaches": _num(".0f"), "Rate": _num(".2%"), "Kupiec p": _num(".3f"),
                "P(breach | breach)": _num(".1%"),
                "Independence p": _num(".3f")}), "",
           "![Lab Figure 8.4](fig8_4_levels.png)", "", "## Part D: Expected shortfall at 97.5%", "",
           _md(dd.rename(columns={"model": "Model", "breach_days": "Breach days", "tail_loss_over_es": "Tail loss / ES",
                                  "mean_std_exceedance": "Mean scaled exceedance", "t": "t", "p": "p (one-sided)"}),
               {"Tail loss / ES": _num(".2f"), "Mean scaled exceedance": _num("+.3f"), "t": _num(".2f"),
                "p (one-sided)": _num(".3f")}), "",
           "A positive t (small p) means the model understated the average loss on its worst days.", "",
           "![Lab Figure 8.5](fig8_5_es.png)", "", "## Questions to think about", "",
           "1. Which models deliver about 1% breaches at 99%? Which pass the independence test?",
           "2. How do the models fare in crises and deeper in the tail (99.5%, 99.9%)?",
           "3. Does expected shortfall fix the weaknesses of VaR, or inherit them?", ""]
    (out / "lab08_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--ff3-daily-file", default=None)
    ap.add_argument("--start", default="1930-01-01", help="first day of the backtest")
    ap.add_argument("--fit-years", type=int, default=10, help="GARCH estimation window")
    ap.add_argument("--hs-window", type=int, default=250)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, args.ff3_daily_file, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    fc = forecasts(d, args.start, args.fit_years, hs_window=args.hs_window)
    a = part_a(fc)
    a["period"] = f"{fc['dates'][fc['first'] + 1].date()} to {fc['dates'][-1].date()}"
    b = part_b(fc)
    c = part_c(fc, d)
    dd = part_d(fc)
    fig_path(fc, out, sim)
    fig_rates(a, out, sim)
    fig_per_year(b, out, sim)
    fig_levels(c, out, sim)
    fig_es(dd, out, sim)
    a["table"].to_csv(out / "table_backtest_99.csv", index=False)
    b["per_year"].to_csv(out / "table_breaches_per_year.csv")
    for k in ("crises", "levels", "windows"):
        c[k].to_csv(out / f"table_{k}.csv", index=False)
    dd.to_csv(out / "table_es_backtest.csv", index=False)
    fc["params"].to_csv(out / "table_garch_parameters.csv", index=False)
    write_report(out, fc, a, b, c, dd, sim)
    summary = {"simulated": sim, "period": a["period"], "backtest_99": a["table"].to_dict(orient="records"),
               "levels": c["levels"].to_dict(orient="records"), "es": dd.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a["table"][["model", "breaches", "expected", "rate", "kupiec_p", "independence_p"]].round(4).to_string(index=False))
    print(dd[["model", "tail_loss_over_es", "t"]].round(3).to_string(index=False))
    print(f"Report: {out / 'lab08_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
