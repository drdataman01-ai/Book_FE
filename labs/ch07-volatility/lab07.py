"""Lab 7: Forecasting Volatility.

Chapter 7 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "Today's volatility forecasts tomorrow's, even though returns themselves
   are close to unforecastable."

WHAT THE LAB DOES
  Part A  Stylized facts: returns have little memory; their squares have a
          lot. The leverage effect.
  Part B  Forecasts of U.S. market variance over the next 1, 5 and 22 trading
          days from a random walk, the long-run average, EWMA, GARCH, GJR,
          HAR and (from 1990) the VIX, walk-forward with yearly re-estimation;
          MSE, QLIKE, Diebold-Mariano tests and Mincer-Zarnowitz regressions.
  Part C  Stress tests: 2008-09, 2020 and 2022.
  Part D  Volatility-managed market exposure (Moreira and Muir 2017), with
          costs and a leverage cap.

DATA
  Kenneth R. French Data Library: Fama/French 3 Factors [Daily] (from 1926).
  FRED: VIXCLS (CBOE Volatility Index, from 1990), optional.

USAGE
  python lab07.py
  python lab07.py --source sim
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
from fintest.forecast import diebold_mariano
from fintest.inference import acf
from fintest.vol import (
    ewma_variance,
    fit_garch,
    future_sum,
    garch_forecast_sum,
    garch_variance,
    har_features,
    load_fred,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
)

ROOT = Path(__file__).resolve().parent
HORIZONS = (1, 5, 22)
MODELS = ["Random walk", "Long-run average", "EWMA", "GARCH", "GJR", "HAR", "VIX"]
COLORS = dict(zip(MODELS, [plots.INK_3, plots.INK_2, plots.AQUA, plots.BLUE, plots.VIOLET, plots.ORANGE, plots.RED]))
CRISES = [("2008–09 financial crisis", "2008-09-01", "2009-06-30"), ("2020 pandemic", "2020-02-15", "2020-06-30"),
          ("2022 rate shock", "2022-01-01", "2022-12-31")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(source: str, ff3_file=None, vix_file=None, use_vix=True, seed=0) -> dict:
    if source == "sim":
        return simulated(seed)
    ff = load_french_factors("ff3_daily", path=ff3_file)
    cols = {c.strip().lower(): c for c in ff.columns}
    r = (ff[cols["mkt-rf"]] + ff[cols["rf"]]).rename("ret")
    vix = None
    if use_vix:
        try:
            vix = load_fred("VIXCLS", vix_file)
        except DataUnavailable as err:
            print(f"VIX not loaded ({str(err).splitlines()[0]}); continuing without it.")
    return {"ret": r.dropna(), "rf": ff[cols["rf"]], "vix": vix, "simulated": False}


def simulated(seed=0, n=26000) -> dict:
    """GJR-GARCH daily returns with t innovations and a VIX-like series. FOR TESTING ONLY."""
    rng = np.random.default_rng(seed)
    om, a, g, b = 1.2e-6, 0.03, 0.10, 0.905
    r, s2 = np.zeros(n), np.zeros(n)
    s2[0] = om / (1 - a - b - g / 2)
    z = rng.standard_t(6, n) / math.sqrt(1.5)
    for t in range(1, n):
        s2[t] = om + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = 0.0003 + math.sqrt(s2[t]) * z[t]
    idx = pd.bdate_range("1926-07-01", periods=n)
    s2_next = om + (a + g * (r < 0)) * r ** 2 + b * s2
    vix = pd.Series(100 * np.sqrt(s2_next * 252) * 1.15 * np.exp(0.05 * rng.standard_normal(n)), index=idx)
    vix = vix[vix.index >= "1990-01-02"]
    return {"ret": pd.Series(r, index=idx, name="ret"), "rf": pd.Series(0.0001, index=idx), "vix": vix,
            "simulated": True}


# ---------------------------------------------------------------------------
# Part A: stylized facts
# ---------------------------------------------------------------------------


def part_a(d: dict, nlags: int = 50) -> dict:
    r = d["ret"].to_numpy()
    x = r - r.mean()
    lev = [float(np.corrcoef(x[:-k], np.abs(x[k:]))[0, 1]) for k in range(1, 11)]
    return {"acf_r": acf(x, nlags), "acf_r2": acf(x ** 2, nlags), "acf_abs": acf(np.abs(x), nlags), "leverage": lev}


# ---------------------------------------------------------------------------
# Part B: forecasts
# ---------------------------------------------------------------------------


def forecasts(d: dict, start: str = "1990-01-01", window_years: int = 20, refit_every: int = 252) -> dict:
    """Walk-forward variance forecasts for each horizon. Parameters re-estimated about yearly on a rolling window."""
    ret = d["ret"]
    r = (ret - ret.mean()).to_numpy()
    dates = ret.index
    n = len(r)
    first = int(np.searchsorted(dates, pd.Timestamp(start)))
    win = 252 * window_years
    r2 = r ** 2
    feats = har_features(r2)
    out = {h: {m: np.full(n, np.nan) for m in MODELS} for h in HORIZONS}
    targets = {h: future_sum(r2, h) for h in HORIZONS}
    ew = ewma_variance(r)
    ew_next = 0.94 * ew + 0.06 * r2  # forecast for t+1 made at the end of t
    cum = np.concatenate([[0.0], np.cumsum(r2)])
    params_log = []
    t = first
    while t < n:
        end = min(t + refit_every, n)
        lo = max(0, t - win)
        hist = r[lo:t]
        pg = fit_garch(hist, gjr=False)
        pj = fit_garch(hist, gjr=True)
        params_log.append({"refit_date": str(dates[t].date()), **{f"garch_{k}": pg[k] for k in ("alpha", "beta", "persistence")},
                           **{f"gjr_{k}": pj[k] for k in ("alpha", "gamma", "beta", "persistence")}})
        seg = r[lo:end]
        sg = garch_variance(seg, pg["omega"], pg["alpha"], pg["beta"])
        sj = garch_variance(seg, pj["omega"], pj["alpha"], pj["beta"], pj["gamma"])
        # one-day-ahead variance for t+1 made at the end of t: apply one more recursion step
        segr = seg
        g_next = pg["omega"] + pg["alpha"] * segr ** 2 + pg["beta"] * sg
        j_next = pj["omega"] + (pj["alpha"] + pj["gamma"] * (segr < 0)) * segr ** 2 + pj["beta"] * sj
        for h in HORIZONS:
            # HAR on the training window: target = average daily variance over the next h days
            X = feats[lo:t]
            y = targets[h][lo:t] / h
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            ok[-h:] = False  # targets not yet observed at t
            A = np.column_stack([np.ones(ok.sum()), X[ok]])
            bh = np.linalg.lstsq(A, y[ok], rcond=None)[0]
            floor = 0.1 * float(np.mean(r2[lo:t]))  # linear HAR can go near zero after calm days
            for k in range(t, end):
                i = k - lo
                out[h]["GARCH"][k] = garch_forecast_sum(g_next[i], pg, h)
                out[h]["GJR"][k] = garch_forecast_sum(j_next[i], pj, h)
                f = bh[0] + feats[k] @ bh[1:]
                out[h]["HAR"][k] = max(f, floor) * h
        t = end
    for h in HORIZONS:
        k = np.arange(n)
        # last 22 days' average squared return, scaled to h days ("last month's volatility continues");
        # a single squared return is too noisy a benchmark, and QLIKE punishes forecasts near zero
        past = np.where(k >= 21, (cum[np.minimum(k + 1, n)] - cum[np.maximum(k + 1 - 22, 0)]) / 22 * h, np.nan)
        out[h]["Random walk"] = past
        out[h]["Long-run average"] = cum[k + 1] / (k + 1) * h
        out[h]["EWMA"] = ew_next * h
        if d.get("vix") is not None:
            v = d["vix"].reindex(dates).ffill(limit=3).to_numpy()
            out[h]["VIX"] = (v / 100) ** 2 / 252 * h
        for m in MODELS:
            out[h][m][:first] = np.nan
    return {"forecasts": out, "targets": targets, "dates": dates, "first": first, "params": pd.DataFrame(params_log)}


def evaluate(fc: dict, mask: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        y = fc["targets"][h]
        avail = [m for m in MODELS if np.isfinite(fc["forecasts"][h][m]).sum() > 0]
        ok = np.isfinite(y) & np.all([np.isfinite(fc["forecasts"][h][m]) for m in avail], axis=0)
        ok &= np.all([fc["forecasts"][h][m] > 0 for m in avail], axis=0)
        if mask is not None:
            ok &= mask
        if ok.sum() < 20:
            continue
        rw_q = qlike_loss(y[ok], fc["forecasts"][h]["Random walk"][ok])
        rw_m = mse_loss(y[ok], fc["forecasts"][h]["Random walk"][ok])
        for m in avail:
            f = fc["forecasts"][h][m][ok]
            q, s = qlike_loss(y[ok], f), mse_loss(y[ok], f)
            mz = mincer_zarnowitz(y[ok], f)
            dm = diebold_mariano(y[ok], fc["forecasts"][h]["Random walk"][ok], f, lags=h - 1) if m != "Random walk" \
                else {"stat": float("nan")}
            dq = rw_q - q
            tq = dq.mean() / (dq.std(ddof=1) / math.sqrt(len(dq))) if m != "Random walk" else float("nan")
            rows.append({"horizon": h, "model": m, "qlike_gain_vs_rw": float(dq.mean()), "qlike_t": tq,
                         "mse_ratio_vs_rw": float(s.mean() / rw_m.mean()), "mse_dm_t": dm["stat"],
                         "mz_slope": mz["slope"], "mz_r2": mz["r2"], "n": int(ok.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part D: volatility-managed portfolio
# ---------------------------------------------------------------------------


def vol_managed(d: dict, cap: float | None = None, cost: float = 0.001) -> dict:
    ret = d["ret"]
    rf = d["rf"].reindex(ret.index).fillna(0)
    ex = ret - rf
    m_ex = (1 + ex).groupby(ex.index.to_period("M")).prod() - 1
    rv = (ex ** 2).groupby(ex.index.to_period("M")).sum()
    w_raw = 1 / rv.shift(1)
    c = m_ex.std() / (w_raw * m_ex).std()  # scale so the managed portfolio has the market's volatility
    w = c * w_raw
    if cap:
        w = w.clip(upper=cap)
    managed = (w * m_ex).dropna()
    turnover = w.diff().abs().reindex(managed.index).fillna(0)
    managed_net = managed - turnover * cost
    mkt = m_ex.reindex(managed.index)
    from fintest.stats import ols_newey_west

    def st(x):
        v = x.dropna().to_numpy()
        wealth = (1 + x.dropna()).cumprod()
        return {"mean_annual": 12 * v.mean(), "vol_annual": v.std(ddof=1) * math.sqrt(12),
                "sharpe": v.mean() / v.std(ddof=1) * math.sqrt(12), "max_drawdown": float((wealth / wealth.cummax() - 1).min())}

    reg = ols_newey_west(managed_net.to_numpy(), mkt.to_numpy())
    rows = [{"portfolio": "Market", **st(mkt)}, {"portfolio": "Volatility-managed (gross)", **st(managed)},
            {"portfolio": f"Volatility-managed, net of {cost * 1e4:.0f} bp per unit of weight change", **st(managed_net)}]
    return {"table": pd.DataFrame(rows), "alpha_annual": 12 * float(reg["coef"][0, 0]), "alpha_t": float(reg["t"][0, 0]),
            "beta": float(reg["coef"][1, 0]), "avg_weight": float(w.mean()), "max_weight": float(w.max()),
            "managed": managed_net, "market": mkt, "weights": w}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_acf(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    k = np.arange(1, len(a["acf_r"]) + 1)
    ax.bar(k - 0.2, a["acf_r"], width=0.4, color=plots.INK_3, label="Returns")
    ax.bar(k + 0.2, a["acf_abs"], width=0.4, color=plots.BLUE, label="Absolute returns")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend(frameon=False, fontsize=8.5)
    plots.titles(fig, "Lab Figure 7.1  Returns forget; volatility remembers",
                 _sub(sim, f"Daily U.S. market returns · correlation of today's return with the next day's absolute "
                           f"return (leverage effect): {a['leverage'][0]:+.2f}"))
    plots.finish(fig, out / "fig7_1_acf.png")
    plt.close(fig)


def fig_paths(fc, out, sim, start="2007-01-01", end="2010-12-31", h=22):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    dates = fc["dates"]
    m = (dates >= start) & (dates <= end)
    if m.sum() < 30:
        m = np.zeros(len(dates), bool)
        m[-1000:] = True
    ann = lambda v: np.sqrt(np.asarray(v) / h * 252)  # noqa: E731
    ax.plot(dates[m], ann(fc["targets"][h][m]), color=plots.INK_3, linewidth=0.8, label="Realized (next 22 days)")
    for mname in ("GARCH", "HAR", "VIX"):
        f = fc["forecasts"][h][mname][m]
        if np.isfinite(f).any():
            ax.plot(dates[m], ann(f), color=COLORS[mname], linewidth=1.4, label=mname)
    plots.percent_axis(ax)
    ax.set_ylabel("Annualized volatility")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 7.2  Forecasts track volatility, with a lag",
                 _sub(sim, "One-month-ahead volatility forecasts made each day, against the volatility that followed"))
    plots.finish(fig, out / "fig7_2_paths.png")
    plt.close(fig)


def fig_losses(ev, out, sim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=False)
    fig.patch.set_facecolor(plots.SURFACE)
    for ax, h in zip(axes, HORIZONS):
        plots.style_axes(ax)
        g = ev[(ev.horizon == h) & (ev.model != "Random walk")]
        y = np.arange(len(g))[::-1]
        ax.barh(y, g["qlike_gain_vs_rw"], color=[COLORS[m] for m in g["model"]])
        ax.set_yticks(y, g["model"], fontsize=8.5)
        ax.axvline(0, color=plots.INK_3, linewidth=1)
        ax.set_title(f"{h}-day horizon", loc="left", fontsize=10, color=plots.INK)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Average QLIKE improvement over the random walk (higher is better)")
    plots.titles(fig, "Lab Figure 7.3  Which forecasts beat yesterday's volatility?",
                 _sub(sim, "Walk-forward forecasts of U.S. market variance from 1990 · QLIKE loss (Patton 2011)"))
    plots.finish(fig, out / "fig7_3_losses.png")
    plt.close(fig)


def fig_crises(crisis, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = crisis[(crisis.horizon == 22) & (crisis.model != "Random walk")]
    periods = list(dict.fromkeys(t["period"]))
    models = list(dict.fromkeys(t["model"]))
    width = 0.8 / len(models)
    for j, mname in enumerate(models):
        g = t[t.model == mname].set_index("period").reindex(periods)
        ax.bar(np.arange(len(periods)) + (j - len(models) / 2 + 0.5) * width, g["qlike_gain_vs_rw"], width=width,
               color=COLORS[mname], label=mname)
    ax.set_xticks(range(len(periods)), periods, fontsize=9)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_ylabel("QLIKE improvement over random walk")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    plots.titles(fig, "Lab Figure 7.4  Forecasting volatility in crises",
                 _sub(sim, "One-month horizon · positive: better than assuming the last month's volatility continues"))
    plots.finish(fig, out / "fig7_4_crises.png")
    plt.close(fig)


def fig_managed(vm, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    dates = vm["managed"].index.to_timestamp()
    ax.plot(dates, (1 + vm["market"]).cumprod(), color=plots.INK_3, linewidth=1.5, label="Market (excess return)")
    ax.plot(dates, (1 + vm["managed"]).cumprod(), color=plots.BLUE, linewidth=1.5, label="Volatility-managed, net")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylabel("Growth of $1 (log)")
    plots.titles(fig, "Lab Figure 7.5  Holding less of the market when it is volatile",
                 _sub(sim, f"Weight ∝ 1 / last month's realized variance, scaled to the market's volatility · alpha "
                           f"{vm['alpha_annual']:+.1%} a year (t = {vm['alpha_t']:.1f}) · average weight "
                           f"{vm['avg_weight']:.2f}, maximum {vm['max_weight']:.1f}"))
    plots.finish(fig, out / "fig7_5_managed.png")
    plt.close(fig)


def _md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


def write_report(out, a, fc, ev, crisis, vm, sim, has_vix):
    md = ["# Lab 7 results: Forecasting Volatility", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**Claim on trial:** today's volatility forecasts tomorrow's, even though returns are close to "
           "unforecastable.", "",
           "## Part A: Stylized facts", "",
           f"- Autocorrelation of daily returns at lag 1: {a['acf_r'][0]:+.3f}; average over lags 1–10: "
           f"{np.mean(a['acf_r'][:10]):+.3f}.",
           f"- Autocorrelation of absolute returns at lag 1: {a['acf_abs'][0]:+.3f}; at lag 50: {a['acf_abs'][49]:+.3f}.",
           f"- Leverage effect: correlation of today's return with tomorrow's absolute return {a['leverage'][0]:+.3f}.",
           "", "![Lab Figure 7.1](fig7_1_acf.png)", ""]
    if not has_vix:
        md += ["> The VIX was not loaded, so it is missing from the comparison below.", ""]
    md += ["## Part B: Forecast comparison", "",
           f"Walk-forward from {fc['dates'][fc['first']].date()}; GARCH, GJR and HAR re-estimated yearly on the "
           "previous 20 years. Targets: the sum of squared daily returns over the next h days.", "",
           _md(ev.rename(columns={"horizon": "Horizon (days)", "model": "Model", "qlike_gain_vs_rw": "QLIKE gain vs RW",
                                  "qlike_t": "t", "mse_ratio_vs_rw": "MSE / RW MSE", "mse_dm_t": "DM t (MSE)",
                                  "mz_slope": "MZ slope", "mz_r2": "MZ R²", "n": "Days"}),
               {"QLIKE gain vs RW": _num("+.3f"), "t": _num(".1f"), "MSE / RW MSE": _num(".2f"), "DM t (MSE)": _num(".1f"),
                "MZ slope": _num(".2f"), "MZ R²": _num(".2f")}), "",
           "RW = random walk (the last 22 days' realized variance, scaled to h days). MZ = Mincer–Zarnowitz regression of realized on "
           "forecast variance; an unbiased forecast has slope 1.", "",
           "![Lab Figure 7.2](fig7_2_paths.png)", "", "![Lab Figure 7.3](fig7_3_losses.png)", ""]
    md += ["## Part C: Crises", "",
           _md(crisis[crisis.horizon == 22][["period", "model", "qlike_gain_vs_rw", "mz_slope", "n"]].rename(columns={
               "period": "Period", "model": "Model", "qlike_gain_vs_rw": "QLIKE gain vs RW", "mz_slope": "MZ slope",
               "n": "Days"}), {"QLIKE gain vs RW": _num("+.3f"), "MZ slope": _num(".2f")}), "",
           "![Lab Figure 7.4](fig7_4_crises.png)", ""]
    md += ["## Part D: Volatility-managed portfolio", "",
           _md(vm["table"], {"mean_annual": _num("+.1%"), "vol_annual": _num(".1%"), "sharpe": _num(".2f"),
                             "max_drawdown": _num(".0%")}), "",
           f"Alpha of the net managed portfolio against the market: {vm['alpha_annual']:+.1%} a year (t = "
           f"{vm['alpha_t']:.2f}), beta {vm['beta']:.2f}. Average weight {vm['avg_weight']:.2f}, maximum "
           f"{vm['max_weight']:.1f}.", "", "![Lab Figure 7.5](fig7_5_managed.png)", ""]
    md += ["## Your verdict", "",
           "1. Does past volatility forecast future volatility better than a constant? At which horizons?",
           "2. Do the more complex models beat a simple EWMA or random walk? Does the VIX add information?",
           "3. Does volatility timing improve a market investor's results after costs and with realistic leverage?", ""]
    (out / "lab07_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--ff3-daily-file", default=None)
    ap.add_argument("--vix-file", default=None, help="FRED VIXCLS CSV")
    ap.add_argument("--no-vix", action="store_true")
    ap.add_argument("--start", default="1990-01-01", help="first forecast date")
    ap.add_argument("--window-years", type=int, default=20)
    ap.add_argument("--weight-cap", type=float, default=None)
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, args.ff3_daily_file, args.vix_file, not args.no_vix, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    has_vix = d.get("vix") is not None and len(d["vix"]) > 0
    a = part_a(d)
    fc = forecasts(d, args.start, args.window_years)
    ev = evaluate(fc)
    crows = []
    for label, s, e in CRISES:
        m = (fc["dates"] >= s) & (fc["dates"] <= e)
        cr = evaluate(fc, m)
        if len(cr):
            cr.insert(0, "period", label)
            crows.append(cr)
    crisis = pd.concat(crows, ignore_index=True) if crows else pd.DataFrame(columns=["period", "horizon", "model"])
    vm = vol_managed(d, args.weight_cap, args.cost)
    fig_acf(a, out, sim)
    fig_paths(fc, out, sim)
    fig_losses(ev, out, sim)
    if len(crisis):
        fig_crises(crisis, out, sim)
    fig_managed(vm, out, sim)
    ev.to_csv(out / "table_forecast_evaluation.csv", index=False)
    crisis.to_csv(out / "table_crises.csv", index=False)
    fc["params"].to_csv(out / "table_garch_parameters.csv", index=False)
    vm["table"].to_csv(out / "table_vol_managed.csv", index=False)
    write_report(out, a, fc, ev, crisis, vm, sim, has_vix)
    summary = {"simulated": sim, "has_vix": has_vix, "evaluation": ev.to_dict(orient="records"),
               "vol_managed": vm["table"].to_dict(orient="records"), "vm_alpha": vm["alpha_annual"],
               "vm_alpha_t": vm["alpha_t"], "acf_abs_lag1": float(a["acf_abs"][0]), "acf_r_lag1": float(a["acf_r"][0])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(ev[["horizon", "model", "qlike_gain_vs_rw", "mse_ratio_vs_rw", "mz_slope"]].round(3).to_string(index=False))
    print(f"Vol-managed alpha {vm['alpha_annual']:+.1%} (t = {vm['alpha_t']:.2f})")
    print(f"Report: {out / 'lab07_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
