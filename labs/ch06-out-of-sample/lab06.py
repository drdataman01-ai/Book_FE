"""Lab 6: Do Valuations Predict the Equity Premium Out of Sample?

Chapter 6 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "In-sample significant predictors beat the historical mean out of sample."

WHAT THE LAB DOES
  Part A  In-sample regressions of next month's equity premium on each of the
          14 Goyal-Welch predictors: slope, White t-statistic and R-squared.
  Part B  Walk-forward forecasts from 1965 with expanding windows: out-of-
          sample R-squared against the historical mean, Clark-West and
          Diebold-Mariano tests, and cumulative squared-error differences
          (Goyal and Welch 2008).
  Part C  Remedies: Campbell-Thompson (2008) sign and forecast restrictions,
          an equal-weighted combination of the 14 forecasts (Rapach, Strauss
          and Zhou 2010), and the certainty-equivalent gain of a mean-variance
          investor.
  Part D  A natural experiment: the same forecasts evaluated only after 2005,
          the end of Goyal and Welch's original sample.
  Part E  White's (2000) Reality Check on a simulated universe of technical
          trading rules with no skill: how good does the best rule look?

DATA
  Amit Goyal's updated Goyal-Welch data (PredictorData.xlsx, "Monthly"
  sheet), free from https://sites.google.com/view/agoyal145. Part E needs no
  market data.

USAGE
  python lab06.py --data-file data/PredictorData2024.xlsx
  python lab06.py --source sim            # fake data, to test the code

OUTPUT
  report/lab06_report.md, five figures, CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, simulated_daily_market
from fintest.forecast import (
    certainty_equivalent,
    clark_west,
    cumulative_sse_difference,
    diebold_mariano,
    mean_variance_weights,
    oos_r2,
    rolling_variance,
    walk_forward,
)
from fintest.predictors import PREDICTORS, PUBLICATION_YEAR, load_predictors, simulated_predictors
from fintest.search import reality_check, rule_positions, rule_universe
from fintest.stats import ols_newey_west

ROOT = Path(__file__).resolve().parent
NAMES = list(PREDICTORS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load(source: str, data_file=None, start: str = "1927-01", end: str | None = None, sim_seed: int = 0) -> pd.DataFrame:
    """Monthly equity premium (simple, decimal) and the 14 predictors, each dated by when it is known."""
    raw = simulated_predictors(seed=sim_seed) if source == "sim" else load_predictors(data_file)
    df = raw[["eqp", "rf"] + NAMES].copy()
    if start:
        df = df.loc[pd.Period(start, "M"):]
    if end:
        df = df.loc[:pd.Period(end, "M")]
    df = df[df["eqp"].notna()].copy()
    df.attrs["simulated"] = raw.attrs.get("simulated", False)
    return df


def next_premium(df: pd.DataFrame) -> np.ndarray:
    """z[t] = equity premium of month t+1, paired with predictors known at the end of month t."""
    return df["eqp"].shift(-1).to_numpy()


# ---------------------------------------------------------------------------
# Part A: in-sample regressions
# ---------------------------------------------------------------------------


def part_a(df: pd.DataFrame) -> pd.DataFrame:
    z = next_premium(df)
    rows = []
    for name in NAMES:
        x = df[name].to_numpy()
        ok = np.isfinite(z) & np.isfinite(x)
        if ok.sum() < 120:
            continue
        reg = ols_newey_west(z[ok][:, None], x[ok], lags=0)  # lags = 0: White's heteroskedasticity-robust errors
        resid_var = np.var(z[ok] - (reg["coef"][0, 0] + reg["coef"][1, 0] * x[ok]))
        r2 = 1 - resid_var / np.var(z[ok])
        n = int(ok.sum())
        rows.append({"predictor": name, "description": PREDICTORS[name][0], "start": str(df.index[ok][0]),
                     "months": n, "slope": float(reg["coef"][1, 0]), "t_white": float(reg["t"][1, 0]),
                     "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - 2), "expected_sign": PREDICTORS[name][1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: walk-forward forecasts
# ---------------------------------------------------------------------------


def _first_index(df: pd.DataFrame, oos_start: str) -> int:
    return int(np.searchsorted(df.index.astype(str).to_numpy(), oos_start))


def forecasts(df: pd.DataFrame, oos_start: str = "1965-01", min_train: int = 240) -> dict:
    """Expanding-window forecasts for each predictor, plain and with Campbell-Thompson restrictions, and the
    kitchen-sink regression on all predictors that are available."""
    z = next_premium(df)
    first = _first_index(df, oos_start)
    out = {}
    for name in NAMES:
        x = df[name].to_numpy()
        out[name] = {"plain": walk_forward(z, x, first, min_train=min_train),
                     "ct": walk_forward(z, x, first, min_train=min_train, sign=PREDICTORS[name][1], nonneg=True)}
    cols = [n for n in NAMES if df[n].iloc[:first].notna().sum() >= min_train]
    out["kitchen sink"] = {"plain": walk_forward(z, df[cols].to_numpy(), first, min_train=min_train)}
    # Equal-weighted combination of the single-predictor forecasts available at each date
    F = np.column_stack([out[n]["plain"]["model"] for n in NAMES])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # months before any forecast exists
        comb = np.nanmean(F, axis=1)
        bench = np.nanmean(np.column_stack([out[n]["plain"]["bench"] for n in NAMES]), axis=1)
    valid = np.isfinite(comb) & np.isfinite(bench) & np.isfinite(z)
    out["combination"] = {"plain": {"actual": z, "model": comb, "bench": bench, "valid": valid}}
    return {"fc": out, "z": z, "first": first, "kitchen_sink_inputs": cols}


def _evaluate(w: dict, mask: np.ndarray | None = None) -> dict:
    v = w["valid"].copy()
    if mask is not None:
        v &= mask
    if v.sum() < 24:
        return {"r2_oos": np.nan, "cw_stat": np.nan, "cw_p": np.nan, "dm_stat": np.nan, "dm_p": np.nan, "n": int(v.sum())}
    a, m, b = w["actual"][v], w["model"][v], w["bench"][v]
    cw = clark_west(a, m, b)
    dm = diebold_mariano(a, b, m)  # positive: the model is more accurate than the historical mean
    return {"r2_oos": oos_r2(a, m, b), "cw_stat": cw["stat"], "cw_p": cw["p_value"], "dm_stat": dm["stat"],
            "dm_p": dm["p_value"], "n": int(v.sum())}


def part_b(df: pd.DataFrame, fc: dict) -> dict:
    rows = []
    for name, w in fc["fc"].items():
        if name == "combination":
            continue
        e = _evaluate(w["plain"])
        e.update(predictor=name)
        rows.append(e)
    t = pd.DataFrame(rows)[["predictor", "n", "r2_oos", "cw_stat", "cw_p", "dm_stat", "dm_p"]]
    cum = {n: cumulative_sse_difference(w["plain"]["actual"], w["plain"]["model"], w["plain"]["bench"])
           for n, w in fc["fc"].items()}
    return {"table": t, "cum": cum}


# ---------------------------------------------------------------------------
# Part C: remedies
# ---------------------------------------------------------------------------


def cer_gain(df: pd.DataFrame, w: dict, gamma: float = 3.0, var_window: int = 60) -> float:
    """Annualized certainty-equivalent return of a mean-variance investor using the model, minus the same investor
    using the historical mean (Campbell and Thompson 2008). Weights between 0 and 1.5."""
    eqp = df["eqp"].to_numpy()
    rf = df["rf"].to_numpy()
    var = rolling_variance(eqp, var_window)  # variance known at the end of month t
    nxt = slice(1, None)
    v = w["valid"][:-1] & np.isfinite(var[:-1])
    if v.sum() < 24:
        return float("nan")
    wm = mean_variance_weights(w["model"][:-1][v], var[:-1][v], gamma)
    wb = mean_variance_weights(w["bench"][:-1][v], var[:-1][v], gamma)
    e, r = eqp[nxt][v], rf[nxt][v]
    return certainty_equivalent(e, r, wm, gamma) - certainty_equivalent(e, r, wb, gamma)


def part_c(df: pd.DataFrame, fc: dict, gamma: float = 3.0) -> dict:
    rows = []
    for name in NAMES + ["combination", "kitchen sink"]:
        w = fc["fc"][name]
        plain = _evaluate(w["plain"])
        row = {"predictor": name, "r2_oos": plain["r2_oos"], "cer_gain": cer_gain(df, w["plain"], gamma)}
        if "ct" in w:
            ct = _evaluate(w["ct"])
            row.update(r2_oos_ct=ct["r2_oos"], cw_p_ct=ct["cw_p"], cer_gain_ct=cer_gain(df, w["ct"], gamma))
        else:
            row.update(r2_oos_ct=np.nan, cw_p_ct=plain["cw_p"] if name == "combination" else np.nan, cer_gain_ct=np.nan)
        rows.append(row)
    return {"table": pd.DataFrame(rows), "gamma": gamma}


# ---------------------------------------------------------------------------
# Part D: after the original sample
# ---------------------------------------------------------------------------


def part_d(df: pd.DataFrame, fc: dict, split: str = "2005-01") -> dict:
    """The same walk-forward forecasts, evaluated before and after `split`. Forecasts for month t+1 are made at t,
    so the evaluation is split by the forecast origin."""
    s = pd.Period(split, "M")
    before = np.asarray(df.index < s)
    after = ~before
    rows = []
    for name in NAMES + ["combination", "kitchen sink"]:
        w = fc["fc"][name]["plain"]
        b, a = _evaluate(w, before), _evaluate(w, after)
        rows.append({"predictor": name, "published": PUBLICATION_YEAR.get(name, np.nan),
                     "r2_oos_before": b["r2_oos"], "r2_oos_after": a["r2_oos"], "cw_p_after": a["cw_p"],
                     "n_after": a["n"]})
    return {"table": pd.DataFrame(rows), "split": split}


# ---------------------------------------------------------------------------
# Part E: the Reality Check on rules with no skill
# ---------------------------------------------------------------------------


def part_e(n_days: int = 6000, n_boot: int = 500, seed: int = 0) -> dict:
    """A market whose excess return has mean zero and no predictability, so no rule has skill; a universe of
    long-or-cash moving-average and breakout rules; each rule's return over Treasury bills; White's Reality Check."""
    mk = simulated_daily_market(n_days=n_days, seed=seed + 11)
    ex = mk["Mkt-RF"].to_numpy()
    ex = ex - ex.mean()  # no equity premium: being in the market is worth nothing on average
    lp = np.cumsum(ex)
    rules = rule_universe(shorts=(1, 2, 5, 10, 20), longs=(25, 50, 100, 150, 200), breakouts=(10, 20, 50, 100, 200),
                          bands=(0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.05), holds=(0, 5, 10, 25))
    pos = rule_positions(lp, rules)
    P = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    d = (P * ex[:, None])[250:]  # excess return over bills while in the market
    rc = reality_check(d, n_boot=n_boot, seed=seed)
    T = d.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_all = d.mean(axis=0) / (d.std(axis=0, ddof=1) / math.sqrt(T))
    return {"n_rules": len(rules), "n_days": T, "best_rule": rules[rc["best"]].name,
            "best_excess_annual": rc["best_mean"] * 252, "best_t": rc["best_t"], "p_nominal": rc["p_nominal"],
            "rules_nominal_5pct": rc["n_nominal_5pct"], "p_reality_check": rc["p_reality_check"],
            "p_spa": rc["p_spa"], "t_all": t_all[np.isfinite(t_all)]}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim: bool, text: str) -> str:
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_in_sample(a, out, sim, period):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = a.sort_values("t_white")
    colors = [plots.BLUE if abs(v) > 1.96 else plots.INK_3 for v in t.t_white]
    ax.barh(t.predictor, t.t_white, color=colors)
    for v in (-1.96, 1.96):
        ax.axvline(v, color=plots.RED, linewidth=0.8, linestyle="--")
    for y, (tv, r2) in enumerate(zip(t.t_white, t.r2)):
        ax.text(tv + (0.1 if tv >= 0 else -0.1), y, f"R² {r2:.1%}", va="center", ha="left" if tv >= 0 else "right",
                fontsize=7.5, color=plots.INK_2)
    ax.set_xlabel("White t-statistic of the slope")
    plots.titles(fig, "Lab Figure 6.1  In-sample t-statistics of the 14 predictors",
                 _sub(sim, f"Next month's equity premium on each predictor, {period} · blue: |t| > 1.96"))
    plots.finish(fig, out / "fig6_1_in_sample.png")
    plt.close(fig)


def fig_oos(c, out, sim, oos_start):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    t = c["table"].set_index("predictor").reindex(NAMES + ["combination", "kitchen sink"])
    y = np.arange(len(t))
    ax.barh(y + 0.2, t.r2_oos, 0.4, color=plots.INK_3, label="Plain")
    ax.barh(y - 0.2, t.r2_oos_ct, 0.4, color=plots.BLUE, label="Campbell–Thompson restrictions")
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.axvline(0, color=plots.INK, linewidth=0.8)
    lo = max(-0.05, np.nanmin(t[["r2_oos", "r2_oos_ct"]].to_numpy()) - 0.002)
    ax.set_xlim(left=lo)
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Out-of-sample R² against the historical mean (bars below −5% are cut off)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    plots.titles(fig, "Lab Figure 6.2  Out-of-sample R² against the historical mean",
                 _sub(sim, f"Expanding-window forecasts from {oos_start}; combination: the average of the 14 forecasts"))
    plots.finish(fig, out / "fig6_2_oos_r2.png")
    plt.close(fig)


def fig_cumulative(df, b, out, sim):
    import matplotlib.pyplot as plt

    names = NAMES + ["combination", "kitchen sink"]
    fig, axes = plt.subplots(4, 4, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor(plots.SURFACE)
    dates = df.index.to_timestamp()
    for ax, n in zip(axes.ravel(), names):
        plots.style_axes(ax)
        c = b["cum"][n]
        ax.plot(dates, c, color=plots.BLUE if c[-1] > 0 else plots.RED, linewidth=1.2)
        ax.axhline(0, color=plots.INK_3, linewidth=0.6)
        ax.set_title(n, loc="left", fontsize=9, color=plots.INK)
        ax.tick_params(labelsize=7)
    plots.titles(fig, "Lab Figure 6.3  Cumulative squared-error difference, historical mean minus model",
                 _sub(sim, "Rising: the predictor beats the historical mean; falling: it loses (Goyal and Welch 2008)"))
    plots.finish(fig, out / "fig6_3_cumulative.png")
    plt.close(fig)


def fig_cer(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = c["table"].set_index("predictor").reindex(NAMES + ["combination", "kitchen sink"])
    y = np.arange(len(t))
    ax.barh(y + 0.2, t.cer_gain, 0.4, color=plots.INK_3, label="Plain")
    ax.barh(y - 0.2, t.cer_gain_ct, 0.4, color=plots.BLUE, label="Campbell–Thompson restrictions")
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.axvline(0, color=plots.INK, linewidth=0.8)
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Certainty-equivalent gain over the historical mean (per year)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    plots.titles(fig, "Lab Figure 6.4  What the forecasts are worth to an investor",
                 _sub(sim, f"Mean-variance investor, risk aversion {c['gamma']:g}, equity weight between 0 and 150%"))
    plots.finish(fig, out / "fig6_4_cer.png")
    plt.close(fig)


def fig_reality_check(e, out):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    ax.hist(e["t_all"], bins=60, color=plots.INK_3, edgecolor=plots.SURFACE)
    ax.axvline(1.645, color=plots.RED, linewidth=0.8, linestyle="--")
    n_sig = int((e["t_all"] > 1.645).sum())
    ax.text(0.02, 0.95, f"{n_sig} of {e['n_rules']:,} useless rules have t > 1.645\n"
                        f"Highest average return: {e['best_rule']}, nominal p = {e['p_nominal']:.3f}\n"
                        f"Reality Check p = {e['p_reality_check']:.2f}, SPA p = {e['p_spa']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, color=plots.INK)
    ax.set_xlabel("t-statistic of each rule's average excess return")
    plots.titles(fig, "Lab Figure 6.5  Searching many rules with no skill",
                 f"{e['n_rules']:,} rules on a simulated market with no predictability, {e['n_days']:,} days · dashed "
                 "line: one-sided 5% cut-off")
    plots.finish(fig, out / "fig6_5_reality_check.png")
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


def write_report(out, df, a, b, c, d, e, sim, period, oos_start):
    md = ["# Lab 6 results: Do Valuations Predict the Equity Premium Out of Sample?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets. In "
               "the simulation only the dividend-price ratio truly predicts the premium.", ""]
    md += ["**Claim on trial:** in-sample significant predictors beat the historical mean out of sample.", "",
           f"Sample: {period}. Target: next month's equity premium (S&P 500 total return minus the Treasury bill rate).",
           "", "## Part A: In-sample regressions", "",
           _md(a[["predictor", "description", "start", "months", "slope", "t_white", "r2", "adj_r2"]],
               {"slope": _num("+.4f"), "t_white": _num("+.2f"), "r2": _num(".2%"), "adj_r2": _num(".2%")}), "",
           "![Lab Figure 6.1](fig6_1_in_sample.png)", "",
           "## Part B: Walk-forward forecasts", "",
           f"Expanding windows, first forecast {oos_start}, at least 20 years of training data. Clark–West and "
           "Diebold–Mariano statistics are one-sided: positive values favor the predictor.", "",
           _md(b["table"], {"r2_oos": _num("+.2%"), "cw_stat": _num(".2f"), "cw_p": _num(".3f"),
                            "dm_stat": _num(".2f"), "dm_p": _num(".3f")}), "",
           "Diebold–Mariano is undersized for a model nested in the historical mean; Clark–West corrects for the noise "
           "of estimating a slope that is zero under the null.", "",
           "![Lab Figure 6.3](fig6_3_cumulative.png)", "",
           "## Part C: Remedies", "",
           _md(c["table"][["predictor", "r2_oos", "r2_oos_ct", "cw_p_ct", "cer_gain", "cer_gain_ct"]],
               {"r2_oos": _num("+.2%"), "r2_oos_ct": _num("+.2%"), "cw_p_ct": _num(".3f"), "cer_gain": _num("+.2%"),
                "cer_gain_ct": _num("+.2%")}), "",
           "Campbell–Thompson: use the historical mean when the estimated slope has the wrong sign, and replace "
           "negative forecasts by zero. CER gain: certainty-equivalent return of a mean-variance investor (risk "
           f"aversion {c['gamma']:g}) using the model, minus the same investor using the historical mean, per year.", "",
           "![Lab Figure 6.2](fig6_2_oos_r2.png)", "", "![Lab Figure 6.4](fig6_4_cer.png)", "",
           f"## Part D: After {d['split'][:4]}", "",
           _md(d["table"], {"published": _num(".0f"), "r2_oos_before": _num("+.2%"), "r2_oos_after": _num("+.2%"),
                            "cw_p_after": _num(".3f")}), "",
           f"## Part E: White's Reality Check on {e['n_rules']:,} rules with no skill", "",
           f"A simulated market with no equity premium and no predictability; each rule is long the market or in bills. "
           f"Best rule {e['best_rule']}: {e['best_excess_annual']:+.1%} a year over bills, t = {e['best_t']:.2f}, "
           f"nominal p = {e['p_nominal']:.3f}. {e['rules_nominal_5pct']} rules are 'significant' at 5% on their own. "
           f"Reality Check p = {e['p_reality_check']:.3f}; SPA p = {e['p_spa']:.3f}.", "",
           "![Lab Figure 6.5](fig6_5_reality_check.png)", "",
           "## Your verdict", "",
           "1. How many predictors were significant in sample, and how many beat the historical mean out of sample?",
           "2. Do the Campbell–Thompson restrictions or the combination recover a gain an investor would value?",
           "3. Did any predictor keep working after 2005?", ""]
    (out / "lab06_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["goyal", "sim"], default="goyal")
    ap.add_argument("--data-file", default=None, help="Goyal-Welch spreadsheet, or a CSV export of its Monthly sheet")
    ap.add_argument("--start", default="1927-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--oos-start", default="1965-01")
    ap.add_argument("--post-split", default="2005-01", help="Part D: first month after the original sample")
    ap.add_argument("--gamma", type=float, default=3.0, help="Part C: risk aversion")
    ap.add_argument("--n-boot", type=int, default=500, help="Part E: bootstrap samples")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        df = load(args.source, args.data_file, args.start, args.end, args.sim_seed)
    except DataUnavailable as e:
        raise SystemExit(str(e)) from None
    sim = df.attrs.get("simulated", False)
    period = f"{df.index[0]} to {df.index[-1]}"
    a = part_a(df)
    fc = forecasts(df, args.oos_start)
    b = part_b(df, fc)
    c = part_c(df, fc, args.gamma)
    d = part_d(df, fc, args.post_split)
    e = part_e(n_boot=args.n_boot, seed=args.seed)
    fig_in_sample(a, out, sim, period)
    fig_oos(c, out, sim, args.oos_start)
    fig_cumulative(df, b, out, sim)
    fig_cer(c, out, sim)
    fig_reality_check(e, out)
    a.to_csv(out / "table_in_sample.csv", index=False)
    b["table"].to_csv(out / "table_out_of_sample.csv", index=False)
    c["table"].to_csv(out / "table_remedies.csv", index=False)
    d["table"].to_csv(out / "table_after_2005.csv", index=False)
    write_report(out, df, a, b, c, d, e, sim, period, args.oos_start)
    summary = {"simulated": sim, "period": period, "in_sample": a.to_dict(orient="records"),
               "out_of_sample": b["table"].to_dict(orient="records"), "remedies": c["table"].to_dict(orient="records"),
               "after_split": d["table"].to_dict(orient="records"),
               "reality_check": {k: v for k, v in e.items() if k != "t_all"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"A  {(a.t_white.abs() > 1.96).sum()} of {len(a)} predictors have |t| > 1.96 in sample")
    bt = b["table"].set_index("predictor")
    print(f"B  {(bt.loc[NAMES, 'r2_oos'] > 0).sum()} of {len(NAMES)} beat the historical mean out of sample; "
          f"kitchen sink R2_OS {bt.loc['kitchen sink', 'r2_oos']:+.2%}")
    ct = c["table"].set_index("predictor")
    print(f"C  combination R2_OS {ct.loc['combination', 'r2_oos']:+.2%}, CER gain {ct.loc['combination', 'cer_gain']:+.2%}"
          f"/yr; with CT restrictions {(ct.loc[NAMES, 'r2_oos_ct'] > 0).sum()} of 14 are positive")
    dt = d["table"].set_index("predictor")
    print(f"D  after {args.post_split[:4]}: {(dt.loc[NAMES, 'r2_oos_after'] > 0).sum()} of 14 positive")
    print(f"E  best of {e['n_rules']} useless rules: nominal p {e['p_nominal']:.3f}, Reality Check p "
          f"{e['p_reality_check']:.3f}")
    print(f"Report: {out / 'lab06_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
