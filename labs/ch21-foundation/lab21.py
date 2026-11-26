"""Lab 21: Do Foundation Models Beat Classical Forecasts?

Chapter 21 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  How foundation models, used zero-shot, compare with classical forecasts.

WHAT THE LAB DOES
  Part A  One-month-ahead forecasts from every month since 2000 for six
          series: the market's excess return, its log realized variance,
          inflation, industrial production growth, the change in the
          unemployment rate and the change in the 10-year yield. Classical
          models (historical mean, naive, AR by BIC, HAR-type, GARCH for
          variance) against whichever foundation models are installed
          (Chronos-Bolt, TimesFM).
  Part B  Significance: Diebold-Mariano tests and cumulative squared-error
          differences against the AR benchmark.
  Part C  Contamination: accuracy before and after a training cutoff date,
          with a deliberately contaminated model to show the signature.
  Part D  Uncertainty: coverage of 80% forecast intervals.

DATA (free)
  Kenneth R. French Data Library: Fama/French 3 Factors, monthly and daily.
  FRED: CPIAUCSL, INDPRO, UNRATE, GS10.

FOUNDATION MODELS (optional, installed separately; need PyTorch)
  pip install chronos-forecasting      # amazon/chronos-bolt-small, -base
  pip install timesfm                  # google TimesFM
  Without them, the lab runs the classical models, the contamination demo
  and (with --models standin) a stand-in that is NOT a foundation model.

USAGE
  python lab21.py --models chronos-bolt-small timesfm
  python lab21.py --source sim --models standin
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.credit import monthly_fred
from fintest.data import DataUnavailable, load_french_factors
from fintest.tsfm import (CLASSICAL, Memorizer, garch_logvar_forecasts, load_foundation_models, model_columns,
                          rolling_origin, scores, simulated_series, split_scores)

ROOT = Path(__file__).resolve().parent
BENCH = "AR (BIC)"
CLASSIC_COLORS = {"Historical mean": plots.INK_3, "Naive (last value)": plots.YELLOW, "AR (BIC)": plots.INK,
                  "HAR-type (1, 3, 12)": plots.AQUA, "GARCH(1,1)": plots.ORANGE}
FM_COLORS = [plots.BLUE, plots.VIOLET, plots.GREEN, plots.MAGENTA]


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        return simulated_series(seed=seed)
    ff = load_french_factors("ff3_monthly", path=files.get("ff3"))
    col = lambda df, n: df[[c for c in df.columns if c.strip().upper() == n][0]]  # noqa: E731
    mkt = col(ff, "MKT-RF")
    series = {"Market excess return": mkt}
    notes = []
    try:
        dd = load_french_factors("ff3_daily", path=files.get("ff3d"))
        r = col(dd, "MKT-RF") + col(dd, "RF")
        rv = (r ** 2).groupby(r.index.to_period("M")).sum()
        series["Log realized variance"] = np.log(rv[rv > 0])
    except DataUnavailable as e:
        notes.append(f"daily factors not loaded ({str(e).splitlines()[0]})")
    for key, sid, label, tr in (("cpi", "CPIAUCSL", "Inflation (CPI, monthly)", "pct"),
                                ("indpro", "INDPRO", "Industrial production growth", "pct"),
                                ("unrate", "UNRATE", "Unemployment rate change", "diff"),
                                ("gs10", "GS10", "10-year yield change", "diff")):
        try:
            s = monthly_fred(sid, files.get(key))
        except DataUnavailable as e:
            notes.append(f"{sid} not loaded ({str(e).splitlines()[0]})")
            continue
        series[label] = (s.pct_change() if tr == "pct" else s.diff()).dropna()
    return {"series": series, "market_return": mkt, "simulated": False, "notes": notes}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part_a(d: dict, fms, eval_start: str, cutoff: str, context: int) -> dict:
    fcs, tabs = {}, []
    for name, s in d["series"].items():
        mem = Memorizer(s.dropna(), cutoff)
        fc = rolling_origin(s, eval_start, models=fms, memorizer=mem, context=context)
        if name == "Log realized variance":
            g = garch_logvar_forecasts(d["market_return"], s.dropna(), [t - 1 for t in fc.index])
            g.index = pd.PeriodIndex([t + 1 for t in g.index], freq="M")
            fc.insert(len(CLASSICAL) * 2 + 1, "GARCH(1,1)", g["GARCH(1,1)"].reindex(fc.index))
        fcs[name] = fc
        t = scores(fc.drop(columns=[Memorizer.name]), BENCH)
        t.insert(0, "series", name)
        tabs.append(t)
    return {"forecasts": fcs, "table": pd.concat(tabs, ignore_index=True)}


def part_c(a: dict, cutoff: str) -> pd.DataFrame:
    rows = []
    for name, fc in a["forecasts"].items():
        t = split_scores(fc, cutoff, BENCH)
        if len(t):
            t.insert(0, "series", name)
            rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def _colors(models):
    out, k = {}, 0
    for m in models:
        if m in CLASSIC_COLORS:
            out[m] = CLASSIC_COLORS[m]
        else:
            out[m] = FM_COLORS[k % len(FM_COLORS)]
            k += 1
    return out


def fig_heatmap(a, out, sim, eval_start):
    fig, ax, plt = plots.new_figure(10.0, 4.8)
    t = a["table"]
    piv = t.pivot_table(index="model", columns="series", values=f"mse_ratio_vs_{BENCH}", sort=False)
    order = [m for m in t["model"].unique()]
    piv = piv.reindex(order)
    v = np.log(piv.to_numpy())
    lim = max(0.1, float(np.nanmax(np.abs(v[np.isfinite(v)]))) if np.isfinite(v).any() else 0.1)
    lim = min(lim, 0.5)
    im = ax.imshow(-v, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            x = piv.iat[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(math.log(x)) > 0.6 * lim else plots.INK)
    ax.set_xticks(range(piv.shape[1]), [textwrap.fill(c, 14) for c in piv.columns], fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8.5)
    ax.grid(False)
    plots.titles(fig, "Lab Figure 21.1  Forecast error relative to the AR benchmark",
                 _sub(sim, f"Mean squared error divided by that of AR (BIC) · one month ahead, from {eval_start} · "
                           "blue: better than AR · red: worse"))
    plots.finish(fig, out / "fig21_1_heatmap.png")
    plt.close(fig)


def fig_cumulative(a, out, sim, names=("Log realized variance", "Inflation (CPI, monthly)")):
    import matplotlib.pyplot as plt

    names = [n for n in names if n in a["forecasts"]]
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 4.4))
    fig.patch.set_facecolor(plots.SURFACE)
    for ax, n in zip(np.atleast_1d(axes), names):
        plots.style_axes(ax)
        fc = a["forecasts"][n]
        mods = [m for m in model_columns(fc) if m not in (BENCH, Memorizer.name, "Historical mean")]
        cols = _colors(mods)
        e_b = (fc["actual"] - fc[BENCH]) ** 2
        for m in mods:
            dif = (e_b - (fc["actual"] - fc[m]) ** 2).cumsum()
            ax.plot(fc.index.to_timestamp(), dif / e_b.sum(), color=cols[m], linewidth=1.4, label=m)
        ax.axhline(0, color=plots.INK_3, linewidth=0.8)
        ax.set_title(n, loc="left", fontsize=9.5, color=plots.INK)
        ax.set_ylabel("Squared error saved vs AR (share of AR total)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    plots.titles(fig, "Lab Figure 21.2  When does each model gain or lose against AR?",
                 _sub(sim, "Rising line: the model is beating AR over that stretch"))
    plots.finish(fig, out / "fig21_2_cumulative.png")
    plt.close(fig)


def fig_contamination(c, out, sim, cutoff):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    if c.empty:
        plt.close(fig)
        return
    col = f"mse_ratio_vs_{BENCH}"
    piv = c.pivot_table(index="model", columns="window", values=col, aggfunc=lambda x: float(np.exp(np.mean(np.log(x)))))
    piv = piv.drop(index=[BENCH], errors="ignore")
    if "before cutoff" not in piv or "after cutoff" not in piv:
        plt.close(fig)
        return
    y = np.arange(len(piv))[::-1]
    ax.barh(y + 0.2, piv["before cutoff"], 0.4, color=plots.INK_3, label=f"Before {cutoff}")
    ax.barh(y - 0.2, piv["after cutoff"], 0.4, color=plots.BLUE, label=f"From {cutoff}")
    ax.axvline(1, color=plots.INK, linestyle="--", linewidth=1)
    ax.set_yticks(y, piv.index, fontsize=8.5)
    ax.set_xlabel("MSE relative to AR, geometric mean across series (below 1: better than AR)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    plots.titles(fig, "Lab Figure 21.3  Accuracy before and after the training cutoff",
                 _sub(sim, "A model that saw the test data in training looks good before the cutoff and ordinary "
                           "after it · the memorizing model is a demonstration, not a real model"))
    plots.finish(fig, out / "fig21_3_contamination.png")
    plt.close(fig)


def fig_coverage(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    t = a["table"].dropna(subset=["coverage_80"])
    if t.empty:
        plt.close(fig)
        return
    piv = t.pivot_table(index="model", columns="series", values="coverage_80", sort=False)
    x = np.arange(piv.shape[1])
    w = 0.8 / max(1, piv.shape[0])
    cols = _colors(list(piv.index))
    for k, m in enumerate(piv.index):
        ax.bar(x - 0.4 + w * (k + 0.5), piv.loc[m], w, color=cols[m], label=m)
    ax.axhline(0.8, color=plots.INK, linestyle="--", linewidth=1)
    ax.set_xticks(x, [textwrap.fill(c, 14) for c in piv.columns], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    plots.percent_axis(ax)
    ax.set_ylabel("Share of outcomes inside the 80% interval")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=min(4, len(piv.index)))
    plots.titles(fig, "Lab Figure 21.4  Are the forecast intervals honest?",
                 _sub(sim, "Dashed line: the 80% a calibrated interval should reach · classical intervals assume "
                           "normal errors"))
    plots.finish(fig, out / "fig21_4_coverage.png")
    plt.close(fig)


def fig_path(a, out, sim, name="Log realized variance", years: int = 10):
    if name not in a["forecasts"]:
        return
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    fc = a["forecasts"][name]
    fc = fc[fc.index >= fc.index.max() - 12 * years]
    dates = fc.index.to_timestamp()
    ax.plot(dates, fc["actual"], color=plots.INK, linewidth=1.6, label="Actual")
    mods = [m for m in model_columns(fc) if m in ("HAR-type (1, 3, 12)",) or m not in CLASSIC_COLORS]
    mods = [m for m in mods if m != Memorizer.name and m != "GARCH(1,1)"]
    cols = _colors(mods)
    for m in mods:
        ax.plot(dates, fc[m], color=cols[m], linewidth=1.1, alpha=0.9, label=m)
    ax.set_ylabel(name)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plots.titles(fig, f"Lab Figure 21.5  {name}: forecasts and outcomes",
                 _sub(sim, f"One month ahead, last {years} years"))
    plots.finish(fig, out / "fig21_5_path.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


def write_report(out, a, c, sim, notes, args, fm_names):
    md = ["# Lab 21 results: Do Foundation Models Beat Classical Forecasts?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += [f"Foundation models evaluated: {', '.join(fm_names) if fm_names else 'none installed'}.", ""]
    if notes:
        md += ["Notes: " + "; ".join(notes), ""]
    fmt = {"mse": _num(".3g"), f"mse_ratio_vs_{BENCH}": _num(".3f"), "r2_vs_mean": _num("+.3f"),
           f"dm_t_vs_{BENCH}": _num("+.2f"), f"win_rate_vs_{BENCH}": _num(".0%"), "coverage_80": _num(".0%")}
    md += ["**What you will explore:** how foundation models, used zero-shot, compare with classical forecasts.", "",
           f"## Parts A, B and D: one month ahead, from {args.eval_start}", "",
           "MSE ratio below 1 and positive DM t: better than AR (BIC). R² against the historical mean. Coverage: share "
           "of outcomes inside the model's 80% interval.", "", _md(a["table"], fmt), "",
           "![Lab Figure 21.1](fig21_1_heatmap.png)", "", "![Lab Figure 21.2](fig21_2_cumulative.png)", "",
           "![Lab Figure 21.4](fig21_4_coverage.png)", "", "![Lab Figure 21.5](fig21_5_path.png)", "",
           f"## Part C: Before and after the cutoff ({args.cutoff})", "",
           "The memorizing model is a demonstration: it is given the true value before the cutoff. Compare its pattern "
           "with the real models'.", ""]
    if not c.empty:
        md += [_md(c, {f"mse_ratio_vs_{BENCH}": _num(".3f")}), "", "![Lab Figure 21.3](fig21_3_contamination.png)", ""]
    md += ["## Questions to think about", "", "1. Does any foundation model beat AR or HAR on a series by a significant margin?",
           "2. Do the gains hold after the cutoff, when the model cannot have seen the data?",
           "3. Are the foundation models' intervals better calibrated than the classical ones?", ""]
    (out / "lab21_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["download", "sim"], default="download")
    for f in ("ff3", "ff3d", "cpi", "indpro", "unrate", "gs10"):
        ap.add_argument(f"--{f}-file", dest=f, default=None)
    ap.add_argument("--models", nargs="*", default=["chronos-bolt-small", "timesfm"],
                    help="foundation models: chronos-bolt-small, chronos-bolt-base, chronos-t5-small, timesfm, standin")
    ap.add_argument("--eval-start", default="2000-01")
    ap.add_argument("--cutoff", default="2024-01", help="date after which the models cannot have seen the data")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, {k: getattr(args, k) for k in ("ff3", "ff3d", "cpi", "indpro", "unrate", "gs10")},
                 args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    fms, msgs = load_foundation_models(args.models, args.context)
    for m in msgs:
        print("Model not loaded:", m)
    notes = d.get("notes", []) + msgs
    a = part_a(d, fms, args.eval_start, args.cutoff, args.context)
    c = part_c(a, args.cutoff)
    fig_heatmap(a, out, sim, args.eval_start)
    fig_cumulative(a, out, sim)
    fig_contamination(c, out, sim, args.cutoff)
    fig_coverage(a, out, sim)
    fig_path(a, out, sim)
    a["table"].to_csv(out / "table_scores.csv", index=False)
    if not c.empty:
        c.to_csv(out / "table_cutoff.csv", index=False)
    for n, fc in a["forecasts"].items():
        fc.to_csv(out / f"forecasts_{n.split(' (')[0].lower().replace(' ', '_').replace('-', '')}.csv")
    fm_names = [m.name for m in fms]
    write_report(out, a, c, sim, notes, args, fm_names)
    summary = {"simulated": sim, "foundation_models": fm_names, "not_loaded": msgs,
               "scores": a["table"].to_dict(orient="records"), "cutoff": c.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    cols = ["series", "model", f"mse_ratio_vs_{BENCH}", f"dm_t_vs_{BENCH}", "coverage_80"]
    print(a["table"][cols].round(3).to_string(index=False))
    print(f"Report: {out / 'lab21_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
