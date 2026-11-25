"""Lab 12: Do Forward Rates Predict Future Interest Rates?

Chapter 12 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL (the expectations hypothesis)
  "Forward rates are unbiased forecasts of future short-term rates, so a
   steep yield curve predicts rising rates, not higher bond returns."

WHAT THE LAB DOES
  Part A  Fama-Bliss (1987) and Campbell-Shiller (1991) regressions on the
          Federal Reserve's zero-coupon Treasury curve since 1961.
  Part B  Cochrane-Piazzesi (2005): one factor from the whole forward curve,
          in sample and out of sample.
  Part C  Stress tests: subperiods (including the zero-lower-bound years)
          and maturities.
  Part D  The investor's view: average excess returns by maturity, and after
          steep versus flat curves.

DATA (free)
  Federal Reserve Board, "The U.S. Treasury Yield Curve: 1961 to the
  Present" (Gurkaynak, Sack and Wright 2007), feds200628.csv.

USAGE
  python lab12.py
  python lab12.py --gsw-file data/feds200628.csv
  python lab12.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable
from fintest.forecast import clark_west, oos_r2, walk_forward
from fintest.rates import (
    campbell_shiller,
    cochrane_piazzesi,
    fama_bliss,
    fama_bliss_short,
    read_gsw,
    simulated_yields,
    term_structure,
)

ROOT = Path(__file__).resolve().parent
SUBPERIODS = [("Start–1984", None, "1984-12"), ("1985–2007", "1985-01", "2007-12"),
              ("2008–2015 (zero lower bound)", "2008-01", "2015-12"), ("2016–", "2016-01", None)]


def load(source: str, path=None, seed: int = 0) -> dict:
    if source == "sim":
        return {"y": simulated_yields(seed=seed), "simulated": True}
    return {"y": read_gsw(path), "simulated": False}


def _slice(df, a, b):
    return df.loc[pd.Period(a, "M") if a else None:pd.Period(b, "M") if b else None]


def _sub_ts(ts: dict, a, b) -> dict:
    """Restrict regressions to forecast dates in [a, b] (targets may extend beyond b)."""
    return {k: _slice(v, a, b) if isinstance(v, pd.DataFrame) else v for k, v in ts.items()}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part_a(ts: dict) -> dict:
    return {"fb": fama_bliss(ts), "fb_short": fama_bliss_short(ts), "cs": campbell_shiller(ts)}


def part_b(ts: dict, oos_start: str = "1985-01") -> dict:
    cp = cochrane_piazzesi(ts)
    z = cp["rxbar"].to_numpy()
    idx = cp["rxbar"].index
    first = int(np.searchsorted(idx, pd.Period(oos_start, "M")))
    out = {}
    specs = {"Cochrane–Piazzesi (y1, f2–f5)": cp["X"].to_numpy(),
             "Fama–Bliss style (f5 − y1)": ts["fs"][5].reindex(idx).to_numpy()[:, None],
             "Level only (y1)": ts["y"][1].reindex(idx).to_numpy()[:, None]}
    rows = []
    for name, X in specs.items():
        wf = walk_forward(z, X, first=first, gap=11, min_train=120)
        v = wf["valid"]
        cw = clark_west(z[v], wf["model"][v], wf["bench"][v], lags=11)
        rows.append({"model": name, "oos_r2": oos_r2(z[v], wf["model"][v], wf["bench"][v]), "cw_t": cw["stat"],
                     "months": int(v.sum())})
        out[name] = wf
    return {"cp": cp, "oos": pd.DataFrame(rows), "wf": out, "oos_start": str(idx[first]) if first < len(idx) else ""}


def part_c(ts: dict) -> pd.DataFrame:
    rows = []
    for label, a, b in SUBPERIODS:
        s = _sub_ts(ts, a, b)
        if s["fs"][5].dropna().shape[0] < 60:
            continue
        fb = fama_bliss(s).set_index("n")
        cs = campbell_shiller(s).set_index("n")
        cp = cochrane_piazzesi(s)
        nmax = max(cs.index[cs["months"] >= 60]) if (cs["months"] >= 60).any() else 5
        rows.append({"period": label, "fb_slope_5y": fb.loc[5, "slope"], "fb_t_5y": fb.loc[5, "t_vs_0"],
                     "fb_r2_5y": fb.loc[5, "r2"], f"cs_slope_max_n": cs.loc[nmax, "slope"], "cs_n": nmax,
                     "cs_t_vs_1": cs.loc[nmax, "t_vs_1"], "cp_r2": cp["r2"], "months": fb.loc[5, "months"]})
    return pd.DataFrame(rows)


def part_d(ts: dict) -> dict:
    rx = ts["rx"]
    rows = []
    for n in rx.columns:
        x = rx[n].dropna()
        rows.append({"n": n, "mean": float(x.mean()), "sd": float(x.std()), "sharpe": float(x.mean() / x.std()),
                     "share_positive": float((x > 0).mean())})
    by_n = pd.DataFrame(rows)
    slope = ts["fs"][5]
    med = slope.expanding(min_periods=60).median().shift(1)  # only past information
    steep = slope > med
    cond = []
    for n in rx.columns:
        x = rx[n]
        ok = x.notna() & med.notna()
        cond.append({"n": n, "after_steep": float(x[ok & steep].mean()), "after_flat": float(x[ok & ~steep].mean())})
    return {"by_n": by_n, "conditional": pd.DataFrame(cond)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_yields(ts, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    y = ts["y"]
    dates = y.index.to_timestamp()
    for n, col in ((1, plots.BLUE), (5, plots.ORANGE), (10, plots.INK_3)):
        if n in y.columns:
            ax.plot(dates, y[n], color=col, linewidth=1.3, label=f"{n}-year zero yield")
    plots.percent_axis(ax)
    ax.set_ylabel("Yield (continuously compounded)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plots.titles(fig, "Lab Figure 12.1  The Treasury yield curve since 1961",
                 _sub(sim, "Month-end zero-coupon yields (Gürkaynak, Sack and Wright 2007)"))
    plots.finish(fig, out / "fig12_1_yields.png")
    plt.close(fig)


def _coef_plot(ax, df, key, ref, color, label):
    ax.errorbar(df["n"], df[key], yerr=2 * df["se"], fmt="o", color=color, capsize=4, label=label)
    ax.axhline(ref, color=plots.INK, linestyle="--", linewidth=1)


def fig_fama_bliss(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    fb = a["fb"][a["fb"]["n"] <= 5]
    _coef_plot(ax, fb, "slope", 0.0, plots.BLUE, "Slope of excess return on forward spread (± 2 s.e.)")
    ax.axhline(1.0, color=plots.INK_3, linestyle=":", linewidth=1)
    ax.text(fb["n"].max() + 0.1, 0.02, "Expectations hypothesis: 0", fontsize=8.5, color=plots.INK, va="bottom",
            ha="right")
    ax.text(fb["n"].max() + 0.1, 1.02, "Forward spread fully a risk premium: 1", fontsize=8.5, color=plots.INK_2,
            va="bottom", ha="right")
    ax.set_xticks(fb["n"], [f"{n}-year" for n in fb["n"]])
    ax.set_xlabel("Bond maturity")
    ax.set_ylabel("Slope coefficient")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 12.2  Do forward spreads forecast bond excess returns?",
                 _sub(sim, "Fama–Bliss regressions, one-year holding period, Newey–West errors (18 lags)"))
    plots.finish(fig, out / "fig12_2_fama_bliss.png")
    plt.close(fig)


def fig_campbell_shiller(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    cs = a["cs"]
    _coef_plot(ax, cs, "slope", 1.0, plots.ORANGE, "Campbell–Shiller slope (± 2 s.e.)")
    ax.axhline(0, color=plots.GRID, linewidth=1)
    ax.text(cs["n"].min() - 0.2, 1.05, "Expectations hypothesis: 1", fontsize=8.5, color=plots.INK, va="bottom")
    ax.set_xticks(cs["n"], [f"{n}y" for n in cs["n"]])
    ax.set_xlabel("Long-bond maturity")
    ax.set_ylabel("Slope coefficient")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    plots.titles(fig, "Lab Figure 12.3  Do long yields move as the hypothesis predicts?",
                 _sub(sim, "Change in the long yield over a year on the yield spread; the hypothesis predicts a slope of 1"))
    plots.finish(fig, out / "fig12_3_campbell_shiller.png")
    plt.close(fig)


def fig_cp(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    cp = b["cp"]
    names = cp["names"][1:]
    ax.plot(range(len(names)), cp["coef"][1:], marker="o", color=plots.VIOLET, linewidth=1.8)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_xticks(range(len(names)), ["y(1)"] + [f"f({n[1:]})" for n in names[1:]])
    ax.set_ylabel("Coefficient")
    oos = b["oos"].set_index("model")
    first = oos.index[0]
    plots.titles(fig, "Lab Figure 12.4  One factor from the whole forward curve",
                 _sub(sim, f"Cochrane–Piazzesi regression of the average 2–5 year excess return · in-sample R² "
                           f"{cp['r2']:.2f} · out-of-sample R² from {b['oos_start']} {oos.loc[first, 'oos_r2']:+.2f}"))
    plots.finish(fig, out / "fig12_4_cp.png")
    plt.close(fig)


def fig_conditional(dd, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    c = dd["conditional"]
    x = np.arange(len(c))
    ax.bar(x - 0.2, c["after_steep"], width=0.4, color=plots.BLUE, label="After a steep curve (slope above its past median)")
    ax.bar(x + 0.2, c["after_flat"], width=0.4, color=plots.INK_3, label="After a flat curve")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_xticks(x, [f"{n}y" for n in c["n"]])
    plots.percent_axis(ax)
    ax.set_xlabel("Bond maturity")
    ax.set_ylabel("Average one-year excess return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 12.5  Bond returns after steep and flat yield curves",
                 _sub(sim, "Excess return over the one-year bond, held for a year"))
    plots.finish(fig, out / "fig12_5_conditional.png")
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


F2 = _num(".2f")


def write_report(out, ts, a, b, c, dd, sim):
    y = ts["y"]
    md = ["# Lab 12 results: Do Forward Rates Predict Future Interest Rates?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**Claim on trial (expectations hypothesis):** forward rates are unbiased forecasts of future short rates.", "",
           f"Data: month-end zero yields, {y.index[0]} to {y.index[-1]}. One-year holding periods overlap; standard "
           "errors are Newey–West with 18 lags.", "", "![Lab Figure 12.1](fig12_1_yields.png)", "",
           "## Part A: Fama–Bliss and Campbell–Shiller", "", "**Excess returns on forward spreads (hypothesis: slope 0)**", "",
           _md(a["fb"].rename(columns={"n": "Maturity (years)", "slope": "Slope", "se": "s.e.", "t_vs_0": "t (vs 0)",
                                       "t_vs_1": "t (vs 1)", "r2": "R²", "months": "Months"}),
               {"Slope": F2, "s.e.": F2, "t (vs 0)": F2, "t (vs 1)": F2, "R²": F2}), "",
           "**Future change in the one-year rate on the forward spread (hypothesis: slope 1)**", "",
           _md(a["fb_short"].rename(columns={"n": "Forward maturity", "horizon_years": "Horizon (years)", "slope": "Slope",
                                             "se": "s.e.", "t_vs_1": "t (vs 1)", "r2": "R²", "months": "Months"}),
               {"Slope": F2, "s.e.": F2, "t (vs 1)": F2, "R²": F2}), "",
           "**Campbell–Shiller: change in the long yield on the scaled spread (hypothesis: slope 1)**", "",
           _md(a["cs"].rename(columns={"n": "Maturity (years)", "slope": "Slope", "se": "s.e.", "t_vs_1": "t (vs 1)",
                                       "r2": "R²", "months": "Months"}),
               {"Slope": F2, "s.e.": F2, "t (vs 1)": F2, "R²": F2}), "",
           "![Lab Figure 12.2](fig12_2_fama_bliss.png)", "", "![Lab Figure 12.3](fig12_3_campbell_shiller.png)", "",
           "## Part B: Cochrane–Piazzesi", ""]
    cp = b["cp"]
    md += ["| Variable | Coefficient | t |", "| --- | --- | --- |"]
    md += [f"| {nm} | {c_:.2f} | {t_:.2f} |" for nm, c_, t_ in zip(cp["names"], cp["coef"], cp["t"])]
    md += ["", f"In-sample R² {cp['r2']:.2f} ({cp['months']} months).", "",
           _md(cp["each"].rename(columns={"n": "Maturity", "loading": "Loading on CP factor", "t": "t", "r2": "R²"}),
               {"Loading on CP factor": F2, "t": F2, "R²": F2}), "",
           f"**Out of sample** (expanding window from {b['oos_start']}, 11-month gap so no target overlaps the forecast "
           "date; benchmark: historical mean excess return)", "",
           _md(b["oos"].rename(columns={"model": "Predictors", "oos_r2": "Out-of-sample R²", "cw_t": "Clark–West t",
                                        "months": "Months"}), {"Out-of-sample R²": _num("+.2f"), "Clark–West t": F2}), "",
           "![Lab Figure 12.4](fig12_4_cp.png)", "", "## Part C: Subperiods", "",
           _md(c.rename(columns={"period": "Period", "fb_slope_5y": "FB slope, 5y", "fb_t_5y": "t", "fb_r2_5y": "R²",
                                 "cs_slope_max_n": "CS slope, longest", "cs_n": "CS maturity", "cs_t_vs_1": "t (vs 1)",
                                 "cp_r2": "CP R²", "months": "Months"}),
               {"FB slope, 5y": F2, "t": F2, "R²": F2, "CS slope, longest": F2, "t (vs 1)": F2, "CP R²": F2}), "",
           "## Part D: The investor's view", "",
           _md(dd["by_n"].rename(columns={"n": "Maturity", "mean": "Mean excess return", "sd": "Std. dev.",
                                          "sharpe": "Sharpe (annual)", "share_positive": "Share of years positive"}),
               {"Mean excess return": _num("+.2%"), "Std. dev.": _num(".2%"), "Sharpe (annual)": F2,
                "Share of years positive": _num(".0%")}), "",
           _md(dd["conditional"].rename(columns={"n": "Maturity", "after_steep": "After steep curve",
                                                 "after_flat": "After flat curve"}),
               {"After steep curve": _num("+.2%"), "After flat curve": _num("+.2%")}), "",
           "Steep/flat: the 5-year forward spread above or below its median up to the previous month.", "",
           "![Lab Figure 12.5](fig12_5_conditional.png)", "", "## Your verdict", "",
           "1. Do forward spreads forecast future short rates one for one, as the hypothesis requires?",
           "2. Do they forecast excess bond returns instead? Does the evidence hold out of sample and across subperiods?",
           "3. What does this imply for reading forward rates as the market's forecast of policy rates?", ""]
    (out / "lab12_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fed", "sim"], default="fed")
    ap.add_argument("--gsw-file", default=None, help="feds200628.csv from the Federal Reserve")
    ap.add_argument("--oos-start", default="1985-01")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, args.gsw_file, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    ts = term_structure(d["y"])
    a, b, c, dd = part_a(ts), part_b(ts, args.oos_start), part_c(ts), part_d(ts)
    fig_yields(ts, out, sim)
    fig_fama_bliss(a, out, sim)
    fig_campbell_shiller(a, out, sim)
    fig_cp(b, out, sim)
    fig_conditional(dd, out, sim)
    for k in ("fb", "fb_short", "cs"):
        a[k].to_csv(out / f"table_{k}.csv", index=False)
    b["oos"].to_csv(out / "table_oos.csv", index=False)
    c.to_csv(out / "table_subperiods.csv", index=False)
    dd["by_n"].to_csv(out / "table_excess_returns.csv", index=False)
    write_report(out, ts, a, b, c, dd, sim)
    summary = {"simulated": sim, "fama_bliss": a["fb"].to_dict(orient="records"),
               "campbell_shiller": a["cs"].to_dict(orient="records"), "cp_r2": b["cp"]["r2"],
               "cp_coef": dict(zip(b["cp"]["names"], map(float, b["cp"]["coef"]))),
               "oos": b["oos"].to_dict(orient="records"), "subperiods": c.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a["fb"].round(3).to_string(index=False))
    print(a["cs"][["n", "slope", "t_vs_1"]].round(2).to_string(index=False))
    print(f"CP R2 {b['cp']['r2']:.2f}"); print(b["oos"].round(3).to_string(index=False))
    print(f"Report: {out / 'lab12_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
