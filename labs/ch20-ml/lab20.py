"""Lab 20: Does Machine Learning Find Hidden Alpha?

Chapter 20 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  Whether machine learning finds return predictability that linear models
  miss, and whether it survives costs.

WHAT THE LAB DOES
  Part A  A horse race on a panel of 149 portfolios (100 size/book-to-market,
          49 industries): OLS on size, value and momentum; OLS on all ten
          characteristics and two market variables; elastic net; gradient-
          boosted trees; a neural network. Expanding-window training,
          tuning on a validation block, out-of-sample from 1975.
  Part B  Robustness: large portfolios only, subperiods, trading costs.
  Part C  Stability and interpretation: neural-network seeds, permutation
          importance of each characteristic.
  Part D  Complexity: market timing with random Fourier features of the
          Goyal-Welch predictors (the "virtue of complexity").

DATA (free)
  Kenneth R. French Data Library: 100 Portfolios Formed on Size and
  Book-to-Market (10 x 10), 49 Industry Portfolios, Fama/French 3 Factors.
  Goyal-Welch predictor data (Part D, optional).

USAGE
  python lab20.py               # about 10 minutes on a laptop
  python lab20.py --fast        # one seed, refit every 10 years: about 3 minutes
  python lab20.py --source sim
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
from fintest.ml import (CHARS, MACRO, build_panel, diebold_mariano, long_short, ls_stats, make_models,
                        permutation_importance, r2_oos, random_features_timing, simulated_panel, walk_forward)
from fintest.predictors import load_predictors, simulated_predictors

ROOT = Path(__file__).resolve().parent
BASE = "OLS-3 (size, value, momentum)"
COLORS = {BASE: plots.INK_3, "OLS (all)": plots.INK, "Elastic net": plots.AQUA, "Gradient-boosted trees": plots.GREEN,
          "Neural network": plots.VIOLET}


def _first(tables, *options):
    for words in options:
        try:
            return find_table(tables, *words)
        except KeyError:
            continue
    return None


def load(source: str, files: dict, seed: int = 0, sim_assets: int = 150, sim_months: int = 1188) -> dict:
    if source == "sim":
        return simulated_panel(n_assets=sim_assets, n_months=sim_months, seed=seed)
    ff = load_french_factors("ff3_monthly", path=files.get("ff3"))
    mkt = ff[[c for c in ff.columns if c.strip().upper() == "MKT-RF"][0]]
    rf = ff[[c for c in ff.columns if c.strip().upper() == "RF"][0]]
    parts = {"ret": [], "size": [], "nfirms": [], "bm": []}
    large, notes = [], []
    for ds, key, prefix in (("size_bm_100", "sbm100", "SBM"), ("industry_49", "ind49", "IND")):
        try:
            t = load_french_tables(ds, path=files.get(key))
        except DataUnavailable as e:
            notes.append(f"{ds} not loaded ({str(e).splitlines()[0]})")
            continue
        vw = find_table(t, "value", "weight")
        cols = [f"{prefix}:{c}" for c in vw.columns]
        ren = dict(zip(vw.columns, cols))
        parts["ret"].append(vw.rename(columns=ren).sub(rf, axis=0))
        size = _first(t, ("market", "cap"), ("firm", "size"))
        nf = _first(t, ("number", "firms"))
        bm = _first(t, ("be", "me"))
        for k, v in (("size", size), ("nfirms", nf), ("bm", bm)):
            if v is not None:
                parts[k].append(v.rename(columns=ren))
            else:
                notes.append(f"{ds}: no monthly table for {k}; that characteristic is set to the median")
        if prefix == "SBM":
            large += [c for i, c in enumerate(cols) if i // 10 >= 5]       # size deciles 6 to 10
        else:
            large += cols
    if not parts["ret"]:
        raise DataUnavailable("Neither portfolio file could be loaded.")
    cat = {k: (pd.concat(v, axis=1) if v else None) for k, v in parts.items()}
    return {"ret": cat["ret"], "mkt": mkt, "size": cat["size"], "nfirms": cat["nfirms"], "bm": cat["bm"],
            "large": large, "simulated": False, "notes": notes}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def evaluate(pred: pd.DataFrame, names, base: str = BASE) -> pd.DataFrame:
    rows = []
    for n in names:
        s = ls_stats(long_short(pred, n))
        rows.append({"model": n, "r2_oos_pct": 100 * r2_oos(pred["y"], pred[n]),
                     "dm_t_vs_ols3": float("nan") if n == base else diebold_mariano(pred, n, base),
                     "ls_mean_monthly": s["mean_monthly"], "ls_t": s["t"], "ls_sharpe": s["sharpe"],
                     "ls_turnover": s["turnover"]})
    return pd.DataFrame(rows)


def part_a(panel, models, test_start, refit_years, verbose=False):
    res = walk_forward(panel, models, test_start=test_start, refit_years=refit_years, verbose=verbose)
    return {"res": res, "table": evaluate(res["pred"], list(models))}


def part_b(pred: pd.DataFrame, names, large, costs=(0.001, 0.0025, 0.005)) -> dict:
    sub = pred[pred["asset"].isin(large)]
    big = evaluate(sub, names)
    periods = []
    mid = pd.Period("2000-01", "M")
    for label, m in (("1975–1999", pred.index < mid), ("2000 on", pred.index >= mid)):
        if m.sum() == 0:
            continue
        t = evaluate(pred[m], names)
        t.insert(0, "period", label)
        periods.append(t[["period", "model", "r2_oos_pct", "ls_mean_monthly", "ls_sharpe"]])
    crow = []
    for n in names:
        row = {"model": n}
        for c in (0.0,) + tuple(costs):
            row[f"sharpe_{c * 1e4:.0f}bp"] = ls_stats(long_short(pred, n, cost=c))["sharpe"]
        crow.append(row)
    return {"large": big, "periods": pd.concat(periods, ignore_index=True), "costs": pd.DataFrame(crow)}


def part_c(panel, fast: bool, test_start: str, refit_years: int, params: dict, split: str | None = None) -> dict:
    allm = make_models(fast=True)
    seeds = (0, 1) if fast else (0, 1, 2, 3)
    # reuse the network penalty chosen most often in Part A, so only the seed changes
    chosen = [p["alpha"] for p in params.get("Neural network", []) if "alpha" in p]
    alpha = max(set(chosen), key=chosen.count) if chosen else 1e-1
    rows = []
    for s in seeds:
        factory, _, feats, _ = allm["Neural network"]
        res = walk_forward(panel, {"Neural network": (factory, [{"alpha": alpha}], feats, [s])}, test_start=test_start,
                           refit_years=refit_years)
        p = res["pred"]
        st = ls_stats(long_short(p, "Neural network"))
        rows.append({"seed": s, "r2_oos_pct": 100 * r2_oos(p["y"], p["Neural network"]), "ls_sharpe": st["sharpe"]})
    seeds_t = pd.DataFrame(rows)
    # importance: fit once on data before `split`, measure on `split` onward (no overlap). Default split: 2000, or
    # the middle of the out-of-sample period if the data end earlier
    last = panel.index.max()
    if split is None:
        split = "2000-01" if last >= pd.Period("2010-01", "M") else \
            f"{(pd.Period(test_start, 'M').year + last.year) // 2}-01"
    cut = pd.Period(split, "M")
    train, test = panel[panel.index < cut - 1], panel[panel.index >= cut]
    imp = {}
    for n in ("Gradient-boosted trees", "Neural network"):
        res = walk_forward(panel[panel.index < cut + 1], {n: allm[n]}, test_start=split, refit_years=1)
        model, cols = res["fitted"][n]
        imp[n] = permutation_importance(model, cols, test)
    lin = make_models(fast=True)["OLS (all)"][0]({}, 0).fit(train[CHARS + MACRO].to_numpy(), train["ret_next_dm"].to_numpy())
    imp["OLS (all)"] = permutation_importance(lin, CHARS + MACRO, test)
    return {"seeds": seeds_t, "importance": pd.DataFrame(imp), "split": split}


def part_d(source: str, gw_file, seed: int = 0) -> pd.DataFrame | None:
    try:
        P = simulated_predictors(seed=seed) if source == "sim" else load_predictors(gw_file)
    except DataUnavailable as e:
        print(f"Part D skipped: {str(e).splitlines()[0]}")
        return None
    X = P.drop(columns=[c for c in ("eqp", "rf") if c in P.columns]).dropna(axis=1, how="all").ffill()
    keep = X.columns[X.notna().mean() > 0.9]
    X = X[keep].dropna()
    y = P["eqp"].reindex(X.index)
    return random_features_timing(X, y)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_r2(a, out, sim, test_start):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    t = a["table"]
    y = np.arange(len(t))[::-1]
    vals = t["r2_oos_pct"].to_numpy()
    ax.barh(y, vals, color=[COLORS.get(n, plots.BLUE) for n in t["model"]])
    for yy, v, dm in zip(y, vals, t["dm_t_vs_ols3"]):
        txt = f"{v:+.2f}%" + ("" if math.isnan(dm) else f"  (DM t = {dm:+.1f})")
        ax.text(max(v, 0) + 0.01 * max(1e-9, np.nanmax(np.abs(vals))), yy, txt, va="center", fontsize=8.5,
                color=plots.INK_2)
    ax.axvline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_yticks(y, t["model"], fontsize=9)
    lo, hi = min(0, vals.min()), max(0, vals.max())
    ax.set_xlim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.6 * (hi - lo + 1e-9))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Out-of-sample R² for next month's return relative to the cross-sectional average (%)")
    plots.titles(fig, "Lab Figure 20.1  How well does each model rank portfolios?",
                 _sub(sim, f"Forecasts from {test_start} · DM: Diebold–Mariano t-statistic against OLS-3 "
                           "(positive: lower forecast error)"))
    plots.finish(fig, out / "fig20_1_r2.png")
    plt.close(fig)


def fig_cumulative(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    pred = a["res"]["pred"]
    for n in a["table"]["model"]:
        ls = long_short(pred, n)
        ax.plot(ls.index.to_timestamp(), np.cumsum(np.log1p(ls["ls"])), color=COLORS.get(n, plots.BLUE),
                linewidth=1.5, label=n)
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_ylabel("Cumulative log return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 20.2  Long–short portfolios built from each model's forecasts",
                 _sub(sim, "Top 10% minus bottom 10% of portfolios by forecast, equal-weighted, before costs"))
    plots.finish(fig, out / "fig20_2_cumulative.png")
    plt.close(fig)


def fig_robust(b, out, sim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    fig.patch.set_facecolor(plots.SURFACE)
    names = list(b["costs"]["model"])
    y = np.arange(len(names))[::-1]
    per = b["periods"]
    ax = axes[0]
    plots.style_axes(ax)
    for k, (lab, col) in enumerate((("1975–1999", plots.INK_3), ("2000 on", plots.BLUE))):
        if not (per.period == lab).any():
            continue
        v = per[per.period == lab].set_index("model").loc[names, "ls_sharpe"]
        ax.barh(y + (0.2 if k == 0 else -0.2), v, 0.4, color=col, label=lab)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_title("By period, before costs", loc="left", fontsize=9.5, color=plots.INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("Long–short Sharpe ratio")
    ax = axes[1]
    plots.style_axes(ax)
    c = b["costs"].set_index("model").loc[names]
    cols = [x for x in c.columns if x.startswith("sharpe_")]
    shades = [plots.INK, plots.BLUE, plots.ORANGE, plots.RED]
    h = 0.8 / len(cols)
    for k, col in enumerate(cols):
        ax.barh(y + 0.4 - h * (k + 0.5), c[col], h, color=shades[k % 4], label=col.replace("sharpe_", "").replace("bp", " bp"))
    ax.set_title("Full period, by cost per unit of turnover", loc="left", fontsize=9.5, color=plots.INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("Long–short Sharpe ratio")
    for a_ in axes:
        a_.axvline(0, color=plots.INK_3, linewidth=0.8)
        a_.grid(axis="y", visible=False)
        lo, hi = a_.get_xlim()
        a_.set_xlim(min(lo, 0), hi + 0.45 * (hi - min(lo, 0)))
    plots.titles(fig, "Lab Figure 20.3  Does the edge survive later years and trading costs?",
                 _sub(sim, "Top 10% minus bottom 10% of portfolios by forecast · costs charged per unit of turnover"))
    plots.finish(fig, out / "fig20_3_robustness.png")
    plt.close(fig)


def fig_importance(c, out, sim):
    import matplotlib.pyplot as plt

    imp = c["importance"]
    fig, axes = plt.subplots(1, imp.shape[1], figsize=(3.4 * imp.shape[1] + 1, 4.6), sharey=True)
    fig.patch.set_facecolor(plots.SURFACE)
    order = imp.mean(axis=1).sort_values().index
    for ax, n in zip(np.atleast_1d(axes), imp.columns):
        plots.style_axes(ax)
        v = 100 * imp.loc[order, n]
        ax.barh(np.arange(len(v)), v, color=COLORS.get(n, plots.BLUE))
        ax.axvline(0, color=plots.INK_3, linewidth=0.8)
        ax.set_title(n, loc="left", fontsize=9.5, color=plots.INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Drop in R² when shuffled (pct. points)")
    np.atleast_1d(axes)[0].set_yticks(np.arange(len(order)), order, fontsize=8.5)
    plots.titles(fig, "Lab Figure 20.4  Which characteristics do the models use?",
                 _sub(sim, f"Permutation importance: models fitted on data before {c['split'][:4]}, measured from "
                           f"{c['split'][:4]} onward"))
    plots.finish(fig, out / "fig20_4_importance.png")
    plt.close(fig)


def fig_complexity(d, out, sim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    fig.patch.set_facecolor(plots.SURFACE)
    shades = [plots.INK_3, plots.BLUE, plots.VIOLET]
    for k, z in enumerate(sorted(d["ridge"].unique())):
        g = d[d.ridge == z]
        axes[0].plot(g["complexity_P_over_T"], 100 * g["r2_oos"], marker="o", color=shades[k % 3], label=f"ridge {z:g}")
        axes[1].plot(g["complexity_P_over_T"], g["timing_sharpe"], marker="o", color=shades[k % 3], label=f"ridge {z:g}")
    for ax, lab in zip(axes, ("Out-of-sample R² (%, symmetric log scale)", "Timing strategy Sharpe ratio")):
        plots.style_axes(ax)
        ax.set_xscale("log")
        if lab.startswith("Out"):
            ax.set_yscale("symlog", linthresh=1.0)
        ax.axhline(0, color=plots.INK_3, linewidth=0.8)
        ax.set_xlabel("Complexity: features per training observation (P/T, log scale)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=8.5)
    plots.titles(fig, "Lab Figure 20.5  Does more complexity help market timing?",
                 _sub(sim, "Ridge on random Fourier features of the Goyal–Welch predictors, 12-month rolling window"))
    plots.finish(fig, out / "fig20_5_complexity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


FMT = {"r2_oos_pct": _num("+.3f"), "dm_t_vs_ols3": _num("+.2f"), "ls_mean_monthly": _num("+.2%"), "ls_t": _num("+.2f"),
       "ls_sharpe": _num(".2f"), "ls_turnover": _num(".2f")}


def write_report(out, a, b, c, d, sim, notes, args):
    md = ["# Lab 20 results: Does Machine Learning Find Hidden Alpha?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    if notes:
        md += ["Notes: " + "; ".join(notes), ""]
    md += ["**What you will explore:** whether machine learning finds return predictability that linear models miss, and whether it survives costs.", "",
           f"## Part A: The horse race (forecasts from {args.test_start}, refit every {args.refit_years} years)", "",
           "Target: next month's excess return minus that month's cross-sectional average. R² is pooled against a "
           "forecast of zero. Long–short: top 10% minus bottom 10% by forecast.", "",
           _md(a["table"], FMT), "", "![Lab Figure 20.1](fig20_1_r2.png)", "", "![Lab Figure 20.2](fig20_2_cumulative.png)", "",
           "## Part B: Robustness", "", "**Large portfolios only** (size deciles 6–10 and industries)", "",
           _md(b["large"], FMT), "", "**Subperiods**", "", _md(b["periods"], FMT), "",
           "**Long–short Sharpe ratio net of costs per unit of turnover**", "",
           _md(b["costs"], {k: _num(".2f") for k in b["costs"].columns if k != "model"}), "",
           "![Lab Figure 20.3](fig20_3_robustness.png)", "", "## Part C: Stability and interpretation", "",
           "**Neural network, one seed at a time** (refit every 10 years, penalty fixed at Part A's most common choice)", "",
           _md(c["seeds"], {"seed": lambda v: str(int(v)), "r2_oos_pct": _num("+.3f"), "ls_sharpe": _num(".2f")}), "",
           f"**Permutation importance** (drop in R², percentage points; fitted before {c.get('split', '2000')[:4]}, "
           f"measured from {c.get('split', '2000')[:4]})", "",
           _md((100 * c["importance"]).reset_index().rename(columns={"index": "feature"}),
               {k: _num("+.4f") for k in c["importance"].columns}), "", "![Lab Figure 20.4](fig20_4_importance.png)", ""]
    if d is not None:
        md += ["## Part D: Complexity in market timing", "",
               _md(d, {"P": lambda v: str(int(v)), "months": lambda v: str(int(v)), "complexity_P_over_T": _num(".2f"),
                       "ridge": _num("g"), "r2_oos": _num("+.2%"), "timing_sharpe": _num(".2f"), "alpha_t": _num("+.2f")}),
               "", "![Lab Figure 20.5](fig20_5_complexity.png)", ""]
    md += ["## Questions to think about", "", "1. Do the nonlinear models beat the linear ones out of sample, and by how much?",
           "2. Does the edge survive large portfolios, recent years and trading costs?",
           "3. Is the gain hidden alpha, or a better way of combining characteristics everyone already knows?", ""]
    (out / "lab20_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    for f in ("ff3", "sbm100", "ind49", "gw"):
        ap.add_argument(f"--{f}-file", dest=f, default=None)
    ap.add_argument("--test-start", default="1975-01")
    ap.add_argument("--refit-years", type=int, default=None, help="default 5 (10 with --fast)")
    ap.add_argument("--fast", action="store_true", help="one seed and refits every 10 years")
    ap.add_argument("--skip-c", action="store_true", help="skip Part C (seeds and importance)")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--sim-assets", type=int, default=150, help=argparse.SUPPRESS)
    ap.add_argument("--sim-months", type=int, default=1188, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(ROOT / "report"))
    args = ap.parse_args(argv)
    if args.refit_years is None:
        args.refit_years = 10 if args.fast else 5
    return args


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, {k: getattr(args, k) for k in ("ff3", "sbm100", "ind49")}, args.sim_seed,
                 args.sim_assets, args.sim_months)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    panel = build_panel(d["ret"], d["mkt"], d.get("size"), d.get("nfirms"), d.get("bm"))
    print(f"Panel: {panel['asset'].nunique()} portfolios, {panel.index.nunique()} months, {len(panel)} rows")
    models = make_models(fast=args.fast)
    a = part_a(panel, models, args.test_start, args.refit_years, verbose=True)
    names = list(models)
    b = part_b(a["res"]["pred"], names, d["large"])
    c = None if args.skip_c else part_c(panel, args.fast, args.test_start, 10, a["res"]["params"])
    dd = part_d(args.source, args.gw, args.sim_seed)
    fig_r2(a, out, sim, args.test_start)
    fig_cumulative(a, out, sim)
    fig_robust(b, out, sim)
    if c is not None:
        fig_importance(c, out, sim)
    if dd is not None:
        fig_complexity(dd, out, sim)
    a["table"].to_csv(out / "table_main.csv", index=False)
    b["large"].to_csv(out / "table_large.csv", index=False)
    b["periods"].to_csv(out / "table_periods.csv", index=False)
    b["costs"].to_csv(out / "table_costs.csv", index=False)
    if c is not None:
        c["seeds"].to_csv(out / "table_seeds.csv", index=False)
        c["importance"].to_csv(out / "table_importance.csv")
    if dd is not None:
        dd.to_csv(out / "table_complexity.csv", index=False)
    if c is None:
        c = {"seeds": pd.DataFrame(columns=["seed", "r2_oos_pct", "ls_sharpe"]), "importance": pd.DataFrame()}
    write_report(out, a, b, c, dd, sim, d.get("notes", []), args)
    summary = {"simulated": sim, "main": a["table"].to_dict(orient="records"),
               "large": b["large"].to_dict(orient="records"), "params": a["res"]["params"],
               "seeds": c["seeds"].to_dict(orient="records"),
               "complexity": None if dd is None else dd.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a["table"].round(3).to_string(index=False))
    print(f"Report: {out / 'lab20_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
