"""Lab 9: Do Optimized Portfolios Beat Naive Ones?

Chapter 9 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  How optimized portfolio strategies compare with equal weighting (1/N) out
  of sample.

WHAT THE LAB DOES
  Part A  Nine allocation rules on ten industry portfolios, walk-forward from
          rolling ten-year windows: 1/N, sample mean-variance (unconstrained
          and long-only), minimum variance (sample, long-only, Ledoit-Wolf),
          inverse-volatility risk parity, equal risk contribution and
          hierarchical risk parity.
  Part B  Other asset sets: 30 industries and 25 size/book-to-market
          portfolios, net of trading costs.
  Part C  Stress tests: estimation window, costs, rebalancing frequency,
          crisis periods.
  Part D  Why: how unstable the optimized weights are.

DATA (free)
  Kenneth R. French Data Library: 10 and 30 Industry Portfolios, 25
  Portfolios Formed on Size and Book-to-Market, Fama/French 3 Factors (for
  the risk-free rate).

USAGE
  python lab09.py
  python lab09.py --source sim
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
from fintest.portfolio import RULES, simulated_assets, summarize, walk_forward

ROOT = Path(__file__).resolve().parent
FAST = ["1/N", "Mean-variance (sample)", "Minimum variance (sample)", "Minimum variance, Ledoit–Wolf",
        "Risk parity (inverse volatility)", "Hierarchical risk parity"]
COLORS = dict(zip(RULES, [plots.INK, plots.RED, plots.ORANGE, plots.BLUE, plots.AQUA, plots.VIOLET, plots.YELLOW,
                          plots.MAGENTA, plots.GREEN]))


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        return {"sets": {"10 industries": simulated_assets(n_assets=10, seed=seed),
                         "30 industries": simulated_assets(n_assets=30, seed=seed + 1),
                         "25 size/book-to-market": simulated_assets(n_assets=25, seed=seed + 2)}, "simulated": True}
    ff = load_french_factors("ff3_monthly", path=files.get("ff3"))
    rf = ff[[c for c in ff.columns if c.strip().upper() == "RF"][0]]
    sets = {}
    for label, ds, key in (("10 industries", "industry_10", "ind10"), ("30 industries", "industry_30", "ind30"),
                           ("25 size/book-to-market", "size_bm_25", "sbm25")):
        try:
            t = load_french_tables(ds, path=files.get(key))
            vw = find_table(t, "value", "weight")
        except (DataUnavailable, KeyError) as e:
            print(f"{label} not loaded ({str(e).splitlines()[0]})")
            continue
        ex = vw.sub(rf.reindex(vw.index), axis=0).dropna()
        sets[label] = ex
    if not sets:
        raise DataUnavailable("No asset sets could be loaded.")
    return {"sets": sets, "simulated": False}


def _rules(names):
    return {n: RULES[n] for n in names}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part_a(R: pd.DataFrame, window: int, names) -> dict:
    res = walk_forward(R, _rules(names), window=window)
    return {"res": res, "table": summarize(res)}


def net_of_costs(res: dict, cost: float) -> dict:
    return {n: {"ret": r["ret"] - cost * r["turnover"], "turnover": r["turnover"], "weights": r["weights"]}
            for n, r in res.items()}


def part_b(sets: dict, window: int, names, cost: float, first_res: dict | None = None) -> pd.DataFrame:
    rows = []
    for label, R in sets.items():
        res = first_res if (first_res is not None and label == next(iter(sets))) else walk_forward(R, _rules(names), window)
        t = summarize(net_of_costs(res, cost))
        t.insert(0, "assets", label)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def part_c(R: pd.DataFrame, names, base_res: dict, windows=(60, 120, 240), costs=(0.0, 0.0025, 0.005),
           every=(1, 3, 12)) -> dict:
    wrows = []
    for w in windows:
        res = base_res if w == 120 else walk_forward(R, _rules(names), window=w)
        t = summarize(res)
        t.insert(0, "window_months", w)
        wrows.append(t[["window_months", "rule", "sharpe", "ce", "turnover_monthly"]])
    crow = []
    for k in costs:
        t = summarize(net_of_costs(base_res, k))
        t.insert(0, "cost_bp", 1e4 * k)
        crow.append(t[["cost_bp", "rule", "sharpe", "ce"]])
    erow = []
    for e in every:
        res = base_res if e == 1 else walk_forward(R, _rules(names), window=120, every=e)
        t = summarize(net_of_costs(res, 0.0025))
        t.insert(0, "rebalance_months", e)
        erow.append(t[["rebalance_months", "rule", "sharpe", "turnover_monthly"]])
    crisis = []
    for label, a, b in (("2008 (Sep 2008–Feb 2009)", "2008-09", "2009-02"), ("2020 (Feb–Mar 2020)", "2020-02", "2020-03"),
                        ("2022", "2022-01", "2022-12")):
        row = {"period": label}
        for n, r in base_res.items():
            x = r["ret"].loc[pd.Period(a, "M"):pd.Period(b, "M")]
            row[n] = float(np.prod(1 + x) - 1) if len(x) else float("nan")
        crisis.append(row)
    return {"windows": pd.concat(wrows, ignore_index=True), "costs": pd.concat(crow, ignore_index=True),
            "rebalance": pd.concat(erow, ignore_index=True), "crisis": pd.DataFrame(crisis)}


def part_d(res: dict, R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, r in res.items():
        W = r["weights"]
        rows.append({"rule": n, "avg_gross_exposure": float(np.abs(W).sum(axis=1).mean()),
                     "avg_max_weight": float(np.abs(W).max(axis=1).mean()),
                     "share_negative_weights": float((W < -1e-9).mean()),
                     "weight_volatility": float(np.diff(W, axis=0).std()), "avg_turnover": float(r["turnover"][1:].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_sharpe(a, label, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    t = a["table"]
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["sharpe"], color=[COLORS.get(n, plots.BLUE) for n in t["rule"]])
    s0 = float(t.loc[t.rule == "1/N", "sharpe"].iloc[0])
    ax.axvline(s0, color=plots.INK, linestyle="--", linewidth=1.2)
    for yy, (_, row) in zip(y, t.iterrows()):
        txt = f"{row.sharpe:.2f}" + ("" if row.rule == "1/N" else f"  (p = {row.p:.2f})")
        ax.text(max(row.sharpe, s0, 0) + 0.01, yy, txt, va="center", fontsize=8.5, color=plots.INK_2)
    ax.set_yticks(y, t["rule"], fontsize=9)
    ax.set_xlim(min(0, t["sharpe"].min() * 1.2), t["sharpe"].max() * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Out-of-sample Sharpe ratio (annualized, before costs)")
    plots.titles(fig, "Lab Figure 9.1  Optimized portfolios against 1/N",
                 _sub(sim, f"{label} · weights from rolling 10-year windows, rebalanced monthly · dashed: 1/N · "
                           "p: Jobson–Korkie–Memmel test against 1/N"))
    plots.finish(fig, out / "fig9_1_sharpe.png")
    plt.close(fig)


def fig_turnover(a, out, sim):
    fig, ax, plt = plots.new_figure(10.0, 4.8)
    t = a["table"]
    for _, row in t.iterrows():
        ax.scatter(row.turnover_monthly, row.sharpe, s=60, color=COLORS.get(row.rule, plots.BLUE), zorder=3,
                   label=row.rule)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Average monthly turnover (sum of absolute weight changes; log scale)")
    ax.set_ylabel("Sharpe ratio")
    plots.titles(fig, "Lab Figure 9.2  Turnover and out-of-sample Sharpe ratio",
                 _sub(sim, "Each point is one allocation rule · before trading costs"))
    plots.finish(fig, out / "fig9_2_turnover.png")
    plt.close(fig)


def fig_heatmap(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    piv = b.pivot_table(index="rule", columns="assets", values="sharpe", sort=False)
    base = piv.loc["1/N"]
    diff = piv.sub(base, axis=1).drop(index="1/N")
    lim = max(0.2, float(np.nanmax(np.abs(diff.to_numpy()))))
    im = ax.imshow(diff.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.6 * lim else plots.INK)
    ax.set_xticks(range(diff.shape[1]), diff.columns, fontsize=9)
    ax.set_yticks(range(diff.shape[0]), diff.index, fontsize=8.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plots.titles(fig, "Lab Figure 9.3  Sharpe ratio minus that of 1/N, net of costs",
                 _sub(sim, "Blue: beat 1/N · red: lost to 1/N · costs per unit of turnover as set in the run"))
    plots.finish(fig, out / "fig9_3_heatmap.png")
    plt.close(fig)


def fig_windows(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    w = c["windows"]
    for n in w["rule"].unique():
        g = w[w.rule == n]
        ax.plot(g["window_months"] / 12, g["sharpe"], marker="o", color=COLORS.get(n, plots.BLUE), linewidth=1.5, label=n)
    ax.set_xlabel("Estimation window (years)")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    plots.titles(fig, "Lab Figure 9.4  Does more data help the optimizers?",
                 _sub(sim, "Ten industry portfolios, before costs"))
    plots.finish(fig, out / "fig9_4_windows.png")
    plt.close(fig)


def fig_weights(res, R, out, sim):
    import matplotlib.pyplot as plt

    names = [n for n in ("Mean-variance (sample)", "Minimum variance, Ledoit–Wolf", "1/N") if n in res]
    fig, axes = plt.subplots(len(names), 1, figsize=(9.0, 2.2 * len(names) + 1), sharex=True)
    fig.patch.set_facecolor(plots.SURFACE)
    cols = list(R.columns)
    palette = plt.get_cmap("tab10" if len(cols) <= 10 else "tab20")
    for ax, n in zip(np.atleast_1d(axes), names):
        plots.style_axes(ax)
        W = res[n]["weights"]
        dates = res[n]["ret"].index.to_timestamp()
        for j in range(W.shape[1]):
            ax.plot(dates, W[:, j], linewidth=0.9, color=palette(j % 20))
        ax.axhline(0, color=plots.INK_3, linewidth=0.8)
        ax.set_title(n, loc="left", fontsize=9.5, color=plots.INK)
        if n == "1/N":
            ax.set_ylim(0, 2.5 / W.shape[1])
        ax.set_ylabel("Weight")
    plots.titles(fig, "Lab Figure 9.5  How stable are the weights?",
                 _sub(sim, "Each line is one industry's weight, re-estimated monthly from the previous 10 years"))
    plots.finish(fig, out / "fig9_5_weights.png")
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


SUMFMT = {"mean_annual": _num("+.1%"), "vol_annual": _num(".1%"), "sharpe": _num(".2f"), "ce": _num("+.1%"),
          "turnover_monthly": _num(".2f"), "max_weight": _num(".2f"), "min_weight": _num(".2f"),
          "max_drawdown": _num(".0%"), "sharpe_vs_1N_z": _num("+.2f"), "p": _num(".2f")}


def write_report(out, label, a, b, c, dd, sim, cost):
    md = ["# Lab 9 results: Do Optimized Portfolios Beat Naive Ones?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** how optimized portfolio strategies compare with equal weighting (1/N) out of sample.",
           "", f"## Part A: Nine rules on {label}", "",
           "Weights estimated each month from the previous 120 months of excess returns and held for one month. CE: "
           "certainty-equivalent return for risk aversion 1. Sharpe-ratio test against 1/N: Jobson–Korkie with Memmel's "
           "correction.", "", _md(a["table"], SUMFMT), "", "![Lab Figure 9.1](fig9_1_sharpe.png)", "",
           "![Lab Figure 9.2](fig9_2_turnover.png)", "", f"## Part B: Other asset sets, net of {cost * 1e4:.0f} bp per unit "
           "of turnover", "", _md(b[["assets", "rule", "sharpe", "ce", "turnover_monthly", "sharpe_vs_1N_z", "p"]], SUMFMT),
           "", "![Lab Figure 9.3](fig9_3_heatmap.png)", "", "## Part C: Stress tests", "", "**Estimation window**", "",
           _md(c["windows"], SUMFMT), "", "![Lab Figure 9.4](fig9_4_windows.png)", "", "**Trading costs**", "",
           _md(c["costs"], {**SUMFMT, "cost_bp": _num(".0f")}), "", "**Rebalancing frequency** (net of 25 bp)", "",
           _md(c["rebalance"], SUMFMT), "", "**Crisis periods (cumulative excess return)**", "",
           _md(c["crisis"], {k: _num("+.1%") for k in c["crisis"].columns if k != "period"}), "",
           "## Part D: Weight stability", "",
           _md(dd, {"avg_gross_exposure": _num(".2f"), "avg_max_weight": _num(".2f"), "share_negative_weights": _num(".0%"),
                    "weight_volatility": _num(".3f"), "avg_turnover": _num(".2f")}), "",
           "![Lab Figure 9.5](fig9_5_weights.png)", "", "## Questions to think about", "",
           "1. Does any optimized rule beat 1/N significantly, before and after costs?",
           "2. Which repairs (shrinkage, constraints, risk-based rules) help most, and by how much?",
           "3. Would you replace a simple rule with an optimizer, given this evidence?", ""]
    (out / "lab09_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    for f in ("ff3", "ind10", "ind30", "sbm25"):
        ap.add_argument(f"--{f}-file", dest=f, default=None)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--cost", type=float, default=0.0025, help="cost per unit of turnover in Part B (default 25 bp)")
    ap.add_argument("--fast", action="store_true", help="skip the slower constrained rules")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = load(args.source, {k: getattr(args, k) for k in ("ff3", "ind10", "ind30", "sbm25")}, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    names = FAST if args.fast else list(RULES)
    label, R = next(iter(d["sets"].items()))
    a = part_a(R, args.window, names)
    b = part_b(d["sets"], args.window, names, args.cost, first_res=a["res"])
    c = part_c(R, [n for n in names if n in FAST], {n: a["res"][n] for n in names if n in FAST})
    dd = part_d(a["res"], R)
    fig_sharpe(a, label, out, sim)
    fig_turnover(a, out, sim)
    fig_heatmap(b, out, sim)
    fig_windows(c, out, sim)
    fig_weights(a["res"], R, out, sim)
    a["table"].to_csv(out / "table_main.csv", index=False)
    b.to_csv(out / "table_asset_sets.csv", index=False)
    for k in ("windows", "costs", "rebalance", "crisis"):
        c[k].to_csv(out / f"table_{k}.csv", index=False)
    dd.to_csv(out / "table_weights.csv", index=False)
    write_report(out, label, a, b, c, dd, sim, args.cost)
    summary = {"simulated": sim, "main": a["table"].to_dict(orient="records"), "asset_sets": b.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a["table"][["rule", "sharpe", "ce", "turnover_monthly", "p"]].round(3).to_string(index=False))
    print(f"Report: {out / 'lab09_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
