"""Lab 16: Two Standards of Evidence for Published Anomalies.

Chapter 16 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  How the share of published predictors that hold up changes when each
  side's standards of evidence are applied to the same data.

WHAT THE LAB DOES
  Part A  Original samples: how many published
          predictors clear t > 1.96, t > 3 and a false-discovery-rate test.
  Part B  Out of sample: returns after each paper's sample ended and after
          it was published (McLean and Pontiff 2016).
  Part C  Methodology: the same signals under a second portfolio
          construction (for example value-weighted, micro-caps excluded),
          if you supply it.
  Part D  Bayesian shrinkage: how much of each in-sample mean to believe,
          and how well the shrunk estimates predict post-publication returns.
  Part E  Known truth: a simulated literature in which only significant
          results get published. How much "decay" does selection alone create?

DATA
  Open Source Asset Pricing (Chen and Zimmermann 2022): long-short portfolio
  returns (wide, e.g. PredictorLSretWide.csv, or long format) and
  SignalDoc.csv. See README.

USAGE
  python lab16.py --returns data/PredictorLSretWide.csv --signal-doc data/SignalDoc.csv
  python lab16.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.anomalies import (
    empirical_bayes,
    mclean_pontiff,
    read_ls_returns,
    read_signal_doc,
    signal_periods,
    simulated_anomalies,
)
from fintest.data import DataUnavailable
from fintest.multiple import benjamini_hochberg, one_sided_pvalues

ROOT = Path(__file__).resolve().parent


def part_a(stats: pd.DataFrame) -> dict:
    t = stats["t_in"].dropna()
    p = one_sided_pvalues(t.to_numpy())
    bh = benjamini_hochberg(p, 0.025)
    rows = [{"rule": "t > 1.96 (conventional)", "share": float((t > 1.96).mean()), "count": int((t > 1.96).sum())},
            {"rule": "t > 3 (Harvey, Liu and Zhu)", "share": float((t > 3).mean()), "count": int((t > 3).sum())},
            {"rule": "Benjamini–Hochberg, 2.5% FDR", "share": float(bh.mean()), "count": int(bh.sum())}]
    return {"table": pd.DataFrame(rows), "n": len(t), "t": t}


def part_b(ret, doc, stats) -> dict:
    mp = mclean_pontiff(ret, doc, stats)
    s = stats[stats["mean_in"] > 0]
    ok = s[["mean_in", "mean_post_pub"]].dropna()
    fit = np.polyfit(ok["mean_in"], ok["mean_post_pub"], 1) if len(ok) > 5 else (float("nan"), float("nan"))
    slope, intercept = float(fit[0]), float(fit[1])
    summary = pd.DataFrame([
        {"period": "In-sample", "avg_mean_monthly": s["mean_in"].mean(), "share_positive": float((s["mean_in"] > 0).mean()),
         "share_t_gt_196": float((s["t_in"] > 1.96).mean())},
        {"period": "Post-sample, before publication", "avg_mean_monthly": s["mean_post_sample"].mean(),
         "share_positive": float((s["mean_post_sample"] > 0).mean()),
         "share_t_gt_196": float((s["t_post_sample"] > 1.96).mean())},
        {"period": "Post-publication", "avg_mean_monthly": s["mean_post_pub"].mean(),
         "share_positive": float((s["mean_post_pub"] > 0).mean()),
         "share_t_gt_196": float((s["t_post_pub"] > 1.96).mean())}])
    return {"mp": mp, "summary": summary, "slope_post_on_in": slope, "intercept": intercept, "pairs": ok,
            "ratio": float(ok["mean_post_pub"].mean() / ok["mean_in"].mean()) if len(ok) else float("nan")}


def part_c(stats_a: pd.DataFrame, stats_b: pd.DataFrame | None, label_a: str, label_b: str) -> pd.DataFrame | None:
    if stats_b is None:
        return None
    common = stats_a.index.intersection(stats_b.index)
    rows = []
    for label, st in ((label_a, stats_a.loc[common]), (label_b, stats_b.loc[common])):
        t = st["t_in"].dropna()
        p = one_sided_pvalues(t.to_numpy())
        rows.append({"construction": label, "signals": len(t), "share_t_gt_196": float((t > 1.96).mean()),
                     "share_t_gt_3": float((t > 3).mean()), "share_bh": float(benjamini_hochberg(p, 0.025).mean()),
                     "median_mean_monthly": float(st["mean_in"].median()),
                     "share_post_pub_positive": float((st["mean_post_pub"] > 0).mean())})
    return pd.DataFrame(rows)


def part_d(stats: pd.DataFrame) -> dict:
    s = stats.dropna(subset=["mean_in", "t_in"]).copy()
    se = s["mean_in"] / s["t_in"]
    eb = empirical_bayes(s["mean_in"].to_numpy(), se.to_numpy())
    s["post_mean"], s["p_positive"], s["shrink"] = eb["post_mean"], eb["p_positive"], eb["shrink"]
    ok = s.dropna(subset=["mean_post_pub"])
    err_raw = float(np.sqrt(np.mean((ok["mean_post_pub"] - ok["mean_in"]) ** 2)))
    err_eb = float(np.sqrt(np.mean((ok["mean_post_pub"] - ok["post_mean"]) ** 2)))
    return {"table": s, "mu0": eb["mu0"], "tau": eb["tau"], "share_p_gt_95": float((s["p_positive"] > 0.95).mean()),
            "avg_shrink": float(s["shrink"].mean()), "rmse_raw": err_raw, "rmse_shrunk": err_eb}


def part_e(seed: int = 0, n_signals: int = 200) -> pd.DataFrame:
    rows = []
    for decay in (0.0, 0.3):
        ret, doc, truth = simulated_anomalies(n_signals=n_signals, post_pub_decay=decay, seed=seed)
        st = signal_periods(ret, doc)
        mp = mclean_pontiff(ret, doc, st)
        rows.append({"true_post_publication_decay": decay, "measured_post_sample_change": mp["coef"]["post_sample"],
                     "measured_post_pub_change": mp["coef"]["post_pub"], "share_published_false": float((truth == 0).mean()),
                     "share_t_gt_3": float((st["t_in"] > 3).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_tstats(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = a["t"].clip(-4, 12)
    ax.hist(t, bins=np.arange(-4, 12.5, 0.5), color=plots.BLUE, edgecolor=plots.SURFACE)
    for v, lab in ((1.96, "1.96"), (3.0, "3.0")):
        ax.axvline(v, color=plots.INK_2, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(v, ax.get_ylim()[1] * 0.95, f" t = {lab}", fontsize=8.5, color=plots.INK_2, va="top")
    ax.set_xlabel("t-statistic of the long-short return in the original paper's sample")
    ax.set_ylabel("Signals")
    plots.titles(fig, "Lab Figure 16.1  How strong was the original evidence?",
                 _sub(sim, f"{a['n']} published predictors, measured over each paper's own sample period"))
    plots.finish(fig, out / "fig16_1_tstats.png")
    plt.close(fig)


def fig_in_vs_post(b, out, sim):
    fig, ax, plt = plots.new_figure(8.4, 5.2)
    p = b["pairs"] * 1200
    ax.scatter(p["mean_in"], p["mean_post_pub"], s=16, color=plots.BLUE, alpha=0.6, linewidths=0)
    lim = [min(p.min().min(), 0) - 0.5, p.max().max() + 0.5]
    ax.plot(lim, lim, color=plots.INK_3, linewidth=1, linestyle=(0, (4, 3)))
    ax.plot(lim, [b["intercept"] * 1200 + b["slope_post_on_in"] * v for v in lim], color=plots.RED, linewidth=1.6)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.set_xlabel("In-sample average return (% per year)")
    ax.set_ylabel("Post-publication average return (% per year)")
    plots.titles(fig, "Lab Figure 16.2  What carried over after publication",
                 _sub(sim, f"Dashed: full persistence (45° line) · red: fitted line, slope {b['slope_post_on_in']:.2f} · "
                           f"average post-publication return = {b['ratio']:.0%} of the in-sample average"))
    plots.finish(fig, out / "fig16_2_in_vs_post.png")
    plt.close(fig)


def fig_decay(b, e, out, sim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(plots.SURFACE)
    ax = axes[0]
    plots.style_axes(ax)
    c = b["mp"]["coef"]
    vals = [1.0, 1 + c["post_sample"], 1 + c["post_pub"]]
    ax.bar(["In-sample", "Post-sample", "Post-publication"], vals, color=[plots.BLUE, plots.AQUA, plots.ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, color=plots.INK)
    plots.percent_axis(ax)
    ax.set_title("Returns relative to the in-sample mean (data)", loc="left", fontsize=10, color=plots.INK)
    ax = axes[1]
    plots.style_axes(ax)
    x = np.arange(len(e))
    ax.bar(x - 0.18, e["measured_post_sample_change"], width=0.36, color=plots.AQUA, label="Post-sample")
    ax.bar(x + 0.18, e["measured_post_pub_change"], width=0.36, color=plots.ORANGE, label="Post-publication")
    ax.set_xticks(x, [f"true decay {v:.0%}" for v in e["true_post_publication_decay"]])
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Measured change when the truth is known (simulation)", loc="left", fontsize=10, color=plots.INK)
    plots.titles(fig, "Lab Figure 16.3  Decay after the sample ends and after publication",
                 _sub(sim, "Left: pooled McLean–Pontiff regression · right: selection alone produces apparent decay "
                           "even when nothing changes"))
    plots.finish(fig, out / "fig16_3_decay.png")
    plt.close(fig)


def fig_shrinkage(d, out, sim):
    fig, ax, plt = plots.new_figure(8.4, 5.0)
    t = d["table"]
    ax.scatter(t["mean_in"] * 1200, t["post_mean"] * 1200, s=16, color=plots.VIOLET, alpha=0.6, linewidths=0)
    lim = [t["mean_in"].min() * 1200 - 0.5, t["mean_in"].max() * 1200 + 0.5]
    ax.plot(lim, lim, color=plots.INK_3, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(d["mu0"] * 1200, color=plots.ORANGE, linewidth=1)
    ax.set_xlabel("In-sample average return (% per year)")
    ax.set_ylabel("Shrunk estimate (% per year)")
    plots.titles(fig, "Lab Figure 16.4  How much to believe each predictor",
                 _sub(sim, f"Empirical Bayes: estimates pulled toward the average ({d['mu0'] * 1200:.1f}% a year, orange) "
                           f"in proportion to their noise · {d['share_p_gt_95']:.0%} have >95% probability of a positive "
                           "true mean"))
    plots.finish(fig, out / "fig16_4_shrinkage.png")
    plt.close(fig)


def _md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _num(spec):
    return lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else format(v, spec)


def write_report(out, a, b, c, d, e, sim, labels):
    md = ["# Lab 16 results: Two Standards of Evidence for Published Anomalies", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** how the share of published predictors that hold up changes when each side's standards of evidence are applied to the same data.", "",
           "## Part A: The original samples", "", f"{a['n']} predictors.", "",
           _md(a["table"], {"share": _num(".0%")}), "", "![Lab Figure 16.1](fig16_1_tstats.png)", ""]
    mp = b["mp"]
    md += ["## Part B: After the sample, after publication", "",
           _md(b["summary"], {"avg_mean_monthly": lambda v: f"{v * 1200:+.2f}% a year", "share_positive": _num(".0%"),
                              "share_t_gt_196": _num(".0%")}), "",
           f"Pooled regression of returns scaled by each signal's in-sample mean ({mp['n_signals']} signals, "
           f"{mp['n_obs']:,} signal-months, month-clustered t): post-sample change {mp['coef']['post_sample']:+.0%} "
           f"(t = {mp['t']['post_sample']:.1f}); post-publication change {mp['coef']['post_pub']:+.0%} "
           f"(t = {mp['t']['post_pub']:.1f}). McLean and Pontiff (2016) report −26% and −58%.", "",
           "![Lab Figure 16.2](fig16_2_in_vs_post.png)", "", "![Lab Figure 16.3](fig16_3_decay.png)", ""]
    md += ["## Part C: Methodology", ""]
    if c is None:
        md += ["Not run: supply a second construction with `--returns-alt` (for example the value-weighted or "
               "'ex NYSE 20th percentile size' implementation from Open Source Asset Pricing).", ""]
    else:
        md += [_md(c, {k: _num(".0%") for k in ("share_t_gt_196", "share_t_gt_3", "share_bh", "share_post_pub_positive")}
                   | {"median_mean_monthly": lambda v: f"{v * 1200:+.2f}% a year"}), ""]
    md += ["## Part D: Bayesian shrinkage", "",
           f"- Prior mean {d['mu0'] * 1200:+.2f}% a year; dispersion of true means (tau) {d['tau'] * 1200:.2f}% a year.",
           f"- Average shrinkage toward the prior: {d['avg_shrink']:.0%}.",
           f"- Share with posterior probability of a positive true mean above 95%: {d['share_p_gt_95']:.0%}.",
           f"- Predicting post-publication means: RMSE {d['rmse_raw'] * 1200:.2f}% a year using raw in-sample means, "
           f"{d['rmse_shrunk'] * 1200:.2f}% using shrunk ones.", "",
           "Caution: the prior here is estimated from the published signals themselves, which were selected for "
           "strong in-sample results. Shrinkage toward their average reduces noise but does not undo that selection; "
           "Part E shows how large the selection effect can be.", "", "![Lab Figure 16.4](fig16_4_shrinkage.png)", ""]
    md += ["## Part E: Known truth (simulation)", "",
           "A simulated literature in which only signals with in-sample t > 2 are published:", "",
           _md(e, {"true_post_publication_decay": _num(".0%"), "measured_post_sample_change": _num("+.0%"),
                   "measured_post_pub_change": _num("+.0%"), "share_published_false": _num(".0%"),
                   "share_t_gt_3": _num(".0%")}), "",
           "Even with no true decay, published signals appear to decline out of sample: selection on significance "
           "inflates in-sample means.", ""]
    md += ["## Questions to think about", "",
           "1. Under which standard (t > 1.96, t > 3, FDR, shrinkage) do most predictors survive? Under which do most fail?",
           "2. How much of the post-publication decline is data mining (post-sample) and how much is something else?",
           "3. Which standard of evidence would you apply, and which methodological choices drive the answer?", ""]
    (out / "lab16_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["osap", "sim"], default="osap")
    ap.add_argument("--returns", default="data/PredictorLSretWide.csv")
    ap.add_argument("--signal-doc", default="data/SignalDoc.csv")
    ap.add_argument("--returns-alt", default=None, help="optional second construction of the same signals")
    ap.add_argument("--label", default="Original papers' construction")
    ap.add_argument("--label-alt", default="Alternative construction")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.source == "sim":
            ret, doc, _ = simulated_anomalies(seed=0, post_pub_decay=0.3)
            ret_alt = None
            sim = True
        else:
            ret = read_ls_returns(ROOT / args.returns if not Path(args.returns).is_absolute() else args.returns)
            doc = read_signal_doc(ROOT / args.signal_doc if not Path(args.signal_doc).is_absolute() else args.signal_doc)
            ret_alt = read_ls_returns(args.returns_alt) if args.returns_alt else None
            sim = False
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    stats = signal_periods(ret, doc)
    a = part_a(stats)
    b = part_b(ret, doc, stats)
    c = part_c(stats, signal_periods(ret_alt, doc) if ret_alt is not None else None, args.label, args.label_alt)
    d = part_d(stats)
    e = part_e(args.seed)
    fig_tstats(a, out, sim)
    fig_in_vs_post(b, out, sim)
    fig_decay(b, e, out, sim)
    fig_shrinkage(d, out, sim)
    stats.to_csv(out / "table_signals.csv")
    d["table"].to_csv(out / "table_shrinkage.csv")
    e.to_csv(out / "table_simulation.csv", index=False)
    write_report(out, a, b, c, d, e, sim, (args.label, args.label_alt))
    summary = {"simulated": sim, "n_signals": a["n"], "replication": a["table"].to_dict(orient="records"),
               "mclean_pontiff": b["mp"], "slope_post_on_in": b["slope_post_on_in"],
               "shrinkage": {k: d[k] for k in ("mu0", "tau", "share_p_gt_95", "avg_shrink", "rmse_raw", "rmse_shrunk")},
               "simulation": e.to_dict(orient="records"),
               "methodology": c.to_dict(orient="records") if c is not None else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a["table"].to_string(index=False))
    print(f"Post-sample {b['mp']['coef']['post_sample']:+.0%}, post-publication {b['mp']['coef']['post_pub']:+.0%}")
    print(f"Shrinkage: {d['share_p_gt_95']:.0%} with P(mean>0) > 95%")
    print(f"Report: {out / 'lab16_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
