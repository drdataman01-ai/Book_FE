"""Lab 3: Do Size and Value Explain Returns That Beta Cannot?

Chapter 3 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  How much additional factors add to the CAPM in explaining differences in
  portfolio returns, and how testing many candidate factors can produce
  false discoveries.

WHAT THE LAB DOES
  Part A  The regressions of Fama and French (1993). The CAPM and the three-factor
          model on the 25 portfolios formed on size and book-to-market: alphas,
          loadings, R-squared and the GRS test.
  Part B  A multiple-testing experiment after Harvey, Liu and Zhu (2016).
          1,000 simulated factors with a true premium of zero, tested at
          |t| > 1.96 and |t| > 3.0, repeated many times; independent and
          correlated factors.
  Part C  Statistical and economic significance. Annualized premium, Sharpe
          ratio and t-statistic of Mkt-RF, SMB and HML, by subperiod, and the
          one-way trading cost that would wipe out each premium.

DATA
  Kenneth R. French Data Library: Fama/French 3 Factors (monthly) and
  25 Portfolios Formed on Size and Book-to-Market (5 x 5). Both free.

USAGE
  python lab03.py
  python lab03.py --source sim            # fake data, to test the code

OUTPUT
  report/lab03_report.md, five figures, CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import (
    DataUnavailable,
    find_table,
    load_french_factors,
    load_french_tables,
    simulated_25_portfolios,
    simulated_ff3_monthly,
)
from fintest.inference import grs_test, ols_all_se

ROOT = Path(__file__).resolve().parent
FACTORS = ["Mkt-RF", "SMB", "HML"]
MODELS = {"CAPM": ["Mkt-RF"], "Three-factor": FACTORS}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def label_25(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the 25 portfolio columns S1B1 ... S5B5 (size quintile, book-to-market quintile).

    The French file lists portfolios size-major (SMALL LoBM ... BIG HiBM), so the
    columns are taken by position and the code does not depend on exact names.
    """
    out = df.iloc[:, :25].copy()
    out.columns = [f"S{i // 5 + 1}B{i % 5 + 1}" for i in range(25)]
    return out


def load(source: str, factors_file=None, portfolios_file=None, sim_seed: int = 0) -> dict:
    if source == "sim":
        ff3 = simulated_ff3_monthly(seed=sim_seed)
        p25 = simulated_25_portfolios(ff3, seed=sim_seed)
        return {"ff3": ff3, "p25": label_25(p25), "simulated": True}
    ff3 = load_french_factors("ff3_monthly", path=factors_file)
    p25 = find_table(load_french_tables("size_bm_25", path=portfolios_file), "value", "weight")
    return {"ff3": ff3, "p25": label_25(p25), "simulated": False}


def panel(data: dict, start: str | None = "1963-07", end: str | None = None):
    """Portfolio excess returns Y and factors F over a common sample (missing values, -99.99, dropped)."""
    ff3, p25 = data["ff3"], data["p25"]
    idx = ff3.index.intersection(p25.index)
    if start:
        idx = idx[idx >= pd.Period(start, freq="M")]
    if end:
        idx = idx[idx <= pd.Period(end, freq="M")]
    p25 = p25.loc[idx].where(p25.loc[idx] > -0.99)
    keep = p25.notna().all(axis=1)
    Y = p25.loc[keep].sub(ff3.loc[idx[keep], "RF"], axis=0)
    F = ff3.loc[idx[keep], FACTORS]
    return Y, F


# ---------------------------------------------------------------------------
# Part A: CAPM and the three-factor model on the 25 portfolios
# ---------------------------------------------------------------------------


def part_a(Y: pd.DataFrame, F: pd.DataFrame, nw_lags: int | None = None) -> dict:
    out = {}
    y = Y.to_numpy()
    for model, cols in MODELS.items():
        f = F[cols].to_numpy()
        reg = ols_all_se(y, f, nw_lags)
        coef, resid = reg["coef"], reg["resid"]
        r2 = 1 - resid.var(axis=0) / y.var(axis=0)
        table = pd.DataFrame({"portfolio": Y.columns, "mean_excess_annual": y.mean(axis=0) * 12,
                              "alpha_annual": coef[0] * 12, "t_alpha": coef[0] / reg["se_nw"][0], "r2": r2})
        for k, c in enumerate(cols, start=1):
            table[f"beta_{c}"] = coef[k]
        # The model's prediction of each portfolio's average excess return: betas times factor means.
        table["predicted_annual"] = (coef[1:].T @ F[cols].mean().to_numpy()) * 12
        out[model] = {"table": table, "grs": grs_test(y, f), "nw_lags": reg["nw_lags"],
                      "summary": {"mean_r2": float(r2.mean()), "median_r2": float(np.median(r2)),
                                  "share_r2_above_90": float((r2 > 0.9).mean()),
                                  "mean_abs_alpha_annual": float(np.abs(coef[0]).mean() * 12),
                                  "significant_alphas": int((np.abs(table["t_alpha"]) > 1.96).sum())}}
    out["sample"] = {"start": str(Y.index[0]), "end": str(Y.index[-1]), "months": len(Y),
                     "n_portfolios": Y.shape[1]}
    return out


# ---------------------------------------------------------------------------
# Part B: 1,000 factors with a true premium of zero
# ---------------------------------------------------------------------------


def simulate_null_factors(n_factors: int, n_months: int, vol: float, rng: np.random.Generator,
                          corr: float = 0.0) -> np.ndarray:
    """Monthly returns of factors whose true mean is zero. With corr > 0 they share a common shock."""
    z = rng.standard_normal((n_months, n_factors))
    if corr > 0:
        common = rng.standard_normal((n_months, 1))
        z = math.sqrt(corr) * common + math.sqrt(1 - corr) * z
    return vol * z


def mean_tstats(r: np.ndarray) -> np.ndarray:
    T = r.shape[0]
    return r.mean(axis=0) / (r.std(axis=0, ddof=1) / math.sqrt(T))


def part_b(n_months: int, n_factors: int = 1000, n_reps: int = 200, vol: float = 0.03, corr: float = 0.5,
           thresholds=(1.96, 3.0), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows, counts = [], {}
    example_t = None
    for label, rho in (("Independent factors", 0.0), (f"Correlated factors (rho = {corr:.1f})", corr)):
        c = np.zeros((n_reps, len(thresholds)), dtype=int)
        for i in range(n_reps):
            t = mean_tstats(simulate_null_factors(n_factors, n_months, vol, rng, rho))
            if example_t is None:
                example_t = t
            c[i] = [(np.abs(t) > h).sum() for h in thresholds]
        counts[label] = c
        for j, h in enumerate(thresholds):
            rows.append({"design": label, "threshold": h, "mean_false_discoveries": c[:, j].mean(),
                         "p5": np.percentile(c[:, j], 5), "p95": np.percentile(c[:, j], 95),
                         "max": c[:, j].max(), "expected_if_normal": n_factors * 2 * _norm_sf(h)})
    return {"table": pd.DataFrame(rows), "counts": counts, "example_t": example_t, "n_factors": n_factors,
            "n_months": n_months, "n_reps": n_reps, "thresholds": thresholds}


def _norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


# ---------------------------------------------------------------------------
# Part C: statistical and economic significance of the factor premiums
# ---------------------------------------------------------------------------


def factor_stats(f: pd.Series) -> dict:
    f = f.dropna()
    T = len(f)
    m, s = f.mean(), f.std(ddof=1)
    return {"months": T, "premium_annual": 12 * m, "sharpe_annual": math.sqrt(12) * m / s,
            "t": m / (s / math.sqrt(T)), "vol_annual": math.sqrt(12) * s}


def part_c(F: pd.DataFrame, turnover: float = 0.03, splits=("1963-07", "1991-01", "2010-01")) -> dict:
    """Premiums by subperiod, and the break-even one-way cost.

    A long-short factor that replaces a share `turnover` of each leg every month
    trades 2 x turnover of its gross position a month on each side; the
    break-even one-way cost is the annual premium / (2 x 2 x 12 x turnover).
    The market factor is a buy-and-hold position, so it has no break-even cost.
    """
    rows = []
    edges = [pd.Period(s, "M") for s in splits] + [F.index[-1] + 1]
    periods = [("Full sample", F.index[0], F.index[-1])] + [
        (f"{a} to {b - 1}", a, b - 1) for a, b in zip(edges[:-1], edges[1:]) if a <= F.index[-1]]
    for name, a, b in periods:
        sub = F.loc[(F.index >= a) & (F.index <= b)]
        for c in FACTORS:
            r = factor_stats(sub[c])
            r.update(period=name, factor=c)
            rows.append(r)
    t = pd.DataFrame(rows)
    full = t[t.period == "Full sample"].copy()
    full["assumed_turnover"] = np.where(full.factor == "Mkt-RF", 0.0, turnover)
    full["break_even_one_way_cost"] = np.where(full.factor == "Mkt-RF", np.nan,
                                               full.premium_annual / (4 * 12 * turnover))
    rolling = F.rolling(120).mean() * 12
    return {"table": t, "break_even": full, "rolling": rolling, "turnover": turnover}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim: bool, text: str) -> str:
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_alpha_grid(a: dict, out: Path, sim: bool):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    fig.patch.set_facecolor(plots.SURFACE)
    cmap = LinearSegmentedColormap.from_list("div", [plots.BLUE, "#f0efec", plots.RED])
    vmax = max(abs(a[m]["table"]["alpha_annual"]).max() for m in MODELS) * 100
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for ax, model in zip(axes, MODELS):
        t = a[model]["table"]
        grid = t["alpha_annual"].to_numpy().reshape(5, 5) * 100
        tgrid = t["t_alpha"].to_numpy().reshape(5, 5)
        ax.imshow(grid, cmap=cmap, norm=norm)
        for i in range(5):
            for j in range(5):
                ax.text(j, i - 0.12, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=9.5,
                        color=plots.INK, fontweight="bold" if abs(tgrid[i, j]) > 1.96 else "normal")
                ax.text(j, i + 0.22, f"t {tgrid[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                        color=plots.INK_2)
        ax.set_xticks(range(5), ["Growth", "2", "3", "4", "Value"], fontsize=9, color=plots.INK_2)
        ax.set_yticks(range(5), ["Small", "2", "3", "4", "Big"], fontsize=9, color=plots.INK_2)
        ax.set_xlabel("Book-to-market quintile", color=plots.INK_2, fontsize=9)
        ax.set_title(f"{model}  ·  GRS p = {a[model]['grs']['p_value']:.3f}", loc="left", fontsize=10,
                     color=plots.INK, pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Size quintile", color=plots.INK_2, fontsize=9)
    s = a["sample"]
    plots.titles(fig, "Lab Figure 3.1  Alphas of the 25 size and book-to-market portfolios (% per year)",
                 _sub(sim, f"{s['start']} to {s['end']} · red: positive, blue: negative · bold: |t| > 1.96 with "
                           "Newey–West errors"))
    plots.finish(fig, out / "fig3_1_alpha_grid.png")
    plt.close(fig)


def fig_r2(a: dict, out: Path, sim: bool):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    x = np.arange(25)
    w = 0.4
    ax.bar(x - w / 2, a["CAPM"]["table"]["r2"], w, color=plots.INK_3, label="CAPM")
    ax.bar(x + w / 2, a["Three-factor"]["table"]["r2"], w, color=plots.BLUE, label="Three-factor")
    ax.axhline(0.9, color=plots.RED, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, a["CAPM"]["table"]["portfolio"], rotation=90, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("R²")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    plots.titles(fig, "Lab Figure 3.2  Adding SMB and HML raises the R² of every portfolio",
                 _sub(sim, f"Time-series R² for each portfolio · mean {a['CAPM']['summary']['mean_r2']:.2f} (CAPM) "
                           f"against {a['Three-factor']['summary']['mean_r2']:.2f} (three-factor) · dashed line: 0.90"))
    plots.finish(fig, out / "fig3_2_r2.png")
    plt.close(fig)


def fig_predicted(a: dict, out: Path, sim: bool):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    fig.patch.set_facecolor(plots.SURFACE)
    allv = np.concatenate([a[m]["table"][c].to_numpy() for m in MODELS for c in ("predicted_annual",
                                                                                 "mean_excess_annual")])
    lo, hi = allv.min() - 0.01, allv.max() + 0.01
    for ax, model in zip(axes, MODELS):
        plots.style_axes(ax)
        t = a[model]["table"]
        ax.plot([lo, hi], [lo, hi], color=plots.INK_3, linewidth=0.8)
        ax.scatter(t["predicted_annual"], t["mean_excess_annual"], color=plots.BLUE, s=22, zorder=3)
        for _, r in t.iterrows():
            if r.portfolio in ("S1B1", "S1B5", "S5B1", "S5B5"):
                ax.annotate(r.portfolio, (r.predicted_annual, r.mean_excess_annual), fontsize=7.5,
                            color=plots.INK_2, textcoords="offset points", xytext=(4, 3))
        ax.set_title(model, loc="left", fontsize=10, color=plots.INK)
        ax.set_xlabel("Predicted average excess return (per year)")
        plots.percent_axis(ax, "x")
        plots.percent_axis(ax, "y")
    axes[0].set_ylabel("Actual average excess return (per year)")
    plots.titles(fig, "Lab Figure 3.3  Actual against predicted average returns",
                 _sub(sim, "On the 45-degree line the model explains the portfolio's average return; the vertical "
                           "distance is the alpha"))
    plots.finish(fig, out / "fig3_3_predicted.png")
    plt.close(fig)


def fig_false_discoveries(b: dict, out: Path, sim: bool):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.patch.set_facecolor(plots.SURFACE)
    ax = axes[0]
    plots.style_axes(ax)
    t = b["example_t"]
    ax.hist(t, bins=50, color=plots.INK_3, edgecolor=plots.SURFACE)
    lines = []
    for h, col in zip(b["thresholds"], (plots.ORANGE, plots.RED)):
        ax.axvline(h, color=col, linewidth=1)
        ax.axvline(-h, color=col, linewidth=1)
        lines.append(f"|t| > {h:g}: {(np.abs(t) > h).sum()} significant")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8.5, color=plots.INK)
    ax.set_xlabel("t-statistic of the average premium")
    ax.set_title(f"One run: {b['n_factors']:,} factors, true premium zero", loc="left", fontsize=10, color=plots.INK)
    ax = axes[1]
    plots.style_axes(ax)
    for (label, c), col in zip(b["counts"].items(), (plots.BLUE, plots.MAGENTA)):
        ax.hist(c[:, 0], bins=30, color=col, alpha=0.6, label=label)
    ax.axvline(b["n_factors"] * 2 * _norm_sf(1.96), color=plots.INK, linewidth=0.8, linestyle="--")
    ax.set_xlabel("False discoveries at |t| > 1.96 per run")
    ax.set_title(f"{b['n_reps']} runs", loc="left", fontsize=10, color=plots.INK)
    ax.legend(frameon=False, fontsize=8.5)
    plots.titles(fig, "Lab Figure 3.4  Testing many useless factors produces many 'discoveries'",
                 f"{b['n_months']} months of simulated returns per factor · dashed line: 5% of the factors")
    plots.finish(fig, out / "fig3_4_false_discoveries.png")
    plt.close(fig)


def fig_premiums(c: dict, out: Path, sim: bool):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    r = c["rolling"].dropna()
    dates = r.index.to_timestamp()
    for col, colr in zip(FACTORS, (plots.INK, plots.BLUE, plots.ORANGE)):
        ax.plot(dates, r[col], color=colr, linewidth=1.5, label=col)
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    plots.percent_axis(ax)
    ax.set_ylabel("Average return over the past 10 years (per year)")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    plots.titles(fig, "Lab Figure 3.5  The factor premiums are not constant",
                 _sub(sim, "Rolling 120-month average of each factor's monthly return, annualized"))
    plots.finish(fig, out / "fig3_5_premiums.png")
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


def write_report(out: Path, a, b, c, sim: bool):
    s = a["sample"]
    md = ["# Lab 3 results: Do Size and Value Explain Returns That Beta Cannot?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** how much additional factors add to the CAPM in explaining differences in portfolio returns, and how testing many candidate factors can produce false discoveries.", "",
           f"Sample for Parts A and C: {s['start']} to {s['end']} ({s['months']} months), {s['n_portfolios']} "
           "value-weighted portfolios.", "", "## Part A: The regressions of Fama and French (1993)", ""]
    rows = []
    for m in MODELS:
        sm, g = a[m]["summary"], a[m]["grs"]
        rows.append({"Model": m, "Mean R²": sm["mean_r2"], "Share R² > 0.90": sm["share_r2_above_90"],
                     "Mean |alpha| (annual)": sm["mean_abs_alpha_annual"], "Alphas with |t| > 1.96": sm["significant_alphas"],
                     "GRS": g["grs"], "GRS p": g["p_value"]})
    md += [_md(pd.DataFrame(rows), {"Mean R²": _num(".2f"), "Share R² > 0.90": _num(".0%"),
                                     "Mean |alpha| (annual)": _num(".2%"), "GRS": _num(".2f"), "GRS p": _num(".4f")}), "",
           "A higher R² means the model explains more of the month-to-month variation. The alphas and the GRS test ask "
           "the harder question: does it explain the differences in average returns?", "",
           "**Three-factor loadings**", "",
           _md(a["Three-factor"]["table"][["portfolio", "alpha_annual", "t_alpha", "beta_Mkt-RF", "beta_SMB", "beta_HML",
                                          "r2"]],
               {"alpha_annual": _num("+.2%"), "t_alpha": _num("+.2f"), "beta_Mkt-RF": _num(".2f"),
                "beta_SMB": _num("+.2f"), "beta_HML": _num("+.2f"), "r2": _num(".2f")}), "",
           "![Lab Figure 3.1](fig3_1_alpha_grid.png)", "", "![Lab Figure 3.2](fig3_2_r2.png)", "",
           "![Lab Figure 3.3](fig3_3_predicted.png)", ""]
    md += ["## Part B: A multiple-testing experiment", "",
           f"{b['n_factors']:,} factors with a true premium of zero, {b['n_months']} months each, {b['n_reps']} runs. "
           "Every rejection is a false discovery.", "",
           _md(b["table"], {"mean_false_discoveries": _num(".1f"), "p5": _num(".0f"), "p95": _num(".0f"),
                            "expected_if_normal": _num(".1f")}), "",
           "Correlated factors leave the average count unchanged but make it far more variable: in some runs many "
           "factors look significant together. This is a demonstration of the problem that motivates Harvey, Liu and "
           "Zhu's higher hurdle, not a replication of their study.", "",
           "![Lab Figure 3.4](fig3_4_false_discoveries.png)", ""]
    md += ["## Part C: Statistical and economic significance", "",
           _md(c["table"][["period", "factor", "premium_annual", "sharpe_annual", "t", "vol_annual", "months"]],
               {"premium_annual": _num("+.2%"), "sharpe_annual": _num(".2f"), "t": _num(".2f"),
                "vol_annual": _num(".1%")}), "",
           f"**Break-even one-way cost**, assuming each long–short factor replaces {c['turnover']:.0%} of each leg a "
           "month (an assumption; change it with `--turnover`):", "",
           _md(c["break_even"][["factor", "premium_annual", "assumed_turnover", "break_even_one_way_cost"]],
               {"premium_annual": _num("+.2%"), "assumed_turnover": _num(".0%"),
                "break_even_one_way_cost": _num(".2%")}), "",
           "![Lab Figure 3.5](fig3_5_premiums.png)", ""]
    md += ["## Questions to think about", "",
           "1. Do SMB and HML explain the returns of the 25 portfolios better than the market alone? Look at the alphas "
           "and the GRS test, not only the R².",
           "2. How many of 1,000 useless factors cleared |t| > 1.96, and how many cleared 3.0?",
           "3. Are the size and value premiums large and stable enough to matter after costs?", ""]
    (out / "lab03_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--factors-file", default=None, help="Fama/French 3 Factors (monthly) zip or CSV")
    ap.add_argument("--portfolios-file", default=None,
                    help="25 Portfolios Formed on Size and Book-to-Market (5 x 5) zip or CSV")
    ap.add_argument("--start", default="1963-07", help="first month for Parts A and C (default 1963-07)")
    ap.add_argument("--end", default=None, help="last month (default: latest)")
    ap.add_argument("--nw-lags", type=int, default=None, help="Newey-West lags (default: rule of thumb)")
    ap.add_argument("--n-factors", type=int, default=1000, help="Part B: candidate factors per run")
    ap.add_argument("--n-reps", type=int, default=200, help="Part B: number of runs")
    ap.add_argument("--corr", type=float, default=0.5, help="Part B: correlation of the correlated factors")
    ap.add_argument("--turnover", type=float, default=0.03, help="Part C: assumed monthly turnover of each leg")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        data = load(args.source, args.factors_file, args.portfolios_file, args.sim_seed)
    except DataUnavailable as e:
        raise SystemExit(str(e)) from None
    sim = data["simulated"]
    Y, F = panel(data, args.start, args.end)
    a = part_a(Y, F, args.nw_lags)
    b = part_b(len(F), args.n_factors, args.n_reps, corr=args.corr, seed=args.seed)
    c = part_c(F, args.turnover)
    fig_alpha_grid(a, out, sim)
    fig_r2(a, out, sim)
    fig_predicted(a, out, sim)
    fig_false_discoveries(b, out, sim)
    fig_premiums(c, out, sim)
    for m in MODELS:
        a[m]["table"].to_csv(out / f"table_{m.lower().replace('-', '_')}.csv", index=False)
    b["table"].to_csv(out / "table_false_discoveries.csv", index=False)
    c["table"].to_csv(out / "table_premiums.csv", index=False)
    c["break_even"].to_csv(out / "table_break_even.csv", index=False)
    write_report(out, a, b, c, sim)
    summary = {"simulated": sim, "sample": a["sample"],
               "models": {m: {"summary": a[m]["summary"], "grs": a[m]["grs"]} for m in MODELS},
               "false_discoveries": b["table"].to_dict(orient="records"),
               "premiums": c["table"].to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for m in MODELS:
        sm = a[m]["summary"]
        print(f"A  {m:12s} mean R2 {sm['mean_r2']:.2f}, mean |alpha| {sm['mean_abs_alpha_annual']:.2%}/yr, "
              f"GRS p = {a[m]['grs']['p_value']:.4f}")
    bt = b["table"]
    for _, r in bt.iterrows():
        print(f"B  {r.design:32s} |t| > {r.threshold:g}: {r.mean_false_discoveries:.1f} false discoveries on average")
    for _, r in c["break_even"].iterrows():
        print(f"C  {r.factor:7s} {r.premium_annual:+.2%}/yr, Sharpe {r.sharpe_annual:.2f}, t = {r.t:.2f}")
    print(f"Report: {out / 'lab03_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
