"""Lab 24: From Claim to Study - Pre-registration and Power.

Chapter 24 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

WHAT THE LAB DOES
  Part A  Power for return claims: years of data needed to detect a Sharpe
          ratio at t = 2 and t = 3, analytically and by simulation with fat
          tails and autocorrelation.
  Part B  Power for predictability claims and the cost of multiple testing:
          observations needed to detect a given R-squared; the t-statistic
          needed when many hypotheses are tested.
  Part C  Forking paths: a specification curve for the size effect over 36
          reasonable research choices, and how often a pure-noise signal
          looks significant somewhere in the same grid.
  Part D  Bayesianized p-values: how likely the null still is after a given
          t-statistic, for different prior odds (Harvey 2017).
  Part E  A pre-registration document for a claim of your choice, with the
          power analysis filled in and a SHA-256 hash to freeze it.

DATA (free)
  Kenneth R. French Data Library: Portfolios Formed on Size (Part C).
  Everything else is computed or simulated.

USAGE
  python lab24.py
  python lab24.py --claim-file my_claim.json      # your own pre-registration fields
  python lab24.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, find_table, load_french_tables
from fintest.design import (min_bayes_factor, months_for_r2, multiple_testing_threshold, noise_grid_experiment,
                            posterior_null, power_for_sharpe, preregistration, simulated_power, specification_grid,
                            years_for_sharpe)

ROOT = Path(__file__).resolve().parent
SHARPES = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0)

EXAMPLE_CLAIM = {
    "title": "Small stocks earn higher returns than large stocks (the size effect)",
    "claim": "US stocks in the smallest size decile earn higher average returns than stocks in the largest decile.",
    "hypotheses": "H0: the mean monthly return of the smallest-minus-largest decile portfolio (value-weighted, "
                  "NYSE breakpoints) is zero or negative. H1: it is positive.",
    "data": "Kenneth R. French Data Library, Portfolios Formed on Size, monthly value-weighted returns, downloaded "
            "on the date above; Fama/French 3 Factors for the market factor.",
    "sample": "July 1926 to the latest month available; additionally the post-publication sample from January 1982 "
              "(Banz 1981). No observations dropped.",
    "primary": "One-sided t-test of the mean monthly decile 1 minus decile 10 return, Newey-West standard errors "
               "with 6 lags; significance at t > 1.96 for the full sample. Primary sample: post-1982.",
    "secondary": "CAPM alpha of the same portfolio; quintile and tercile versions; equal weighting; excluding "
                 "January. All 36 combinations of cut, weighting, start year and January treatment are reported "
                 "as a specification curve.",
    "power": "{power}",
    "decision": "Support: post-1982 t > 1.96 and CAPM alpha t > 1.96. Failure: post-1982 point estimate at or below "
                "zero. Inconclusive: anything else, reported as such.",
    "reporting": "All 36 specifications, the full-sample and post-1982 results, and the power analysis, whatever "
                 "the outcome.",
}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part_a(sim_check: bool = True) -> pd.DataFrame:
    rows = []
    for s in SHARPES:
        r = {"sharpe": s, "years_t2": years_for_sharpe(s, 1.96), "years_t3": years_for_sharpe(s, 3.0),
             "power_20y_t2": power_for_sharpe(s, 20, 1.96), "power_50y_t2": power_for_sharpe(s, 50, 1.96)}
        if sim_check:
            y = r["years_t2"]
            r["sim_power_at_years_t2_normal"] = simulated_power(s, y, df=np.inf, n_sims=2000)
            r["sim_power_fat_tails_autocorr"] = simulated_power(s, y, df=4, rho=0.15, n_sims=2000)
        rows.append(r)
    return pd.DataFrame(rows)


def part_b() -> dict:
    r2 = pd.DataFrame([{"r2": x, "months_t2": months_for_r2(x, 1.96), "months_t3": months_for_r2(x, 3.0)}
                       for x in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)])
    mt = pd.DataFrame([{"tests": n, "t_bonferroni": multiple_testing_threshold(n)} for n in (1, 5, 10, 50, 100, 300, 1000)])
    return {"r2": r2, "multiple": mt}


def size_spec_function(vw: pd.DataFrame, ew: pd.DataFrame, mkt: pd.Series | None):
    cuts = {"tercile": ("Lo 30", "Hi 30"), "quintile": ("Lo 20", "Hi 20"), "decile": ("Lo 10", "Hi 10")}

    def f(cut, weighting, start, january):
        t = vw if weighting == "value" else ew
        lo, hi = cuts[cut]
        x = (t[lo] - t[hi]).loc[pd.Period(start, "M"):].dropna()
        if january == "excluded":
            x = x[x.index.month != 1]
        n = len(x)
        e = x - x.mean()
        s = e @ e / n
        for L in range(1, 7):
            s += 2 * (1 - L / 7) * (e.iloc[L:].to_numpy() @ e.iloc[:-L].to_numpy()) / n
        return {"estimate": float(x.mean() * 12), "t": float(x.mean() / math.sqrt(s / n)), "months": n}

    return f


def part_c(args) -> dict:
    grid = {"cut": ["tercile", "quintile", "decile"], "weighting": ["value", "equal"],
            "start": ["1926-07", "1963-07", "1982-01"], "january": ["included", "excluded"]}
    if args.source == "sim":
        rng = np.random.default_rng(args.sim_seed)
        idx = pd.period_range("1926-07", periods=1188, freq="M")
        base = 0.004 * (idx.month == 1) + 0.0005
        cols = ["Lo 30", "Hi 30", "Lo 20", "Hi 20", "Lo 10", "Hi 10"]
        mk = 0.006 + 0.05 * rng.standard_normal(1188)
        tilt = {"Lo 30": 1.0, "Lo 20": 1.4, "Lo 10": 2.0, "Hi 30": 0, "Hi 20": 0, "Hi 10": 0}
        vw = pd.DataFrame({c: mk + tilt[c] * base + (0.02 + 0.01 * tilt[c]) * rng.standard_normal(1188) for c in cols},
                          index=idx)
        ew = vw + pd.DataFrame({c: tilt[c] * 0.001 + 0.005 * rng.standard_normal(1188) for c in cols}, index=idx)
        spec = specification_grid(size_spec_function(vw, ew, None), grid)
    else:
        try:
            t = load_french_tables("size_portfolios", path=args.size_file)
            vw = find_table(t, "value", "weight")
            ew = find_table(t, "equal", "weight")
            vw.columns = [c.strip() for c in vw.columns]
            ew.columns = [c.strip() for c in ew.columns]
            spec = specification_grid(size_spec_function(vw, ew, None), grid)
        except (DataUnavailable, KeyError) as e:
            print(f"Size portfolios not loaded ({str(e).splitlines()[0]}); specification curve skipped")
            spec = None
    noise = noise_grid_experiment(n_trials=args.noise_trials, seed=args.sim_seed)
    return {"spec": spec, "noise": noise}


def part_d() -> pd.DataFrame:
    rows = []
    for t in (1.96, 2.5, 3.0, 3.5, 4.0):
        r = {"t": t, "p_two_sided": 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2)))), "min_bayes_factor": min_bayes_factor(t)}
        for odds, lab in ((1, "even_odds"), (4, "odds_4_to_1_against"), (9, "odds_9_to_1_against"),
                          (99, "odds_99_to_1_against")):
            r[f"p_null_{lab}"] = posterior_null(t, odds)
        rows.append(r)
    return pd.DataFrame(rows)


def part_e(args, a: pd.DataFrame, spec) -> str:
    fields = dict(EXAMPLE_CLAIM)
    if args.claim_file:
        fields.update(json.loads(Path(args.claim_file).read_text()))
    s = args.expected_sharpe
    power = (f"Expected annualized Sharpe ratio of the effect: {s:.2f} (set before seeing the data). Years needed for "
             f"80% power at t = 1.96: {years_for_sharpe(s):.0f}; at t = 3.0: {years_for_sharpe(s, 3.0):.0f}. "
             f"Power with 44 years (1982 onward) at t = 1.96: {power_for_sharpe(s, 44):.0%}.")
    if "{power}" in fields.get("power", "{power}"):
        fields["power"] = power
    return preregistration(fields)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_power(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    s = np.linspace(0.15, 1.2, 100)
    ax.plot(s, [years_for_sharpe(x, 1.96) for x in s], color=plots.BLUE, linewidth=1.8, label="t = 1.96, 80% power")
    ax.plot(s, [years_for_sharpe(x, 3.0) for x in s], color=plots.RED, linewidth=1.8, label="t = 3.0, 80% power")
    for y, lab in ((100, "US stock data since 1926"), (40, "Typical post-publication sample")):
        ax.axhline(y, color=plots.INK_3, linewidth=0.8, linestyle="--")
        ax.text(1.18, y * 1.08, lab, ha="right", fontsize=8, color=plots.INK_2)
    ax.set_yscale("log")
    ax.set_xlabel("True annualized Sharpe ratio of the effect")
    ax.set_ylabel("Years of monthly data needed (log scale)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    plots.titles(fig, "Lab Figure 24.1  How long a sample does a claim need?",
                 "Normal returns, no autocorrelation; the table checks the effect of fat tails and autocorrelation by simulation")
    plots.finish(fig, out / "fig24_1_power.png")
    plt.close(fig)


def fig_multiple(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.2)
    n = np.unique(np.round(np.logspace(0, 3.5, 60)).astype(int))
    ax.plot(n, [multiple_testing_threshold(int(k)) for k in n], color=plots.VIOLET, linewidth=1.8)
    ax.axhline(1.96, color=plots.INK_3, linewidth=0.8, linestyle="--")
    ax.axhline(3.0, color=plots.RED, linewidth=0.8, linestyle="--")
    ax.text(1.1, 2.03, "t = 1.96", fontsize=8, color=plots.INK_2)
    ax.text(1.1, 3.07, "t = 3.0 (Harvey, Liu and Zhu)", fontsize=8, color=plots.RED)
    ax.set_xscale("log")
    ax.set_xlabel("Number of hypotheses tested (log scale)")
    ax.set_ylabel("t-statistic needed (Bonferroni, 5%)")
    plots.titles(fig, "Lab Figure 24.2  The price of testing many ideas",
                 "Family-wise 5% significance, two-sided, independent tests")
    plots.finish(fig, out / "fig24_2_multiple.png")
    plt.close(fig)


def fig_spec(spec, out, sim):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    fig.patch.set_facecolor(plots.SURFACE)
    x = np.arange(len(spec))
    sig = spec["t"].abs() > 1.96
    plots.style_axes(ax1)
    ax1.scatter(x[sig], 100 * spec["estimate"][sig], color=plots.BLUE, s=22, label="|t| > 1.96", zorder=3)
    ax1.scatter(x[~sig], 100 * spec["estimate"][~sig], color=plots.INK_3, s=22, label="not significant", zorder=3)
    ax1.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax1.set_ylabel("Small minus big, % a year")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    plots.style_axes(ax2)
    rows = [("cut", ["tercile", "quintile", "decile"]), ("weighting", ["value", "equal"]),
            ("start", ["1926-07", "1963-07", "1982-01"]), ("january", ["excluded"])]
    labels, yy = [], 0
    for col, opts in rows:
        for o in opts:
            m = (spec[col] == o).to_numpy()
            ax2.scatter(x[m], np.full(m.sum(), yy), marker="|", s=60, color=plots.INK)
            labels.append(f"{col}: {o}")
            yy += 1
    ax2.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Specification, sorted by estimate")
    ax2.grid(False)
    plots.titles(fig, "Lab Figure 24.3  One claim, 36 reasonable ways to test it",
                 _sub(sim, "Size effect (smallest minus largest size portfolio) under every combination of research choices"))
    plots.finish(fig, out / "fig24_3_spec_curve.png")
    plt.close(fig)


def fig_noise(c, out):
    fig, ax, plt = plots.new_figure(9.0, 4.2)
    n = c["noise"]
    ax.hist(n["max_abs_t"], bins=30, color=plots.INK_3, edgecolor=plots.SURFACE)
    ax.axvline(1.96, color=plots.RED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Largest |t| among 36 specifications of a pure-noise signal")
    ax.set_ylabel("Number of simulated studies")
    plots.titles(fig, "Lab Figure 24.4  The garden of forking paths",
                 f"Pre-chosen specification significant in {n['pre_specified_rate']:.0%} of studies; at least one of 36 "
                 f"significant in {n['any_spec_rate']:.0%} · simulated noise by design")
    plots.finish(fig, out / "fig24_4_noise.png")
    plt.close(fig)


def fig_bayes(d, out):
    fig, ax, plt = plots.new_figure(9.0, 4.2)
    cols = [c for c in d.columns if c.startswith("p_null_")]
    shades = [plots.GREEN, plots.BLUE, plots.VIOLET, plots.RED]
    for c, col in zip(cols, shades):
        ax.plot(d["t"], d[c], marker="o", color=col, linewidth=1.6,
                label=c.replace("p_null_", "").replace("_", " ").replace("odds", "prior odds"))
    plots.percent_axis(ax)
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Lowest probability the null is true")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    plots.titles(fig, "Lab Figure 24.5  What a t-statistic of 2 is worth",
                 "Minimum Bayes factor exp(−t²/2) combined with prior odds against a real effect (Harvey 2017)")
    plots.finish(fig, out / "fig24_5_bayes.png")
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


def write_report(out, a, b, c, d, prereg, sim):
    md = ["# Lab 24 results: From Claim to Study", ""]
    if sim:
        md += ["> **Simulated data** for the specification curve. These results test the code only.", ""]
    md += ["## Part A: Years of data needed to detect a Sharpe ratio", "",
           _md(a, {"sharpe": _num(".2f"), "years_t2": _num(".0f"), "years_t3": _num(".0f"), "power_20y_t2": _num(".0%"),
                   "power_50y_t2": _num(".0%"), "sim_power_at_years_t2_normal": _num(".0%"),
                   "sim_power_fat_tails_autocorr": _num(".0%")}), "",
           "The simulated columns re-check the analytic answer at the years it gives for t = 1.96: about 80% with normal "
           "returns. With fat tails and autocorrelation the naive t-test's rejection rate changes, which is why backtests "
           "need robust standard errors.", "", "![Lab Figure 24.1](fig24_1_power.png)", "",
           "## Part B: Predictability and multiple testing", "",
           _md(b["r2"], {"r2": _num(".1%"), "months_t2": _num(",.0f"), "months_t3": _num(",.0f")}), "",
           _md(b["multiple"], {"t_bonferroni": _num(".2f")}), "", "![Lab Figure 24.2](fig24_2_multiple.png)", "",
           "## Part C: Forking paths", ""]
    if c["spec"] is not None:
        sp = c["spec"]
        md += [f"Size effect across {len(sp)} specifications: estimates from {sp['estimate'].min():+.1%} to "
               f"{sp['estimate'].max():+.1%} a year; {int((sp['t'].abs() > 1.96).sum())} of {len(sp)} significant at "
               "|t| > 1.96.", "", _md(sp, {"estimate": _num("+.2%"), "t": _num("+.2f")}), "",
               "![Lab Figure 24.3](fig24_3_spec_curve.png)", ""]
    n = c["noise"]
    md += [f"Pure-noise signal, {n['n_trials']} simulated studies: the pre-chosen specification was significant in "
           f"{n['pre_specified_rate']:.1%} of studies (about 5% expected); at least one of {n['n_specs']} specifications "
           f"was significant in {n['any_spec_rate']:.1%}.", "", "![Lab Figure 24.4](fig24_4_noise.png)", "",
           "## Part D: Bayesianized p-values", "",
           _md(d, {k: _num(".3f") for k in d.columns if k != "t"} | {"t": _num(".2f")}), "",
           "![Lab Figure 24.5](fig24_5_bayes.png)", "", "## Part E: Pre-registration", "",
           "Written to `preregistration.md`. Publish it (or its hash) somewhere time-stamped before running the primary "
           "test.", "", "```", prereg, "```", ""]
    (out / "lab24_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    ap.add_argument("--size-file", help="Portfolios_Formed_on_ME_CSV.zip")
    ap.add_argument("--claim-file", help="JSON with pre-registration fields to override the example")
    ap.add_argument("--expected-sharpe", type=float, default=0.3, help="expected Sharpe ratio for the power section")
    ap.add_argument("--noise-trials", type=int, default=300)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = args.source == "sim"
    a = part_a()
    b = part_b()
    c = part_c(args)
    d = part_d()
    prereg = part_e(args, a, c["spec"])
    (out / "preregistration.md").write_text(prereg)
    fig_power(a, out, sim)
    fig_multiple(b, out, sim)
    if c["spec"] is not None:
        fig_spec(c["spec"], out, sim)
        c["spec"].to_csv(out / "table_spec_curve.csv", index=False)
    fig_noise(c, out)
    fig_bayes(d, out)
    a.to_csv(out / "table_power.csv", index=False)
    d.to_csv(out / "table_bayes.csv", index=False)
    write_report(out, a, b, c, d, prereg, sim)
    summary = {"simulated": sim, "power": a.to_dict(orient="records"),
               "noise": {k: v for k, v in c["noise"].items() if k != "max_abs_t"},
               "spec_significant": None if c["spec"] is None else int((c["spec"]["t"].abs() > 1.96).sum()),
               "spec_n": None if c["spec"] is None else len(c["spec"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a[["sharpe", "years_t2", "years_t3"]].round(1).to_string(index=False))
    print(f"Noise: pre-specified {c['noise']['pre_specified_rate']:.1%}, any of 36 {c['noise']['any_spec_rate']:.1%}")
    print(f"Report: {out / 'lab24_report.md'}")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
