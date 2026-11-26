"""Lab 11: Do High-Interest Currencies Depreciate Enough to Offset Their Yield?

Chapter 11 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  Whether high-interest currencies fall enough to cancel their extra yield,
  as uncovered interest parity says they should.

WHAT THE LAB DOES
  Part A  Fama (1984) regressions of exchange-rate changes on interest
          differentials for nine currencies against the dollar.
  Part B  A G10 carry portfolio: long the three highest-rate currencies,
          short the three lowest; where the return comes from.
  Part C  Stress tests: subperiods, transaction costs, skewness and crashes,
          exposure to stocks and to the VIX.
  Part D  The investor's view: drawdowns and the worst months.

DATA (free, FRED)
  Federal Reserve H.10 exchange rates (EXJPUS, EXUSUK, ...) and OECD
  three-month interbank rates (IR3TIB01xxM156N). Optional: VIXCLS; French
  market factor for stock-market exposure.

USAGE
  python lab11.py
  python lab11.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.credit import monthly_fred
from fintest.data import DataUnavailable, load_french_factors
from fintest.fx import CURRENCIES, carry_portfolio, excess_returns, fama_regressions, load_fx, simulated_fx
from fintest.inference import moments
from fintest.stats import ols_newey_west

ROOT = Path(__file__).resolve().parent
SUBPERIODS = [("Start–1998", None, "1998-12"), ("1999–2007", "1999-01", "2007-12"), ("2008–", "2008-01", None)]


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        d = simulated_fx(seed=seed)
        d["simulated"] = True
        return d
    d = load_fx(files)
    try:
        d["vix"] = monthly_fred("VIXCLS", files.get("VIXCLS"))
    except DataUnavailable:
        d["vix"] = None
    try:
        ff = load_french_factors("ff3_monthly", path=files.get("ff3"))
        d["mkt"] = ff[[c for c in ff.columns if c.strip().lower() == "mkt-rf"][0]]
    except DataUnavailable:
        d["mkt"] = None
    d["simulated"] = False
    return d


def _stats(x: pd.Series) -> dict:
    v = x.dropna().to_numpy()
    w = np.exp(np.cumsum(v))
    mo = moments(v)
    return {"mean_annual": 12 * v.mean(), "vol_annual": v.std(ddof=1) * math.sqrt(12),
            "sharpe": v.mean() / v.std(ddof=1) * math.sqrt(12), "t": v.mean() / (v.std(ddof=1) / math.sqrt(len(v))),
            "skewness": mo["skewness"], "worst_month": float(v.min()),
            "max_drawdown": float((w / np.maximum.accumulate(w) - 1).min()), "months": len(v)}


def part_a(er: dict) -> pd.DataFrame:
    return fama_regressions(er["fx"], er["diff"])


def part_b(er: dict, cost: float = 0.0) -> dict:
    cp = carry_portfolio(er["rx"], er["diff"], cost=cost)
    rows = [{"portfolio": n, **_stats(cp[c])} for n, c in (("Carry (high minus low)", "carry"),
                                                           ("High-rate currencies (long only)", "high"),
                                                           ("Low-rate currencies (long only)", "low"),
                                                           ("Dollar (all currencies)", "dollar"))]
    comp = {"interest_annual": 12 * cp["carry_interest"].mean(), "fx_annual": 12 * cp["carry_fx"].mean()}
    return {"cp": cp, "table": pd.DataFrame(rows), "components": comp}


def part_c(er: dict, d: dict, b: dict) -> dict:
    cp = b["cp"]
    sub = []
    for label, a, e in SUBPERIODS:
        x = cp["carry"].loc[pd.Period(a, "M") if a else None:pd.Period(e, "M") if e else None]
        if len(x) >= 24:
            sub.append({"period": label, **_stats(x)})
    costs = []
    for k in (0.0, 0.0005, 0.001, 0.002):
        c2 = carry_portfolio(er["rx"], er["diff"], cost=k)
        costs.append({"cost_bp": 1e4 * k, **{kk: v for kk, v in _stats(c2["carry"]).items()
                                             if kk in ("mean_annual", "sharpe")},
                      "avg_turnover": float(c2["turnover"].mean())})
    exposures = []
    for name, x in (("Stock market (excess return)", d.get("mkt")), ("Change in VIX (points)", d.get("vix"))):
        if x is None:
            continue
        xx = x.diff() if "VIX" in name else x
        df = pd.concat([cp["carry"].rename("y"), xx.rename("x")], axis=1).dropna()
        if len(df) < 36:
            continue
        r = ols_newey_west(df["y"].to_numpy(), df["x"].to_numpy(), lags=3)
        down = df[df["x"] < 0] if "Stock" in name else df[df["x"] > 0]
        rd = np.polyfit(down["x"], down["y"], 1)[0] if len(down) > 10 else float("nan")
        exposures.append({"factor": name, "beta": float(r["coef"][1, 0]), "t": float(r["t"][1, 0]),
                          "beta_bad_months": float(rd), "corr": float(df["y"].corr(df["x"])), "months": len(df)})
    worst = cp["carry"].nsmallest(10).to_frame("carry")
    worst["high"] = cp.loc[worst.index, "high"]
    worst["low"] = cp.loc[worst.index, "low"]
    worst.index = worst.index.astype(str)
    return {"subperiods": pd.DataFrame(sub), "costs": pd.DataFrame(costs), "exposures": pd.DataFrame(exposures),
            "worst": worst}


def part_d(b: dict) -> dict:
    x = b["cp"]["carry"]
    w = np.exp(x.cumsum())
    dd = w / w.cummax() - 1
    trough = dd.idxmin()
    peak = w.loc[:trough].idxmax()
    rec = w.loc[trough:]
    back = rec[rec >= w.loc[peak]]
    k = max(1, int(round(0.05 * len(x))))
    losses = -x.clip(upper=0)
    return {"max_drawdown": float(dd.min()), "peak": str(peak), "trough": str(trough),
            "recovered": str(back.index[0]) if len(back) else "not yet",
            "share_losses_worst_5pct": float(losses.nlargest(k).sum() / losses.sum()),
            "share_months_positive": float((x > 0).mean()), "drawdown": dd}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_fama(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = a[a.currency != "Pooled (demeaned)"]
    x = np.arange(len(t))
    ax.errorbar(x, t["slope"], yerr=2 * t["se"], fmt="o", color=plots.BLUE, capsize=4, label="Slope (± 2 s.e.)")
    ax.axhline(1, color=plots.INK, linestyle="--", linewidth=1)
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    ax.text(-0.4, 1.15, "Uncovered interest parity: 1", ha="left", va="bottom", fontsize=8.5, color=plots.INK)
    pooled = a[a.currency == "Pooled (demeaned)"]
    if len(pooled):
        ax.axhline(float(pooled["slope"].iloc[0]), color=plots.ORANGE, linestyle=":", linewidth=1.5,
                   label=f"Pooled slope {float(pooled['slope'].iloc[0]):.2f}")
    ax.set_xticks(x, t["currency"])
    ax.set_ylabel("Slope coefficient")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    plots.titles(fig, "Lab Figure 11.1  Do high-rate currencies depreciate by their interest advantage?",
                 _sub(sim, "Monthly change in the exchange rate on last month's interest differential against the dollar"))
    plots.finish(fig, out / "fig11_1_fama.png")
    plt.close(fig)


def fig_cumulative(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    cp = b["cp"]
    dates = cp.index.to_timestamp()
    ax.plot(dates, cp["carry"].cumsum(), color=plots.BLUE, linewidth=1.6, label="Carry portfolio (total)")
    ax.plot(dates, cp["carry_interest"].cumsum(), color=plots.AQUA, linewidth=1.2, linestyle="--",
            label="Interest differential earned")
    ax.plot(dates, cp["carry_fx"].cumsum(), color=plots.RED, linewidth=1.2, linestyle="--",
            label="Exchange-rate change")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.set_ylabel("Cumulative log return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 11.2  Borrow low, lend high: where the return comes from",
                 _sub(sim, "Long the three highest-rate currencies, short the three lowest, rebalanced monthly"))
    plots.finish(fig, out / "fig11_2_cumulative.png")
    plt.close(fig)


def fig_hist(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    x = b["cp"]["carry"].dropna()
    ax.hist(x, bins=50, density=True, color=plots.BLUE, alpha=0.7, label="Carry portfolio, monthly")
    g = np.linspace(x.min(), x.max(), 200)
    ax.plot(g, np.exp(-0.5 * ((g - x.mean()) / x.std()) ** 2) / (x.std() * math.sqrt(2 * math.pi)), color=plots.INK,
            linewidth=1.2, label="Normal with the same mean and volatility")
    plots.percent_axis(ax, "x")
    ax.set_xlabel("Monthly return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    st = b["table"].iloc[0]
    plots.titles(fig, "Lab Figure 11.3  Up the stairs, down the elevator?",
                 _sub(sim, f"Skewness {st['skewness']:.2f} · worst month {st['worst_month']:.1%}"))
    plots.finish(fig, out / "fig11_3_hist.png")
    plt.close(fig)


def fig_drawdown(dd, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.2)
    x = dd["drawdown"]
    ax.fill_between(x.index.to_timestamp(), x, 0, color=plots.RED, alpha=0.5)
    plots.percent_axis(ax)
    ax.set_ylabel("Drawdown from previous peak")
    plots.titles(fig, "Lab Figure 11.4  Carry drawdowns",
                 _sub(sim, f"Largest {dd['max_drawdown']:.0%} ({dd['peak']} to {dd['trough']}; recovered {dd['recovered']})"))
    plots.finish(fig, out / "fig11_4_drawdown.png")
    plt.close(fig)


def fig_vix(b, d, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    if d.get("vix") is None:
        ax.text(0.5, 0.5, "VIX not loaded", ha="center", transform=ax.transAxes)
    else:
        df = pd.concat([b["cp"]["carry"].rename("y"), d["vix"].diff().rename("x")], axis=1).dropna()
        ax.scatter(df["x"], df["y"], s=12, alpha=0.6, color=plots.BLUE, edgecolors="none")
        c = np.polyfit(df["x"], df["y"], 1)
        g = np.linspace(df["x"].min(), df["x"].max(), 50)
        ax.plot(g, np.polyval(c, g), color=plots.RED, linewidth=1.5, label=f"Fitted slope {c[0]:.2%} per VIX point")
        ax.axhline(0, color=plots.GRID, linewidth=1)
        ax.axvline(0, color=plots.GRID, linewidth=1)
        plots.percent_axis(ax)
        ax.set_xlabel("Change in the VIX during the month (points)")
        ax.set_ylabel("Carry return")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    plots.titles(fig, "Lab Figure 11.5  Carry returns when fear rises",
                 _sub(sim, "Monthly carry returns against the change in the VIX"))
    plots.finish(fig, out / "fig11_5_vix.png")
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


STAT = {"mean_annual": _num("+.1%"), "vol_annual": _num(".1%"), "sharpe": _num(".2f"), "t": _num(".2f"),
        "skewness": _num("+.2f"), "worst_month": _num(".1%"), "max_drawdown": _num(".0%")}


def write_report(out, d, a, b, c, dd, sim):
    md = ["# Lab 11 results: Do High-Interest Currencies Depreciate Enough to Offset Their Yield?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    if d.get("missing"):
        md += ["> Series not loaded: " + "; ".join(d["missing"]), ""]
    md += ["**What you will explore:** whether high-interest currencies fall enough to cancel their extra yield, as uncovered interest parity says they should.",
           "", "## Part A: Fama regressions", "",
           _md(a.rename(columns={"currency": "Currency", "slope": "Slope", "se": "s.e.", "t_vs_1": "t (vs 1)",
                                 "t_vs_0": "t (vs 0)", "months": "Months", "start": "From"}),
               {"Slope": _num(".2f"), "s.e.": _num(".2f"), "t (vs 1)": _num(".2f"), "t (vs 0)": _num(".2f")}), "",
           "Dependent variable: monthly change in log dollars per foreign unit. Regressor: last month's U.S. minus "
           "foreign three-month rate, divided by 12. Newey–West errors (3 lags).", "", "![Lab Figure 11.1](fig11_1_fama.png)",
           "", "## Part B: The carry portfolio", "",
           _md(b["table"].rename(columns={"portfolio": "Portfolio"}), STAT), "",
           f"Carry return split: interest differential {b['components']['interest_annual']:+.1%} a year, exchange-rate "
           f"change {b['components']['fx_annual']:+.1%} a year.", "", "![Lab Figure 11.2](fig11_2_cumulative.png)", "",
           "![Lab Figure 11.3](fig11_3_hist.png)", "", "## Part C: Stress tests", "", "**Subperiods**", "",
           _md(c["subperiods"], STAT), "", "**Transaction costs** (per unit of weight traded)", "",
           _md(c["costs"], {"cost_bp": _num(".0f"), "mean_annual": _num("+.1%"), "sharpe": _num(".2f"),
                            "avg_turnover": _num(".2f")}), ""]
    if len(c["exposures"]):
        md += ["**Exposure to stocks and to fear**", "",
               _md(c["exposures"], {"beta": _num("+.3f"), "t": _num(".2f"), "beta_bad_months": _num("+.3f"),
                                    "corr": _num("+.2f")}), "",
               "beta_bad_months: slope estimated only in months when stocks fell (or the VIX rose).", ""]
    md += ["**Worst ten months**", "", _md(c["worst"].reset_index().rename(columns={"date": "Month"}),
                                          {"carry": _num(".1%"), "high": _num(".1%"), "low": _num(".1%")}), "",
           "## Part D: The investor's view", "",
           f"Largest drawdown {dd['max_drawdown']:.0%} from {dd['peak']} to {dd['trough']}; recovered: {dd['recovered']}. "
           f"Positive in {dd['share_months_positive']:.0%} of months; the worst 5% of months produced "
           f"{dd['share_losses_worst_5pct']:.0%} of all losses.", "", "![Lab Figure 11.4](fig11_4_drawdown.png)", "",
           "![Lab Figure 11.5](fig11_5_vix.png)", "", "## Questions to think about", "",
           "1. Do high-rate currencies depreciate by their interest advantage, as UIP requires?",
           "2. Does the carry portfolio earn a premium after costs? How is it distributed over time?",
           "3. When does carry lose money, and what does that suggest about why it pays?", ""]
    (out / "lab11_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fred", "sim"], default="fred")
    ap.add_argument("--data-dir", default=None, help="folder with FRED CSVs named <SERIES>.csv (optional)")
    ap.add_argument("--ff3-file", default=None)
    ap.add_argument("--cost", type=float, default=0.0, help="cost per unit of weight traded, e.g. 0.001")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"ff3": args.ff3_file}
    if args.data_dir:
        for p in Path(args.data_dir).glob("*.csv"):
            files[p.stem.upper()] = str(p)
    try:
        d = load(args.source, files, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    er = excess_returns(d["spot"], d["rate"], d["us"])
    a = part_a(er)
    b = part_b(er, args.cost)
    c = part_c(er, d, b)
    dd = part_d(b)
    fig_fama(a, out, sim)
    fig_cumulative(b, out, sim)
    fig_hist(b, out, sim)
    fig_drawdown(dd, out, sim)
    fig_vix(b, d, out, sim)
    a.to_csv(out / "table_fama.csv", index=False)
    b["table"].to_csv(out / "table_carry.csv", index=False)
    b["cp"].to_csv(out / "table_carry_monthly.csv")
    c["subperiods"].to_csv(out / "table_subperiods.csv", index=False)
    c["costs"].to_csv(out / "table_costs.csv", index=False)
    write_report(out, d, a, b, c, dd, sim)
    summary = {"simulated": sim, "fama": a.to_dict(orient="records"), "carry": b["table"].to_dict(orient="records"),
               "components": b["components"], "exposures": c["exposures"].to_dict(orient="records"),
               "max_drawdown": dd["max_drawdown"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(a.round(2).to_string(index=False))
    print(b["table"].round(3).to_string(index=False))
    print(f"Report: {out / 'lab11_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
