"""Lab 16: Do Credit Spreads Compensate Mainly for Expected Default Losses?

Chapter 16 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

CLAIM ON TRIAL
  "The yield spread of corporate bonds over Treasuries mostly reflects the
   losses investors expect from default."

WHAT THE LAB DOES
  Part A  Spreads since 1919 (Baa-Aaa) and 1953 (Baa and Aaa over the 10-year
          Treasury): levels, decades, recessions.
  Part B  Decomposition: expected default losses from adjustable default
          and recovery assumptions versus the observed spread; the default
          rate that would be needed to justify the whole spread.
  Part C  The residual as a risk premium: behaviour in recessions and power
          to forecast industrial production.
  Part D  Stress tests: a grid of default and recovery assumptions, crisis
          peaks and subperiods.

DATA (free, FRED)
  AAA, BAA (Moody's seasoned corporate bond yields, from 1919), GS10 (from
  1953), USREC (NBER recessions), INDPRO (industrial production).
  Optional: Gilchrist-Zakrajsek excess bond premium (Federal Reserve).

USAGE
  python lab16.py
  python lab16.py --pd-baa 0.0027 --pd-aaa 0.0 --recovery 0.38
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
from fintest.credit import breakeven_default_rate, expected_loss_spread, monthly_fred, read_ebp, simulated_credit
from fintest.data import DataUnavailable
from fintest.stats import ols_newey_west

ROOT = Path(__file__).resolve().parent
SERIES = ["AAA", "BAA", "GS10", "USREC", "INDPRO"]
PEAKS = [("Great Depression", "1929-09", "1934-12"), ("1974–75", "1973-11", "1975-12"), ("1980–82", "1980-01", "1983-06"),
         ("2001–02", "2001-03", "2003-03"), ("2008–09", "2007-12", "2009-12"), ("2020", "2020-01", "2020-12")]


def load(source: str, files: dict, seed: int = 0) -> dict:
    if source == "sim":
        df = simulated_credit(seed=seed)
        return {"df": df, "ebp": None, "simulated": True}
    cols = {s: monthly_fred(s, files.get(s.lower())) for s in SERIES}
    df = pd.DataFrame(cols)
    for c in ("AAA", "BAA", "GS10"):
        df[c] = df[c] / 100
    ebp = read_ebp(files["ebp"]) if files.get("ebp") else None
    return {"df": df.dropna(subset=["AAA", "BAA"]), "ebp": ebp, "simulated": False}


def spreads(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({"Baa–Aaa": df["BAA"] - df["AAA"], "Baa–Treasury": df["BAA"] - df["GS10"],
                      "Aaa–Treasury": df["AAA"] - df["GS10"]})
    s["recession"] = df["USREC"]
    return s


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def part_a(s: pd.DataFrame) -> dict:
    rows = []
    for c in ("Baa–Aaa", "Baa–Treasury", "Aaa–Treasury"):
        x = s[c].dropna()
        r = s.loc[x.index, "recession"]
        rows.append({"spread": c, "from": str(x.index[0]), "mean": x.mean(), "median": x.median(), "min": x.min(),
                     "max": x.max(), "max_date": str(x.idxmax()), "mean_recession": x[r == 1].mean(),
                     "mean_expansion": x[r == 0].mean()})
    dec = s[["Baa–Aaa", "Baa–Treasury"]].groupby((s.index.year // 10) * 10).mean()
    dec.index = [f"{d}s" for d in dec.index]
    return {"table": pd.DataFrame(rows), "decades": dec}


def part_b(s: pd.DataFrame, pd_baa: float, pd_aaa: float, recovery: float) -> dict:
    el_baa = float(expected_loss_spread(pd_baa, recovery))
    el_aaa = float(expected_loss_spread(pd_aaa, recovery))
    el = {"Baa–Treasury": el_baa, "Aaa–Treasury": el_aaa, "Baa–Aaa": el_baa - el_aaa}
    rows = []
    for c, e in el.items():
        x = s[c].dropna()
        rows.append({"spread": c, "average_spread": x.mean(), "expected_loss": e, "residual": x.mean() - e,
                     "share_expected_loss": e / x.mean(), "share_months_above_half": float((e / x > 0.5).mean())})
    be = pd.DataFrame({"Baa–Treasury": breakeven_default_rate(s["Baa–Treasury"], recovery),
                       "Baa–Aaa (relative to Aaa)": breakeven_default_rate(s["Baa–Aaa"], recovery)}, index=s.index)
    return {"table": pd.DataFrame(rows), "breakeven": be, "el": el,
            "assumptions": {"pd_baa": pd_baa, "pd_aaa": pd_aaa, "recovery": recovery}}


def part_c(s: pd.DataFrame, df: pd.DataFrame, ebp: pd.DataFrame | None, h: int = 12) -> dict:
    ip = np.log(df["INDPRO"])
    fut = (ip.shift(-h) - ip).rename("ip_growth")  # log growth over the next h months
    rows = []
    preds = {"Baa–Aaa": s["Baa–Aaa"], "Baa–Treasury": s["Baa–Treasury"]}
    if ebp is not None:
        if "gz_spread" in ebp:
            preds["GZ credit spread"] = ebp["gz_spread"]
        preds["Excess bond premium (GZ)"] = ebp["ebp"]
    slope = (df["GS10"] - 0).rename("level")
    for name, x in preds.items():
        d = pd.concat([fut, x.rename("x")], axis=1).dropna()
        d = d[d.index >= pd.Period("1953-04", "M")] if name.startswith("Baa–T") else d
        r = ols_newey_west(d["ip_growth"].to_numpy(), d["x"].to_numpy(), lags=18)
        e = d["ip_growth"] - (r["coef"][0, 0] + r["coef"][1, 0] * d["x"])
        sd = d["x"].std()
        rows.append({"predictor": name, "from": str(d.index[0]), "effect_of_1sd": float(r["coef"][1, 0] * sd),
                     "t": float(r["t"][1, 0]), "r2": float(1 - e.var() / d["ip_growth"].var()), "months": len(d)})
    # spread around recession starts
    rec = df["USREC"].fillna(0)
    starts = rec.index[(rec == 1) & (rec.shift(1) == 0)]
    path = []
    x = s["Baa–Aaa"]
    for st in starts:
        i = x.index.get_loc(st)
        if i - 12 < 0 or i + 24 >= len(x):
            continue
        path.append(x.iloc[i - 12:i + 25].to_numpy() - x.iloc[i - 12])
    event = pd.DataFrame(path, columns=range(-12, 25)).mean() if path else pd.Series(dtype=float)
    q = pd.concat([fut, s["Baa–Aaa"].rename("x")], axis=1).dropna()
    q["quintile"] = pd.qcut(q["x"], 5, labels=[1, 2, 3, 4, 5])
    byq = q.groupby("quintile", observed=True).agg(spread=("x", "mean"), ip_growth=("ip_growth", "mean")).reset_index()
    return {"forecast": pd.DataFrame(rows), "event": event, "n_events": len(path), "quintiles": byq}


def part_d(s: pd.DataFrame, pds=(0.001, 0.0027, 0.005, 0.01, 0.02), recs=(0.2, 0.38, 0.5, 0.6)) -> dict:
    avg = s["Baa–Treasury"].dropna().mean()
    grid = pd.DataFrame([[float(expected_loss_spread(p, r)) / avg for r in recs] for p in pds],
                        index=[f"{p:.2%}" for p in pds], columns=[f"{r:.0%}" for r in recs])
    rows = []
    for label, a, b in PEAKS:
        x = s.loc[pd.Period(a, "M"):pd.Period(b, "M")]
        if x["Baa–Aaa"].dropna().empty:
            continue
        rows.append({"episode": label, "peak_baa_aaa": x["Baa–Aaa"].max(), "peak_date": str(x["Baa–Aaa"].idxmax()),
                     "peak_baa_tsy": x["Baa–Treasury"].max() if x["Baa–Treasury"].notna().any() else float("nan")})
    sub = []
    for label, a, b in (("1919–1952", None, "1952-12"), ("1953–1984", "1953-01", "1984-12"),
                        ("1985–2007", "1985-01", "2007-12"), ("2008–", "2008-01", None)):
        x = s.loc[pd.Period(a, "M") if a else None:pd.Period(b, "M") if b else None]
        sub.append({"period": label, "baa_aaa": x["Baa–Aaa"].mean(), "baa_tsy": x["Baa–Treasury"].mean()})
    return {"grid": grid, "avg_baa_tsy": avg, "peaks": pd.DataFrame(rows), "subperiods": pd.DataFrame(sub)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def _shade(ax, rec):
    r = rec.fillna(0).astype(int)
    on = None
    for d, v in r.items():
        if v and on is None:
            on = d
        if not v and on is not None:
            ax.axvspan(on.to_timestamp(), d.to_timestamp(), color=plots.SHADE, linewidth=0)
            on = None
    if on is not None:
        ax.axvspan(on.to_timestamp(), r.index[-1].to_timestamp(), color=plots.SHADE, linewidth=0)


def fig_spreads(s, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    _shade(ax, s["recession"])
    dates = s.index.to_timestamp()
    ax.plot(dates, s["Baa–Treasury"], color=plots.BLUE, linewidth=1.2, label="Baa minus 10-year Treasury")
    ax.plot(dates, s["Baa–Aaa"], color=plots.ORANGE, linewidth=1.2, label="Baa minus Aaa")
    plots.percent_axis(ax)
    ax.set_ylabel("Yield spread")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    plots.titles(fig, "Lab Figure 16.1  Corporate bond spreads since 1919",
                 _sub(sim, "Moody's seasoned corporate bond yields · shaded: NBER recessions"))
    plots.finish(fig, out / "fig16_1_spreads.png")
    plt.close(fig)


def fig_decomposition(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.4)
    t = b["table"]
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["expected_loss"], color=plots.INK_3, label="Expected default loss (assumed)")
    ax.barh(y, t["residual"], left=t["expected_loss"], color=plots.BLUE, label="Residual: risk premium, liquidity, taxes")
    for yy, (_, row) in zip(y, t.iterrows()):
        ax.text(row.average_spread + 0.0003, yy, f"{row.share_expected_loss:.0%} expected loss", va="center",
                fontsize=8.5, color=plots.INK_2)
    ax.set_yticks(y, t["spread"])
    plots.percent_axis(ax, "x")
    ax.set_xlim(0, t["average_spread"].max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average spread")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    a = b["assumptions"]
    plots.titles(fig, "Lab Figure 16.2  How much of the spread is expected default loss?",
                 _sub(sim, f"Assumed annual default rates: Baa {a['pd_baa']:.2%}, Aaa {a['pd_aaa']:.2%} · recovery "
                           f"{a['recovery']:.0%}"))
    plots.finish(fig, out / "fig16_2_decomposition.png")
    plt.close(fig)


def fig_breakeven(b, s, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    _shade(ax, s["recession"])
    be = b["breakeven"]["Baa–Treasury"].dropna()
    ax.plot(be.index.to_timestamp(), be, color=plots.BLUE, linewidth=1.2,
            label="Default rate needed for expected loss to equal the Baa–Treasury spread")
    ax.axhline(b["assumptions"]["pd_baa"], color=plots.RED, linestyle="--", linewidth=1.2,
               label=f"Assumed long-run Baa default rate ({b['assumptions']['pd_baa']:.2%})")
    plots.percent_axis(ax)
    ax.set_ylabel("Annual default rate")
    ax.legend(frameon=False, fontsize=8.5, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plots.titles(fig, "Lab Figure 16.3  The default rate the spread would require",
                 _sub(sim, f"Recovery {b['assumptions']['recovery']:.0%} · shaded: recessions"))
    plots.finish(fig, out / "fig16_3_breakeven.png")
    plt.close(fig)


def fig_forecast(c, out, sim):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    fig.patch.set_facecolor(plots.SURFACE)
    for ax in (a1, a2):
        plots.style_axes(ax)
    q = c["quintiles"]
    a1.bar(q["quintile"].astype(int), q["ip_growth"], color=plots.BLUE)
    a1.axhline(0, color=plots.INK_3, linewidth=1)
    a1.set_xticks(q["quintile"].astype(int), [f"Q{i}\n{v:.1%}" for i, v in zip(q["quintile"].astype(int), q["spread"])],
                  fontsize=8.5)
    plots.percent_axis(a1)
    a1.set_title("Output growth over the next 12 months, by Baa–Aaa spread quintile", fontsize=9.5, loc="left",
                 color=plots.INK)
    ev = c["event"]
    if len(ev):
        a2.plot(ev.index, ev.values, color=plots.ORANGE, linewidth=1.8)
        a2.axvline(0, color=plots.INK_3, linestyle="--", linewidth=1)
        a2.axhline(0, color=plots.GRID, linewidth=1)
        plots.percent_axis(a2)
        a2.set_xlabel("Months from the start of a recession")
        a2.set_title(f"Change in Baa–Aaa spread around {c['n_events']} recession starts", fontsize=9.5, loc="left",
                     color=plots.INK)
    plots.titles(fig, "Lab Figure 16.4  Spreads, recessions and future output",
                 _sub(sim, "Industrial production growth (log) and NBER recession dates"))
    plots.finish(fig, out / "fig16_4_cycle.png")
    plt.close(fig)


def fig_grid(dd, out, sim):
    fig, ax, plt = plots.new_figure(8.0, 4.6)
    g = dd["grid"]
    im = ax.imshow(g.to_numpy(), cmap="Blues", vmin=0, vmax=max(1.0, float(g.to_numpy().max())), aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 * max(1.0, float(g.to_numpy().max())) else plots.INK)
    ax.set_xticks(range(g.shape[1]), g.columns)
    ax.set_yticks(range(g.shape[0]), g.index)
    ax.set_xlabel("Recovery rate")
    ax.set_ylabel("Annual Baa default rate")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plots.titles(fig, "Lab Figure 16.5  Share of the Baa–Treasury spread explained by expected loss",
                 _sub(sim, f"Average spread {dd['avg_baa_tsy']:.2%} · each cell: expected-loss spread ÷ average spread"))
    plots.finish(fig, out / "fig16_5_grid.png")
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


P2 = _num(".2%")


def write_report(out, a, b, c, dd, sim):
    md = ["# Lab 16 results: Do Credit Spreads Compensate Mainly for Expected Default Losses?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**Claim on trial:** the yield spread of corporate bonds over Treasuries mostly reflects expected default losses.",
           "", "## Part A: Spreads", "",
           _md(a["table"].rename(columns={"spread": "Spread", "from": "From", "mean": "Mean", "median": "Median",
                                          "min": "Min", "max": "Max", "max_date": "Max date",
                                          "mean_recession": "Mean in recessions", "mean_expansion": "Mean in expansions"}),
               {"Mean": P2, "Median": P2, "Min": P2, "Max": P2, "Mean in recessions": P2, "Mean in expansions": P2}), "",
           "**By decade**", "", _md(a["decades"].reset_index().rename(columns={"index": "Decade"}),
                                    {"Baa–Aaa": P2, "Baa–Treasury": P2}), "", "![Lab Figure 16.1](fig16_1_spreads.png)", "",
           "## Part B: Expected losses versus spreads", ""]
    asn = b["assumptions"]
    md += [f"Assumptions: annual default rates Baa {asn['pd_baa']:.2%}, Aaa {asn['pd_aaa']:.2%}; recovery "
           f"{asn['recovery']:.0%} of face value. Expected-loss spread = −ln(1 − PD × (1 − R)).", "",
           _md(b["table"].rename(columns={"spread": "Spread", "average_spread": "Average spread",
                                          "expected_loss": "Expected loss", "residual": "Residual",
                                          "share_expected_loss": "Share expected loss",
                                          "share_months_above_half": "Months where EL > half the spread"}),
               {"Average spread": P2, "Expected loss": P2, "Residual": P2, "Share expected loss": _num(".0%"),
                "Months where EL > half the spread": _num(".0%")}), "",
           f"Break-even Baa default rate (whole Baa–Treasury spread as expected loss): median "
           f"{b['breakeven']['Baa–Treasury'].median():.2%}, peak {b['breakeven']['Baa–Treasury'].max():.2%}.", "",
           "Moody's seasoned yields are on long-maturity bonds (around 20 years or more), while the Treasury benchmark is "
           "10 years; corporate interest is also subject to state income tax, which Treasury interest is not. Both widen "
           "the measured spread for reasons other than default.", "", "![Lab Figure 16.2](fig16_2_decomposition.png)", "",
           "![Lab Figure 16.3](fig16_3_breakeven.png)", "", "## Part C: The residual as a risk premium", "",
           _md(c["forecast"].rename(columns={"predictor": "Predictor", "from": "From",
                                             "effect_of_1sd": "Effect of +1 s.d. on next-12-month IP growth", "t": "t",
                                             "r2": "R²", "months": "Months"}),
               {"Effect of +1 s.d. on next-12-month IP growth": _num("+.2%"), "t": _num(".2f"), "R²": _num(".2f")}), "",
           _md(c["quintiles"].rename(columns={"quintile": "Spread quintile", "spread": "Average Baa–Aaa",
                                              "ip_growth": "Next-12-month IP growth"}),
               {"Average Baa–Aaa": P2, "Next-12-month IP growth": _num("+.2%")}), "",
           "Newey–West t-statistics with 18 lags (overlapping 12-month growth).", "", "![Lab Figure 16.4](fig16_4_cycle.png)",
           "", "## Part D: Stress tests", "", "**Share of the average Baa–Treasury spread explained by expected loss**", "",
           _md(dd["grid"].reset_index().rename(columns={"index": "Baa default rate \\ recovery"}),
               {c_: _num(".0%") for c_ in dd["grid"].columns}), "", "![Lab Figure 16.5](fig16_5_grid.png)", "",
           "**Crisis peaks**", "",
           _md(dd["peaks"].rename(columns={"episode": "Episode", "peak_baa_aaa": "Peak Baa–Aaa", "peak_date": "Date",
                                           "peak_baa_tsy": "Peak Baa–Treasury"}),
               {"Peak Baa–Aaa": P2, "Peak Baa–Treasury": P2}), "", "**Subperiods**", "",
           _md(dd["subperiods"].rename(columns={"period": "Period", "baa_aaa": "Baa–Aaa", "baa_tsy": "Baa–Treasury"}),
               {"Baa–Aaa": P2, "Baa–Treasury": P2}), "", "## Your verdict", "",
           "1. Under reasonable default and recovery assumptions, what share of the Baa spread is expected default loss?",
           "2. What default rate would be needed to justify the whole spread? Has it ever been observed?",
           "3. Does the residual behave like a risk premium: rising in recessions and forecasting weaker activity?", ""]
    (out / "lab16_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fred", "sim"], default="fred")
    for s in SERIES:
        ap.add_argument(f"--{s.lower()}-file", dest=s.lower(), default=None, help=f"FRED {s} CSV")
    ap.add_argument("--ebp-file", dest="ebp", default=None, help="Gilchrist-Zakrajsek ebp_csv.csv (optional)")
    ap.add_argument("--pd-baa", type=float, default=0.0027, help="annual Baa default rate (Moody's 1920-2006: 0.27%%)")
    ap.add_argument("--pd-aaa", type=float, default=0.0, help="annual Aaa default rate")
    ap.add_argument("--recovery", type=float, default=0.38, help="recovery rate (Moody's senior unsecured 1982-2006: 38%%)")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: getattr(args, k) for k in [s.lower() for s in SERIES] + ["ebp"]}
    try:
        d = load(args.source, files, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    s = spreads(d["df"])
    a = part_a(s)
    b = part_b(s, args.pd_baa, args.pd_aaa, args.recovery)
    c = part_c(s, d["df"], d["ebp"])
    dd = part_d(s)
    fig_spreads(s, out, sim)
    fig_decomposition(b, out, sim)
    fig_breakeven(b, s, out, sim)
    fig_forecast(c, out, sim)
    fig_grid(dd, out, sim)
    a["table"].to_csv(out / "table_spreads.csv", index=False)
    b["table"].to_csv(out / "table_decomposition.csv", index=False)
    c["forecast"].to_csv(out / "table_forecast.csv", index=False)
    dd["grid"].to_csv(out / "table_grid.csv")
    write_report(out, a, b, c, dd, sim)
    summary = {"simulated": sim, "spreads": a["table"].to_dict(orient="records"),
               "decomposition": b["table"].to_dict(orient="records"), "forecast": c["forecast"].to_dict(orient="records"),
               "assumptions": b["assumptions"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(b["table"].round(4).to_string(index=False))
    print(c["forecast"].round(3).to_string(index=False))
    print(f"Report: {out / 'lab16_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
