"""Lab 15: The Low-Risk Anomaly.

Chapter 15 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  Whether low-risk stocks earn more than the CAPM predicts, and how much the
  answer depends on how the portfolios are built.

WHAT THE LAB DOES
  Part A  The security market line: average excess returns, realized betas
          and alphas of portfolios sorted on beta, total volatility and
          idiosyncratic volatility.
  Part B  Long-short constructions: low minus high, and a betting-against-
          beta (BAB) style portfolio that is leveraged to be beta-neutral
          using betas estimated in real time. CAPM and four-factor alphas.
  Part C  Stress tests: equal vs value weights, subperiods, interest-rate
          exposure, crisis years, rolling alphas.
  Part D  The investor's view: a low-volatility portfolio against the market,
          including tracking error and years of underperformance.

DATA (free)
  Kenneth R. French Data Library: Portfolios Formed on Market Beta, on
  Variance and on Residual Variance (from 1963); Fama/French 3 Factors;
  Momentum Factor.
  Optional: FRED GS10 (10-year Treasury yield) for Part C; AQR's "Betting
  Against Beta: Equity Factors, Monthly" for comparison.

USAGE
  python lab15.py
  python lab15.py --source sim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.data import DataUnavailable, load_french_factors, load_french_tables
from fintest.inference import moments
from fintest.stats import ols_newey_west
from fintest.vol import load_fred

ROOT = Path(__file__).resolve().parent
SORTS = {"beta": "Market beta", "var": "Total volatility", "resvar": "Idiosyncratic volatility"}
SORT_COLORS = {"beta": plots.BLUE, "var": plots.ORANGE, "resvar": plots.VIOLET, "aqr": plots.INK_2}
SUBPERIODS = [("1963–1990", None, "1990-12"), ("1991–2009", "1991-01", "2009-12"), ("2010–", "2010-01", None)]
CRISIS_YEARS = [2008, 2020, 2022]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _col(df, *names):
    cols = {c.strip().lower().replace(" ", "").replace("-", ""): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "").replace("-", "")
        if k in cols:
            return df[cols[k]]
    raise KeyError(f"none of {names} in {list(df.columns)}")


def _block(df: pd.DataFrame, first: str, n: int) -> pd.DataFrame:
    """The n columns starting at `first` (e.g. 'Lo 10' -> the ten deciles)."""
    names = [c.strip() for c in df.columns]
    if first not in names:
        raise KeyError(f"column {first!r} not found; columns are {names}")
    i = names.index(first)
    out = df.iloc[:, i:i + n].copy()
    out.columns = [f"{'D' if n == 10 else 'Q'}{k + 1}" for k in range(n)]
    return out


def _weighted_tables(tables: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    titles = list(tables)
    vw = next((tables[t] for t in titles if "value" in t.lower()), tables[titles[0]])
    ew = next((tables[t] for t in titles if "equal" in t.lower()), tables[titles[1]] if len(titles) > 1 else None)
    return vw, ew


def read_aqr_bab(path, country: str = "USA") -> pd.Series:
    """AQR 'Betting Against Beta: Equity Factors, Monthly' (xlsx sheet 'BAB Factors', or a CSV saved from it).

    Finds the header row whose first cell is DATE and returns the `country` column as monthly decimal returns.
    """
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        try:
            raw = pd.read_excel(p, sheet_name="BAB Factors", header=None)
        except Exception as e:  # noqa: BLE001
            raise DataUnavailable(f"Could not read the 'BAB Factors' sheet of {p} ({e}); save it as CSV instead.") from e
    else:
        raw = pd.read_csv(p, header=None, dtype=str)
    first = raw.iloc[:, 0].astype(str).str.strip().str.upper()
    hdr = first[first == "DATE"].index
    if len(hdr) == 0:
        raise DataUnavailable("No header row starting with DATE in the AQR file.")
    h = hdr[0]
    cols = [str(c).strip() for c in raw.iloc[h]]
    body = raw.iloc[h + 1:].copy()
    body.columns = cols
    if country not in cols:
        raise DataUnavailable(f"Column {country!r} not in the AQR file; found {cols[:10]}...")
    dates = pd.to_datetime(body["DATE"], errors="coerce")
    vals = pd.to_numeric(body[country], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=dates).dropna()
    s = s[s.index.notna()]
    s.index = s.index.to_period("M")
    return s.rename("aqr")


def load(source: str, files: dict, sim_seed: int = 0) -> dict:
    if source == "sim":
        return simulated(seed=sim_seed)
    ff3 = load_french_factors("ff3_monthly", path=files.get("ff3"))
    mom = load_french_factors("momentum_monthly", path=files.get("mom"))
    d = {"mkt": _col(ff3, "Mkt-RF"), "smb": _col(ff3, "SMB"), "hml": _col(ff3, "HML"), "rf": _col(ff3, "RF"),
         "umd": _col(mom, "Mom", "UMD"), "simulated": False}
    for key, ds in (("beta", "beta_portfolios"), ("var", "var_portfolios"), ("resvar", "resvar_portfolios")):
        vw, ew = _weighted_tables(load_french_tables(ds, path=files.get(key)))
        d[key] = {"vw": _block(vw, "Lo 10", 10), "ew": _block(ew, "Lo 10", 10) if ew is not None else None,
                  "vw_q": _block(vw, "Lo 20", 5)}
    d["dy10"] = None
    if not files.get("no_gs10"):
        try:
            gs = load_fred("GS10", files.get("gs10"))
            gs.index = gs.index.to_period("M")
            d["dy10"] = (gs.groupby(level=0).last() / 100).diff().rename("dy10")
        except DataUnavailable as err:
            print(f"GS10 not loaded ({str(err).splitlines()[0]}); skipping the interest-rate test.")
    d["aqr"] = read_aqr_bab(files["aqr"]) if files.get("aqr") else None
    return d


def simulated(n_months: int = 750, seed: int = 0, start: str = "1963-07") -> dict:
    """Fake factor and portfolio returns with a flat security market line and bond-like low-risk stocks. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    t5 = lambda k: rng.standard_t(5, (n_months, k)) / math.sqrt(5 / 3)  # noqa: E731
    f = t5(4)
    mkt, smb, hml, umd = 0.0060 + 0.045 * f[:, 0], 0.002 + 0.03 * f[:, 1], 0.003 + 0.03 * f[:, 2], 0.006 + 0.04 * f[:, 3]
    rf = np.full(n_months, 0.0035)
    dy = 0.0025 * rng.standard_normal(n_months)
    d = {"mkt": pd.Series(mkt, idx), "smb": pd.Series(smb, idx), "hml": pd.Series(hml, idx), "rf": pd.Series(rf, idx),
         "umd": pd.Series(umd, idx), "dy10": pd.Series(dy, idx, name="dy10"), "aqr": None, "simulated": True}
    k = np.arange(10)
    specs = {"beta": (np.linspace(0.55, 1.65, 10), np.full(10, 0.0055)),
             "var": (np.linspace(0.65, 1.50, 10), np.linspace(0.0065, 0.0035, 10)),
             "resvar": (np.linspace(0.75, 1.40, 10), np.linspace(0.0062, 0.0030, 10))}
    for key, (betas, means) in specs.items():
        idio = np.linspace(0.012, 0.045, 10)
        rate = -2.5 * (1 - k / 9)  # low-risk deciles fall when yields rise
        common = means - betas * 0.0060 + betas[None, :] * mkt[:, None] + rate[None, :] * dy[:, None]
        vw = common + 0.15 * (k[None, :] / 9) * smb[:, None] + idio * rng.standard_normal((n_months, 10))
        ew = vw + 0.4 * smb[:, None] + 0.001 * (1 - k / 9) + 0.01 * rng.standard_normal((n_months, 10))
        cols = [f"D{i + 1}" for i in range(10)]
        vwdf = pd.DataFrame(vw + rf[:, None], idx, cols)
        q = pd.DataFrame({f"Q{j + 1}": vwdf.iloc[:, 2 * j:2 * j + 2].mean(axis=1) for j in range(5)})
        d[key] = {"vw": vwdf, "ew": pd.DataFrame(ew + rf[:, None], idx, cols), "vw_q": q}
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factors(d, model: str) -> pd.DataFrame:
    cols = {"capm": ["mkt"], "ff4": ["mkt", "smb", "hml", "umd"]}[model]
    return pd.concat({c: d[c] for c in cols}, axis=1)


def alpha(y: pd.Series, d: dict, model: str = "capm") -> dict:
    X = _factors(d, model)
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    r = ols_newey_west(df["y"].to_numpy(), df[X.columns].to_numpy())
    return {"alpha_annual": 12 * float(r["coef"][0, 0]), "alpha_t": float(r["t"][0, 0]),
            "beta_mkt": float(r["coef"][1, 0]), "months": len(df)}


def _stats(r: pd.Series) -> dict:
    x = r.dropna()
    v = x.to_numpy()
    m, s = v.mean(), v.std(ddof=1)
    w = (1 + x).cumprod()
    return {"mean_annual": 12 * m, "t": m / (s / math.sqrt(len(v))), "vol_annual": s * math.sqrt(12),
            "sharpe": m / s * math.sqrt(12), "skewness": moments(v)["skewness"],
            "max_drawdown": float((w / w.cummax() - 1).min()), "months": len(v)}


def _slice(s, a=None, b=None):
    return s.loc[pd.Period(a, "M") if a else None:pd.Period(b, "M") if b else None]


def excess(port: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    return port.sub(rf.reindex(port.index), axis=0)


def exante_beta(y: pd.Series, mkt: pd.Series, window: int = 60, shrink: float = 1.0) -> pd.Series:
    """Rolling beta over the previous `window` months, known at the start of each month.

    `shrink` < 1 pulls the estimate toward 1, as Frazzini and Pedersen (2014) do for noisy single-stock betas
    (they use 0.6). Portfolio betas are estimated precisely, so the default applies no shrinkage; shrinking them
    understates the low leg's leverage and leaves the portfolio with a negative realized beta.
    """
    cov = y.rolling(window, min_periods=36).cov(mkt)
    var = mkt.rolling(window, min_periods=36).var()
    return (shrink * cov / var + (1 - shrink)).shift(1)


def low_high(dec_ex: pd.DataFrame, d: dict, n_side: int = 3) -> dict:
    """Low-minus-high and BAB-style long-short returns from ten decile excess returns.

    Low = average of the n_side lowest deciles, high = average of the n_side highest. The BAB-style portfolio holds
    1/beta_L of the low leg and shorts 1/beta_H of the high leg, each financed at the risk-free rate, using
    ex-ante betas (Frazzini and Pedersen 2014), so its ex-ante beta is zero.
    """
    lo = dec_ex.iloc[:, :n_side].mean(axis=1)
    hi = dec_ex.iloc[:, -n_side:].mean(axis=1)
    mkt = d["mkt"].reindex(dec_ex.index)
    bl, bh = exante_beta(lo, mkt), exante_beta(hi, mkt)
    bab = (lo / bl - hi / bh).dropna()
    return {"lmh": (lo - hi).rename("lmh"), "bab": bab.rename("bab"), "beta_low": bl, "beta_high": bh,
            "lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# Part A: security market line
# ---------------------------------------------------------------------------


def part_a(d: dict) -> dict:
    out = {}
    for key in SORTS:
        for w in ("vw", "ew"):
            port = d[key].get(w)
            if port is None:
                continue
            ex = excess(port, d["rf"])
            rows = []
            for c in ex.columns:
                a1, a4 = alpha(ex[c], d, "capm"), alpha(ex[c], d, "ff4")
                st = _stats(ex[c])
                rows.append({"portfolio": c, "mean_excess_annual": st["mean_annual"], "vol_annual": st["vol_annual"],
                             "sharpe": st["sharpe"], "beta": a1["beta_mkt"], "capm_alpha": a1["alpha_annual"],
                             "capm_t": a1["alpha_t"], "ff4_alpha": a4["alpha_annual"], "ff4_t": a4["alpha_t"]})
            t = pd.DataFrame(rows)
            slope = np.polyfit(t["beta"], t["mean_excess_annual"], 1)
            out[(key, w)] = {"table": t, "sml_slope": float(slope[0]), "sml_intercept": float(slope[1])}
    mkt = d["mkt"].reindex(d["beta"]["vw"].index).dropna()
    return {"sorts": out, "market_premium": 12 * float(mkt.mean()),
            "period": f"{d['beta']['vw'].index[0]} to {d['beta']['vw'].index[-1]}"}


# ---------------------------------------------------------------------------
# Part B: long-short constructions
# ---------------------------------------------------------------------------


def part_b(d: dict) -> dict:
    series, rows = {}, []
    for key, label in SORTS.items():
        lh = low_high(excess(d[key]["vw"], d["rf"]), d)
        series[key] = lh
        for kind, name in (("lmh", "Low minus high"), ("bab", "BAB-style (beta-neutral)")):
            y = lh[kind]
            rows.append({"sort": label, "construction": name, **_stats(y),
                         **{f"capm_{k}": v for k, v in alpha(y, d, "capm").items() if k != "months"},
                         **{f"ff4_{k}": v for k, v in alpha(y, d, "ff4").items() if k != "months"}})
    if d.get("aqr") is not None:
        y = d["aqr"]
        rows.append({"sort": "AQR BAB factor (USA)", "construction": "Frazzini–Pedersen, stock level", **_stats(y),
                     **{f"capm_{k}": v for k, v in alpha(y, d, "capm").items() if k != "months"},
                     **{f"ff4_{k}": v for k, v in alpha(y, d, "ff4").items() if k != "months"}})
    return {"series": series, "table": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Part C: stress tests
# ---------------------------------------------------------------------------


def part_c(d: dict, b: dict, window: int = 120) -> dict:
    # equal vs value weights
    ew_rows = []
    for key, label in SORTS.items():
        for w, wname in (("vw", "Value-weighted"), ("ew", "Equal-weighted")):
            if d[key].get(w) is None:
                continue
            y = low_high(excess(d[key][w], d["rf"]), d)["bab"]
            a4 = alpha(y, d, "ff4")
            ew_rows.append({"sort": label, "weighting": wname, "mean_annual": 12 * y.mean(),
                            "ff4_alpha": a4["alpha_annual"], "ff4_t": a4["alpha_t"]})
    # subperiods
    sub_rows = []
    for key, label in SORTS.items():
        y = b["series"][key]["bab"]
        for name, s, e in SUBPERIODS:
            ys = _slice(y, s, e)
            if len(ys) < 36:
                continue
            a1, a4 = alpha(ys, d, "capm"), alpha(ys, d, "ff4")
            sub_rows.append({"sort": label, "period": name, "mean_annual": 12 * ys.mean(), "sharpe": _stats(ys)["sharpe"],
                             "capm_alpha": a1["alpha_annual"], "capm_t": a1["alpha_t"], "ff4_alpha": a4["alpha_annual"],
                             "ff4_t": a4["alpha_t"]})
    # interest rates
    rate_rows = []
    if d.get("dy10") is not None:
        for key, label in SORTS.items():
            for kind, name in (("bab", "BAB-style"), ("lo", "Low-risk leg (excess)")):
                y = b["series"][key][kind]
                df = pd.concat([y.rename("y"), d["mkt"].rename("mkt"), d["dy10"]], axis=1).dropna()
                r = ols_newey_west(df["y"].to_numpy(), df[["mkt", "dy10"]].to_numpy())
                rate_rows.append({"sort": label, "series": name, "dy10_coef": float(r["coef"][2, 0]),
                                  "dy10_t": float(r["t"][2, 0]), "alpha_annual": 12 * float(r["coef"][0, 0]),
                                  "alpha_t": float(r["t"][0, 0]), "months": len(df)})
    # crisis years
    lowvol = excess(d["var"]["vw_q"], d["rf"])["Q1"]
    cands = {"BAB-style, beta": b["series"]["beta"]["bab"], "BAB-style, volatility": b["series"]["var"]["bab"],
             "Low-volatility quintile (excess)": lowvol, "Market (excess)": d["mkt"]}
    cy = []
    for yr in CRISIS_YEARS:
        row = {"year": str(yr)}
        for name, s in cands.items():
            x = s[s.index.year == yr]
            row[name] = float((1 + x).prod() - 1) if len(x) >= 6 else float("nan")
        cy.append(row)
    # rolling CAPM alpha
    roll = {}
    for key in ("beta", "var"):
        y = b["series"][key]["bab"]
        mkt = d["mkt"].reindex(y.index)
        cov = y.rolling(window).cov(mkt)
        beta = cov / mkt.rolling(window).var()
        roll[key] = 12 * (y.rolling(window).mean() - beta * mkt.rolling(window).mean())
    return {"weights": pd.DataFrame(ew_rows), "subperiods": pd.DataFrame(sub_rows),
            "rates": pd.DataFrame(rate_rows), "crisis": pd.DataFrame(cy), "rolling": roll}


# ---------------------------------------------------------------------------
# Part D: the investor's view
# ---------------------------------------------------------------------------


def part_d(d: dict) -> dict:
    mkt = d["mkt"]
    rows, ser = [], {}
    for key, label in (("var", "Lowest-volatility quintile"), ("beta", "Lowest-beta quintile")):
        y = excess(d[key]["vw_q"], d["rf"])["Q1"]
        m = mkt.reindex(y.index)
        rel = y - m
        a1 = alpha(y, d, "capm")
        yearly = (1 + y).groupby(y.index.year).prod() - (1 + m).groupby(m.index.year).prod()
        st = _stats(y)
        rows.append({"portfolio": label, "mean_excess_annual": st["mean_annual"], "vol_annual": st["vol_annual"],
                     "sharpe": st["sharpe"], "beta": a1["beta_mkt"], "capm_alpha": a1["alpha_annual"],
                     "capm_t": a1["alpha_t"], "max_drawdown": st["max_drawdown"],
                     "tracking_error": rel.std() * math.sqrt(12),
                     "information_ratio": rel.mean() / rel.std() * math.sqrt(12),
                     "share_years_behind_market": float((yearly < 0).mean())})
        ser[key] = y
    st = _stats(mkt.reindex(ser["var"].index))
    rows.append({"portfolio": "Market", "mean_excess_annual": st["mean_annual"], "vol_annual": st["vol_annual"],
                 "sharpe": st["sharpe"], "beta": 1.0, "capm_alpha": 0.0, "capm_t": float("nan"),
                 "max_drawdown": st["max_drawdown"], "tracking_error": 0.0, "information_ratio": float("nan"),
                 "share_years_behind_market": float("nan")})
    return {"table": pd.DataFrame(rows), "lowvol": ser["var"], "market": mkt.reindex(ser["var"].index)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_sml(a, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 5.0)
    for w, color, name in (("vw", plots.BLUE, "Value-weighted deciles"), ("ew", plots.ORANGE, "Equal-weighted deciles")):
        if ("beta", w) not in a["sorts"]:
            continue
        t = a["sorts"][("beta", w)]["table"]
        ax.plot(t["beta"], t["mean_excess_annual"], "o-", color=color, label=name, markersize=6, linewidth=1.2)
        for i in (0, 9):
            dy = -4 if w == "vw" else 4
            ax.annotate(f"{'Lowest' if i == 0 else 'Highest'} beta", (t["beta"].iloc[i], t["mean_excess_annual"].iloc[i]),
                        textcoords="offset points", xytext=(-9 if i == 0 else 9, dy), fontsize=8, color=color,
                        ha="right" if i == 0 else "left", va="center")
    xs = np.array([0, max(1.9, ax.get_xlim()[1])])
    ax.plot(xs, a["market_premium"] * xs, color=plots.INK_3, linestyle="--", linewidth=1.2,
            label="CAPM prediction (market premium × beta)")
    ax.plot([1], [a["market_premium"]], "s", color=plots.INK, markersize=7, label="Market")
    ax.set_xlim(0, xs[1])
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    plots.percent_axis(ax)
    ax.set_xlabel("Realized market beta")
    ax.set_ylabel("Average excess return (annual)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 15.1  The security market line is too flat",
                 _sub(sim, f"Ten portfolios sorted on past beta, {a['period']}"))
    plots.finish(fig, out / "fig15_1_sml.png")
    plt.close(fig)


def fig_cumulative(b, d, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.8)
    for key, label in SORTS.items():
        y = b["series"][key]["bab"]
        ax.plot(y.index.to_timestamp(), (1 + y).cumprod(), color=SORT_COLORS[key], linewidth=1.5,
                label=f"Sorted on {label.lower()}")
    if d.get("aqr") is not None:
        y = d["aqr"].loc[b["series"]["beta"]["bab"].index[0]:]
        ax.plot(y.index.to_timestamp(), (1 + y).cumprod(), color=SORT_COLORS["aqr"], linewidth=1.2, linestyle="--",
                label="AQR BAB factor (stock level)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 15.2  Betting against beta, built from portfolios",
                 _sub(sim, "Long the three lowest-risk deciles, short the three highest, each scaled to a beta of 1 "
                           "with ex-ante betas"))
    plots.finish(fig, out / "fig15_2_bab.png")
    plt.close(fig)


def fig_rolling(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    for key in ("beta", "var"):
        r = c["rolling"][key].dropna()
        ax.plot(r.index.to_timestamp(), r, color=SORT_COLORS[key], linewidth=1.5,
                label=f"BAB-style, sorted on {SORTS[key].lower()}")
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.set_ylabel("CAPM alpha, trailing 10 years (annual)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    plots.titles(fig, "Lab Figure 15.3  Has the low-risk premium shrunk?",
                 _sub(sim, "Rolling ten-year CAPM alpha of the beta-neutral long-short portfolios"))
    plots.finish(fig, out / "fig15_3_rolling.png")
    plt.close(fig)


def fig_crisis(c, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    t = c["crisis"].set_index("year")
    names = list(t.columns)
    colors = [plots.BLUE, plots.ORANGE, plots.AQUA, plots.INK_3]
    width = 0.8 / len(names)
    x = np.arange(len(t))
    for j, (n, col) in enumerate(zip(names, colors)):
        ax.bar(x + (j - len(names) / 2 + 0.5) * width, t[n], width=width, color=col, label=n)
    ax.set_xticks(x, [str(y) for y in t.index])
    ax.axhline(0, color=plots.INK_3, linewidth=1)
    plots.percent_axis(ax)
    ax.set_ylabel("Calendar-year return")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plots.titles(fig, "Lab Figure 15.4  Low risk in bad years",
                 _sub(sim, "2008 and 2020: equity crashes · 2022: stocks and bonds fell together as yields rose"))
    plots.finish(fig, out / "fig15_4_crisis.png")
    plt.close(fig)


def fig_investor(dd, out, sim):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(plots.SURFACE)
    for ax in (ax1, ax2):
        plots.style_axes(ax)
    dates = dd["lowvol"].index.to_timestamp()
    wl, wm = (1 + dd["lowvol"]).cumprod(), (1 + dd["market"]).cumprod()
    ax1.plot(dates, wm, color=plots.INK_3, linewidth=1.5, label="Market")
    ax1.plot(dates, wl, color=plots.AQUA, linewidth=1.5, label="Lowest-volatility quintile")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.plot(dates, wl / wm, color=plots.AQUA, linewidth=1.3)
    ax2.axhline(1, color=plots.INK_3, linewidth=1)
    ax2.set_ylabel("Relative wealth")
    t = dd["table"].iloc[0]
    plots.titles(fig, "Lab Figure 15.5  Low volatility as an investor experiences it",
                 _sub(sim, f"Excess returns, value-weighted · Sharpe {t['sharpe']:.2f} vs market "
                           f"{dd['table'].iloc[-1]['sharpe']:.2f} · behind the market in "
                           f"{t['share_years_behind_market']:.0%} of years"))
    plots.finish(fig, out / "fig15_5_investor.png")
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


PCT1, T2, F2 = _num("+.1%"), _num(".2f"), _num(".2f")


def write_report(out, d, a, b, c, dd, sim):
    md = ["# Lab 15 results: The Low-Risk Anomaly", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** whether low-risk stocks earn more than the CAPM predicts, and how much the answer depends on how the portfolios are built.", "", f"Sample: {a['period']}. Market premium {a['market_premium']:+.1%} a year.", "",
           "## Part A: The security market line", ""]
    for (key, w), v in a["sorts"].items():
        md += [f"**Sorted on {SORTS[key].lower()}, {'value' if w == 'vw' else 'equal'}-weighted** · slope of average "
               f"excess return on beta: {v['sml_slope']:+.1%} a year per unit of beta (CAPM: {a['market_premium']:+.1%})",
               "", _md(v["table"].rename(columns={"portfolio": "Decile", "mean_excess_annual": "Mean excess",
                                                    "vol_annual": "Volatility", "sharpe": "Sharpe", "beta": "Beta",
                                                    "capm_alpha": "CAPM α", "capm_t": "t", "ff4_alpha": "4-factor α",
                                                    "ff4_t": "t (4f)"}),
                       {"Mean excess": PCT1, "Volatility": _num(".1%"), "Sharpe": F2, "Beta": F2, "CAPM α": PCT1,
                        "t": T2, "4-factor α": PCT1, "t (4f)": T2}), ""]
    md += ["![Lab Figure 15.1](fig15_1_sml.png)", "", "## Part B: Long-short portfolios", "",
           _md(b["table"][["sort", "construction", "mean_annual", "sharpe", "capm_beta_mkt", "capm_alpha_annual",
                           "capm_alpha_t", "ff4_alpha_annual", "ff4_alpha_t", "max_drawdown", "months"]].rename(columns={
               "sort": "Sort", "construction": "Construction", "mean_annual": "Mean", "sharpe": "Sharpe",
               "capm_beta_mkt": "Beta", "capm_alpha_annual": "CAPM α", "capm_alpha_t": "t",
               "ff4_alpha_annual": "4-factor α", "ff4_alpha_t": "t (4f)", "max_drawdown": "Max drawdown",
               "months": "Months"}),
               {"Mean": PCT1, "Sharpe": F2, "Beta": _num("+.2f"), "CAPM α": PCT1, "t": T2, "4-factor α": PCT1,
                "t (4f)": T2, "Max drawdown": _num(".0%")}), "",
           "4-factor = market, size, value and momentum. Newey–West t-statistics.", "",
           "![Lab Figure 15.2](fig15_2_bab.png)", "", "## Part C: Stress tests", "", "**Equal vs value weights (BAB-style)**",
           "", _md(c["weights"].rename(columns={"sort": "Sort", "weighting": "Weighting", "mean_annual": "Mean",
                                                 "ff4_alpha": "4-factor α", "ff4_t": "t"}),
                   {"Mean": PCT1, "4-factor α": PCT1, "t": T2}), "",
           "**Subperiods (BAB-style, value-weighted)**", "",
           _md(c["subperiods"].rename(columns={"sort": "Sort", "period": "Period", "mean_annual": "Mean",
                                               "sharpe": "Sharpe", "capm_alpha": "CAPM α", "capm_t": "t",
                                               "ff4_alpha": "4-factor α", "ff4_t": "t (4f)"}),
               {"Mean": PCT1, "Sharpe": F2, "CAPM α": PCT1, "t": T2, "4-factor α": PCT1, "t (4f)": T2}), ""]
    if len(c["rates"]):
        md += ["**Interest-rate exposure** · regression on the market and the monthly change in the 10-year yield "
               "(decimal); a coefficient of −2 means a 1-point rise in yields costs 2% that month", "",
               _md(c["rates"].rename(columns={"sort": "Sort", "series": "Series", "dy10_coef": "Yield change coef.",
                                              "dy10_t": "t", "alpha_annual": "α", "alpha_t": "t (α)", "months": "Months"}),
                   {"Yield change coef.": _num("+.2f"), "t": T2, "α": PCT1, "t (α)": T2}), ""]
    else:
        md += ["> The 10-year yield (FRED GS10) was not loaded, so the interest-rate test was skipped.", ""]
    md += ["**Crisis years**", "", _md(c["crisis"], {k: _num("+.1%") for k in c["crisis"].columns if k != "year"}), "",
           "![Lab Figure 15.3](fig15_3_rolling.png)", "", "![Lab Figure 15.4](fig15_4_crisis.png)", "",
           "## Part D: The investor's view", "",
           _md(dd["table"].rename(columns={"portfolio": "Portfolio", "mean_excess_annual": "Mean excess",
                                           "vol_annual": "Volatility", "sharpe": "Sharpe", "beta": "Beta",
                                           "capm_alpha": "CAPM α", "capm_t": "t", "max_drawdown": "Max drawdown",
                                           "tracking_error": "Tracking error", "information_ratio": "Information ratio",
                                           "share_years_behind_market": "Years behind market"}),
               {"Mean excess": PCT1, "Volatility": _num(".1%"), "Sharpe": F2, "Beta": F2, "CAPM α": PCT1, "t": T2,
                "Max drawdown": _num(".0%"), "Tracking error": _num(".1%"), "Information ratio": _num("+.2f"),
                "Years behind market": _num(".0%")}), "",
           "![Lab Figure 15.5](fig15_5_investor.png)", "", "## Questions to think about", "",
           "1. Is the security market line flatter than the CAPM predicts? Does the answer depend on beta or volatility "
           "sorts, and on weighting?",
           "2. Do the low-risk alphas survive the four-factor model, recent decades and interest-rate controls?",
           "3. Would a benchmarked investor stick with a low-volatility portfolio? What does the tracking error imply?", ""]
    (out / "lab15_report.md").write_text("\n".join(md))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["french", "sim"], default="french")
    for f in ("ff3", "mom", "beta", "var", "resvar", "gs10", "aqr"):
        ap.add_argument(f"--{f}-file", dest=f"{f}_file", default=None)
    ap.add_argument("--no-gs10", action="store_true")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {k: getattr(args, f"{k}_file") for k in ("ff3", "mom", "beta", "var", "resvar", "gs10", "aqr")}
    files["no_gs10"] = args.no_gs10
    try:
        d = load(args.source, files, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    sim = d["simulated"]
    a = part_a(d)
    b = part_b(d)
    c = part_c(d, b)
    dd = part_d(d)
    fig_sml(a, out, sim)
    fig_cumulative(b, d, out, sim)
    fig_rolling(c, out, sim)
    fig_crisis(c, out, sim)
    fig_investor(dd, out, sim)
    for (key, w), v in a["sorts"].items():
        v["table"].to_csv(out / f"table_sml_{key}_{w}.csv", index=False)
    b["table"].to_csv(out / "table_long_short.csv", index=False)
    for k in ("weights", "subperiods", "rates", "crisis"):
        c[k].to_csv(out / f"table_{k}.csv", index=False)
    dd["table"].to_csv(out / "table_investor.csv", index=False)
    write_report(out, d, a, b, c, dd, sim)
    summary = {"simulated": sim, "market_premium": a["market_premium"],
               "sml_slopes": {f"{k}_{w}": v["sml_slope"] for (k, w), v in a["sorts"].items()},
               "long_short": b["table"].to_dict(orient="records"), "subperiods": c["subperiods"].to_dict(orient="records"),
               "rates": c["rates"].to_dict(orient="records"), "investor": dd["table"].to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Market premium {a['market_premium']:+.1%}; SML slope (beta, VW) {a['sorts'][('beta', 'vw')]['sml_slope']:+.1%}")
    print(b["table"][["sort", "construction", "mean_annual", "capm_alpha_annual", "capm_alpha_t", "ff4_alpha_annual",
                      "ff4_alpha_t"]].round(3).to_string(index=False))
    print(f"Report: {out / 'lab15_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
