"""Lab 5: Do Simple Trading Rules Beat Buy-and-Hold?

Chapter 5 of *Financial Engineering in Practice*.

WHAT YOU WILL EXPLORE
  How trend rules perform against buy-and-hold, and how much of their
  apparent success remains after correcting for the number of rules tried.

WHAT THE LAB DOES
  Part A  The 26 rules of Brock, Lakonishok and LeBaron (1992) on the daily
          U.S. market: buy and sell days, mean returns after each signal, the
          buy-sell difference and its t-statistic, 1926-1986 and 1987 on.
  Part B  The search correction of Sullivan, Timmermann and White (1999): a
          universe of over 1,600 moving-average and breakout rules against
          buy-and-hold, with White's Reality Check and Hansen's SPA test
          (stationary bootstrap, mean block 10 days), for both periods and in
          rolling ten-year windows.
  Part C  Time-series momentum across asset classes: AQR's factors
          (Moskowitz, Ooi and Pedersen 2012), before and after 2009.
  Part D  The smile: quarterly trend returns on the market and its square,
          and the ten worst quarters for U.S. stocks.
  Part E  Asset by asset: about a dozen assets from free data (U.S. stocks, a
          10-year Treasury bond, nine currencies, crude oil) with the tests of
          Huang, Li, Wang and Zhou (2020).

DATA (all free)
  Kenneth R. French Data Library: Fama/French 3 Factors, daily and monthly.
  AQR: Time Series Momentum: Factors, Monthly (xlsx).
  FRED: 10-year Treasury yield (GS10), H.10 exchange rates and three-month
  interbank rates for nine currencies, WTI crude oil (MCOILWTICO).
  Optional: the daily Dow Jones index from Stooq (free key required).

USAGE
  python lab05.py
  python lab05.py --source sim            # fake data, to test the code

OUTPUT
  report/lab05_report.md, five figures, CSV tables and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from fintest import plots
from fintest.corr import bond_returns_from_yields
from fintest.credit import monthly_fred
from fintest.data import DEFAULT_CACHE, DataUnavailable, load_french_factors
from fintest.fx import excess_returns, load_fx, simulated_fx
from fintest.search import (
    breakout_signal,
    crossings,
    ma_signal,
    reality_check,
    rule_positions,
    rule_universe,
)
from fintest.stats import ols_newey_west

ROOT = Path(__file__).resolve().parent
SPLIT = "1987-01-01"  # Brock, Lakonishok and LeBaron's sample ends in 1986
AQR_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
           "Time-Series-Momentum-Factors-Monthly.xlsx")
AQR_PAGE = "https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_stooq(path) -> pd.Series:
    """Daily closing prices from a Stooq CSV (Date, Open, High, Low, Close, Volume)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    s = pd.Series(pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[cols["date"]]), name="close").dropna()
    return s.sort_index()


def parse_aqr_tsmom(raw: pd.DataFrame) -> pd.DataFrame:
    """AQR 'Time Series Momentum: Factors, Monthly': find the header row containing 'TSMOM', parse dates in the
    first column and returns (decimals) in the TSMOM columns. Works on the xlsx sheet or a CSV export."""
    header_row = None
    for i in range(min(len(raw), 60)):
        cells = [str(x).strip().upper() for x in raw.iloc[i].tolist()]
        if any(c.startswith("TSMOM") for c in cells):
            header_row = i
            break
    if header_row is None:
        raise DataUnavailable("No 'TSMOM' header found in the AQR file.")
    names = [str(x).strip() for x in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:].copy()
    body.columns = names
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    keep = dates.notna()
    cols = [c for c in names if c.upper().startswith("TSMOM")]
    out = body.loc[keep, cols].apply(pd.to_numeric, errors="coerce")
    out.index = pd.PeriodIndex(dates[keep], freq="M")
    out.index.name = "date"
    out = out.dropna(how="all")
    if out.abs().stack().median() > 0.5:  # given in percent
        out = out / 100
    rename = {c: c.replace("^", "_").replace(" ", "") for c in cols}
    return out.rename(columns=rename)


def load_aqr(path=None, cache_dir: Path = DEFAULT_CACHE) -> pd.DataFrame:
    if path is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "Time-Series-Momentum-Factors-Monthly.xlsx"
        if not path.exists():
            try:
                req = urllib.request.Request(AQR_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    path.write_bytes(resp.read())
            except Exception as e:  # noqa: BLE001
                raise DataUnavailable(f"Could not download the AQR file ({e}). Download 'Time Series Momentum: "
                                      f"Factors, Monthly' from {AQR_PAGE} and pass it with --aqr-file.") from e
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        for raw in sheets.values():
            try:
                return parse_aqr_tsmom(raw)
            except DataUnavailable:
                continue
        raise DataUnavailable(f"No TSMOM table found in {path}.")
    return parse_aqr_tsmom(pd.read_csv(path, header=None))


def load(source: str, files: dict, sim_seed: int = 0) -> dict:
    if source == "sim":
        return simulated(sim_seed)
    daily = load_french_factors("ff3_daily", path=files.get("ff3_daily"))
    monthly = load_french_factors("ff3_monthly", path=files.get("ff3"))
    out = {"simulated": False, "notes": []}
    r = (daily["Mkt-RF"] + daily["RF"])
    out["index_name"] = "U.S. market (French daily, total return)"
    out["logp"] = np.log1p(r).cumsum()
    out["rf_daily"] = daily["RF"]
    out["mkt_daily"] = r
    if files.get("dow"):
        close = read_stooq(files["dow"])
        out["logp_dow"] = np.log(close)
    out["mkt_m"], out["rf_m"] = monthly["Mkt-RF"], monthly["RF"]
    try:
        out["aqr"] = load_aqr(files.get("aqr"))
    except (DataUnavailable, ImportError, ValueError) as e:
        out["aqr"] = None
        out["notes"].append(f"Parts C and D skipped: {str(e).splitlines()[0]}")
    out["assets"] = load_assets(files, out["mkt_m"], out["rf_m"], out["notes"])
    return out


def load_assets(files: dict, mkt: pd.Series, rf: pd.Series, notes: list) -> pd.DataFrame:
    """Monthly excess returns of about a dozen assets built from free data."""
    cols = {"US stocks": mkt}
    try:
        gs10 = monthly_fred("GS10", files.get("GS10"))
        cols["10y Treasury"] = bond_returns_from_yields(gs10) - rf.reindex(gs10.index)
    except DataUnavailable as e:
        notes.append(f"Bond skipped: {str(e).splitlines()[0]}")
    try:
        fx = load_fx(files.get("fx") or {})
        rx = excess_returns(fx["spot"], fx["rate"], fx["us"])["rx"]
        for c in rx.columns:
            cols[c] = rx[c]
        notes += [f"Currency skipped: {m}" for m in fx["missing"]]
    except DataUnavailable as e:
        notes.append(f"Currencies skipped: {str(e).splitlines()[0]}")
    try:
        oil = monthly_fred("MCOILWTICO", files.get("MCOILWTICO"))
        cols["Crude oil"] = np.log(oil).diff() - rf.reindex(oil.index)
        notes.append("Crude oil uses the change in the WTI spot price less the bill rate, a proxy for a futures return.")
    except DataUnavailable as e:
        notes.append(f"Oil skipped: {str(e).splitlines()[0]}")
    return pd.DataFrame(cols)


def simulated(seed: int = 0, n_days: int = 25000) -> dict:
    """Fake data: a daily market with mild trends, a trend factor with a smile, and assets with weak time-series
    momentum. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1926-07-01", periods=n_days)
    vol = np.empty(n_days)
    r = np.empty(n_days)
    v, drift = 1e-4, 0.0
    for t in range(n_days):
        drift = 0.995 * drift + 0.00004 * rng.standard_normal()  # slowly moving drift: something for trend rules
        z = rng.standard_t(6) / math.sqrt(1.5)
        r[t] = 0.0003 + drift + math.sqrt(v) * z
        v = 1.5e-6 + 0.08 * (r[t] - 0.0003) ** 2 + 0.90 * v
        vol[t] = v
    mkt_d = pd.Series(r, idx)
    rf_d = pd.Series(0.00011, idx)
    months = pd.period_range("1926-07", idx[-1].strftime("%Y-%m"), freq="M")
    mkt_m = (np.log1p(mkt_d).groupby(idx.to_period("M")).sum()).pipe(np.expm1).reindex(months) - 0.0023
    rf_m = pd.Series(0.0023, months)
    # Trend factor: a smile in the market (convex), plus noise
    m = mkt_m.loc["1985-01":]
    tsmom = 0.006 + 0.0 * m + 3.0 * (m ** 2 - (m ** 2).mean()) + 0.03 * rng.standard_normal(len(m))
    aqr = pd.DataFrame({"TSMOM": tsmom, "TSMOM_EQ": tsmom * 0.5 + 0.02 * rng.standard_normal(len(m)),
                        "TSMOM_FI": 0.003 + 0.02 * rng.standard_normal(len(m)),
                        "TSMOM_FX": 0.002 + 0.02 * rng.standard_normal(len(m)),
                        "TSMOM_CM": 0.004 + 0.03 * rng.standard_normal(len(m))}, index=m.index)
    fx = simulated_fx(n_months=600, seed=seed, start="1976-01")
    assets = {"US stocks": mkt_m}
    rx = excess_returns(fx["spot"], fx["rate"], fx["us"])["rx"]
    for c in rx.columns:
        assets[c] = rx[c]
    T = len(months)
    for name, sd in (("10y Treasury", 0.02), ("Crude oil", 0.09)):
        mu = np.zeros(T)
        e = sd * rng.standard_normal(T)
        x = np.zeros(T)
        for t in range(1, T):
            mu[t] = 0.9 * mu[t - 1] + 0.1 * e[t - 1]  # weak persistence in expected returns
            x[t] = 0.002 + 0.3 * mu[t] + e[t]
        assets[name] = pd.Series(x, months)
    return {"simulated": True, "notes": [], "index_name": "Simulated daily market", "logp": np.log1p(mkt_d).cumsum(),
            "rf_daily": rf_d, "mkt_daily": mkt_d, "mkt_m": mkt_m, "rf_m": rf_m, "aqr": aqr,
            "assets": pd.DataFrame(assets)}


# ---------------------------------------------------------------------------
# Part A: the 26 rules of Brock, Lakonishok and LeBaron (1992)
# ---------------------------------------------------------------------------

BLL_MA = [(1, 50), (1, 150), (5, 150), (1, 200), (2, 200)]
BLL_TRB = [50, 150, 200]


def bll_rules() -> list[tuple]:
    rules = [("VMA", s, l, b) for s, l in BLL_MA for b in (0.0, 0.01)]
    rules += [("FMA", s, l, b) for s, l in BLL_MA for b in (0.0, 0.01)]
    rules += [("TRB", 1, n, b) for n in BLL_TRB for b in (0.0, 0.01)]
    return rules


def _two_sample_t(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def thin_events(ev: np.ndarray, hold: int) -> np.ndarray:
    """Keep a signal only if no kept signal occurred in the previous `hold` days, as Brock et al. ignore
    signals during a holding period."""
    out = np.zeros_like(ev)
    last = -10 ** 9
    for t in np.flatnonzero(ev):
        if t - last >= hold:
            out[t] = ev[t]
            last = t
    return out


def bll_table(logp: pd.Series, hold: int = 10) -> pd.DataFrame:
    """Buy and sell statistics for the 26 rules. Variable-length rules (VMA) report mean daily log returns on buy
    and sell days; fixed-length rules (FMA, TRB) report mean 10-day log returns after each signal."""
    lp = logp.to_numpy()
    r_next = np.append(np.diff(lp), np.nan)  # return of day t+1, decided at the close of t
    rows = []
    for kind, s, l, b in bll_rules():
        if kind == "VMA":
            st = ma_signal(lp, s, l, b)
            buy, sell = r_next[(st == 1) & np.isfinite(r_next)], r_next[(st == -1) & np.isfinite(r_next)]
            unit = "daily"
        else:
            ev = crossings(ma_signal(lp, s, l, b)) if kind == "FMA" else breakout_signal(lp, l, b)
            ev = thin_events(ev, hold)
            fwd = pd.Series(lp).shift(-hold).to_numpy() - lp
            buy, sell = fwd[(ev == 1) & np.isfinite(fwd)], fwd[(ev == -1) & np.isfinite(fwd)]
            unit = f"{hold}-day"
        name = (f"{kind}({s},{l},{b:.0%})" if kind != "TRB" else f"TRB({l},{b:.0%})")
        rows.append({"rule": name, "type": kind, "unit": unit, "n_buy": len(buy), "n_sell": len(sell),
                     "buy_mean": buy.mean() if len(buy) else np.nan, "sell_mean": sell.mean() if len(sell) else np.nan,
                     "buy_minus_sell": (buy.mean() - sell.mean()) if len(buy) and len(sell) else np.nan,
                     "t": _two_sample_t(buy, sell)})
    return pd.DataFrame(rows)


def part_a(d: dict) -> dict:
    lp = d["logp"]
    split = pd.Timestamp(SPLIT)
    periods = {"1926-1986": lp[lp.index < split], f"1987-{lp.index[-1].year}": lp[lp.index >= split]}
    # Each subperiod keeps the earlier prices needed for its first moving averages
    out = {}
    for name, sub in periods.items():
        start = sub.index[0]
        ext = lp[lp.index >= start - pd.Timedelta(days=400)] if name.startswith("1987") else sub
        t = bll_table(ext)
        out[name] = t
    combined = out[list(out)[0]][["rule", "type", "buy_minus_sell", "t"]].merge(
        out[list(out)[1]][["rule", "buy_minus_sell", "t"]], on="rule", suffixes=("_early", "_late"))
    daily = np.diff(lp.to_numpy())
    return {"tables": out, "combined": combined, "unconditional_daily": float(np.nanmean(daily)),
            "periods": list(out)}


# ---------------------------------------------------------------------------
# Part B: the search correction
# ---------------------------------------------------------------------------


def strategy_minus_benchmark(d: dict, pos: np.ndarray, cost: float = 0.0) -> pd.DataFrame:
    """Daily return of each long-or-cash rule minus buy-and-hold. pos[t] is decided at the close of t."""
    mkt = d["mkt_daily"].to_numpy()
    rf = d["rf_daily"].reindex(d["mkt_daily"].index).fillna(0).to_numpy()
    P = np.vstack([np.zeros((1, pos.shape[1]), dtype=pos.dtype), pos[:-1]])  # position held during day t
    strat = P * mkt[:, None] + (1 - P) * rf[:, None]
    if cost:
        strat -= cost * np.abs(np.diff(np.vstack([P[:1], P]), axis=0))
    return pd.DataFrame(strat - mkt[:, None], index=d["mkt_daily"].index)


def part_b(d: dict, n_boot: int = 1000, block: float = 10, window_years: int = 10, window_boot: int = 200,
           cost: float = 0.0, seed: int = 0, universe=None) -> dict:
    rules = universe or rule_universe()
    lp = d["logp"]
    pos = rule_positions(lp.to_numpy(), rules)
    dm = strategy_minus_benchmark(d, pos, cost)
    burn = 300  # skip the start, where the longest averages are not yet defined
    dm = dm.iloc[burn:]
    split = pd.Timestamp(SPLIT)
    rows = []
    for name, sub in (("1926-1986", dm[dm.index < split]), (f"1987-{dm.index[-1].year}", dm[dm.index >= split])):
        rc = reality_check(sub.to_numpy(), n_boot=n_boot, mean_block=block, seed=seed)
        rows.append({"period": name, "rules": rc["n_rules"], "days": rc["n_obs"], "best_rule": rules[rc["best"]].name,
                     "best_excess_annual": rc["best_mean"] * 252, "best_t": rc["best_t"],
                     "p_nominal": rc["p_nominal"], "rules_nominal_5pct": rc["n_nominal_5pct"],
                     "p_reality_check": rc["p_reality_check"], "p_spa": rc["p_spa"]})
    # Rolling ten-year windows, one per year
    years = sorted(set(dm.index.year))
    roll = []
    for y in years:
        if y + window_years - 1 > years[-1]:
            break
        w = dm[(dm.index.year >= y) & (dm.index.year < y + window_years)]
        rc = reality_check(w.to_numpy(), n_boot=window_boot, mean_block=block, seed=seed + y)
        roll.append({"end_year": y + window_years - 1, "p_nominal": rc["p_nominal"],
                     "p_reality_check": rc["p_reality_check"], "p_spa": rc["p_spa"],
                     "best_rule": rules[rc["best"]].name, "best_excess_annual": rc["best_mean"] * 252})
    return {"table": pd.DataFrame(rows), "rolling": pd.DataFrame(roll), "n_rules": len(rules), "cost": cost,
            "block": block, "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Parts C and D: AQR time-series momentum factors
# ---------------------------------------------------------------------------


def _stats(r: pd.Series, mkt: pd.Series) -> dict:
    r = r.dropna()
    m = mkt.reindex(r.index)
    ok = m.notna()
    reg = ols_newey_west(r[ok].to_numpy()[:, None], m[ok].to_numpy(), lags=6)
    return {"months": len(r), "mean_annual": 12 * r.mean(), "sharpe": math.sqrt(12) * r.mean() / r.std(ddof=1),
            "t_mean": r.mean() / (r.std(ddof=1) / math.sqrt(len(r))),
            "alpha_annual": 12 * float(reg["coef"][0, 0]), "alpha_t": float(reg["t"][0, 0]),
            "beta": float(reg["coef"][1, 0])}


def part_c(d: dict, split: str = "2009-01") -> dict | None:
    aqr = d["aqr"]
    if aqr is None:
        return None
    rows = []
    s = pd.Period(split, "M")
    for c in aqr.columns:
        for name, sub in (("Full sample", aqr[c]), (f"to {s - 1}", aqr[c][aqr.index < s]),
                          (f"{s} on", aqr[c][aqr.index >= s])):
            if sub.dropna().shape[0] > 24:
                st = _stats(sub, d["mkt_m"])
                st.update(factor=c, period=name)
                rows.append(st)
    t = pd.DataFrame(rows)
    return {"table": t[["factor", "period", "months", "mean_annual", "sharpe", "t_mean", "alpha_annual", "alpha_t",
                        "beta"]]}


def _quarterly(r: pd.Series) -> pd.Series:
    q = np.log1p(r.dropna()).groupby(r.dropna().index.asfreq("Q")).sum()
    counts = r.dropna().groupby(r.dropna().index.asfreq("Q")).size()
    return np.expm1(q[counts == 3])


def part_d(d: dict, factor: str = "TSMOM", n_worst: int = 10) -> dict | None:
    aqr = d["aqr"]
    if aqr is None:
        return None
    col = factor if factor in aqr.columns else aqr.columns[0]
    tq = _quarterly(aqr[col])
    mq = _quarterly(d["mkt_m"]).reindex(tq.index)
    ok = mq.notna()
    x = np.column_stack([mq[ok], mq[ok] ** 2])
    reg = ols_newey_west(tq[ok].to_numpy()[:, None], x, lags=2)
    worst = pd.DataFrame({"market": mq[ok], "trend": tq[ok]}).nsmallest(n_worst, "market")
    reg_t = pd.DataFrame({"term": ["intercept", "market", "market squared"], "estimate": reg["coef"][:, 0],
                          "t": reg["t"][:, 0]})
    return {"factor": col, "regression": reg_t, "worst": worst, "tq": tq[ok], "mq": mq[ok],
            "worst_trend_mean": float(worst["trend"].mean()), "worst_market_mean": float(worst["market"].mean())}


# ---------------------------------------------------------------------------
# Part E: asset by asset, after Huang, Li, Wang and Zhou (2020)
# ---------------------------------------------------------------------------


def _scaled(R: pd.DataFrame, lookback: int = 12, vol_window: int = 36):
    sig = R.rolling(vol_window, min_periods=24).std().shift(1)  # known before month t
    past = R.rolling(lookback, min_periods=lookback).sum().shift(1)
    y = R / sig
    x = past / sig
    return y, x, sig


def pooled_slope(R: pd.DataFrame, lookback: int = 12) -> float:
    y, x, _ = _scaled(R, lookback)
    st = pd.concat([y.stack(), x.stack()], axis=1, keys=["y", "x"]).dropna()
    xd = st.x - st.x.mean()
    return float((xd * (st.y - st.y.mean())).sum() / (xd ** 2).sum())


def part_e(d: dict, lookback: int = 12, n_boot: int = 500, block: float = 12, target: float = 0.40,
           min_history: int = 60, seed: int = 0) -> dict | None:
    R = d["assets"].dropna(how="all")
    R = R.loc[:, R.notna().sum() > 120]
    if R.shape[1] < 2:
        return None
    y, x, sig = _scaled(R, lookback)
    rows = []
    for c in R.columns:
        ok = y[c].notna() & x[c].notna()
        if ok.sum() < 60:
            continue
        reg = ols_newey_west(y[c][ok].to_numpy()[:, None], x[c][ok].to_numpy(), lags=12)
        rows.append({"asset": c, "months": int(ok.sum()), "slope": float(reg["coef"][1, 0]),
                     "t": float(reg["t"][1, 0])})
    sep = pd.DataFrame(rows)
    # Pooled slope, with a bootstrap p-value under the null of no time-series predictability:
    # resample months in blocks (keeping the cross-section together) and recompute everything.
    b = pooled_slope(R, lookback)
    rng = np.random.default_rng(seed)
    from fintest.search import stationary_bootstrap_matrix
    Rv = R.to_numpy()
    idx = stationary_bootstrap_matrix(len(R), n_boot, block, rng)
    bs = np.array([pooled_slope(pd.DataFrame(Rv[i], columns=R.columns), lookback) for i in idx])
    bs = bs - bs.mean()  # centre under the null
    p_pooled = float((np.abs(bs) >= abs(b)).mean())
    # Time-series momentum against time-series history, each asset scaled to `target` annual volatility
    w = target / (sig * math.sqrt(12))
    tsm = (np.sign(R.rolling(lookback, min_periods=lookback).sum().shift(1)) * w * R).mean(axis=1, skipna=True)
    hist = R.expanding(min_periods=min_history).mean().shift(1)
    tsh = (np.sign(hist) * w * R).mean(axis=1, skipna=True)
    both = pd.DataFrame({"TSM": tsm, "TSH": tsh}).dropna()
    both = both[(both != 0).all(axis=1)]
    st = {c: {"mean_annual": 12 * both[c].mean(), "sharpe": math.sqrt(12) * both[c].mean() / both[c].std(ddof=1)}
          for c in both}
    reg = ols_newey_west(both["TSM"].to_numpy()[:, None], both["TSH"].to_numpy(), lags=6)
    return {"separate": sep, "pooled_slope": b, "p_pooled": p_pooled, "n_boot": n_boot, "strategies": both,
            "strategy_stats": st, "corr": float(both.corr().iloc[0, 1]),
            "alpha_tsm_on_tsh_annual": 12 * float(reg["coef"][0, 0]), "alpha_t": float(reg["t"][0, 0]),
            "assets": list(R.columns)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _sub(sim, text):
    return text + ("  ·  SIMULATED DATA: tests the code only" if sim else "")


def fig_rules(a, out, sim):
    fig, ax, plt = plots.new_figure(9.5, 5.0)
    c = a["combined"]
    x = np.arange(len(c))
    w = 0.4
    vma = c["type"] == "VMA"
    for col, lab, colr, off in (("buy_minus_sell_early", a["periods"][0], plots.BLUE, -w / 2),
                                ("buy_minus_sell_late", a["periods"][1], plots.ORANGE, w / 2)):
        scale = np.where(vma, 252, 252 / 10)  # annualize daily and 10-day differences
        ax.bar(x + off, c[col] * scale, w, color=colr, label=lab)
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_xticks(x, c["rule"], rotation=90, fontsize=7.5)
    plots.percent_axis(ax)
    ax.set_ylabel("Buy minus sell, annualized")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    plots.titles(fig, "Lab Figure 5.1  The 26 rules of Brock, Lakonishok and LeBaron, before and after 1987",
                 _sub(sim, "Mean log return after buy signals minus after sell signals · VMA: daily returns on buy and "
                           "sell days × 252; FMA and TRB: 10-day returns after each signal × 25.2"))
    plots.finish(fig, out / "fig5_1_rules.png")
    plt.close(fig)


def fig_rolling(b, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    r = b["rolling"]
    ax.plot(r.end_year, r.p_nominal, color=plots.INK_3, linewidth=1.2, label="Best rule alone (nominal)")
    ax.plot(r.end_year, r.p_reality_check, color=plots.BLUE, linewidth=1.8, label="Reality Check")
    ax.plot(r.end_year, r.p_spa, color=plots.ORANGE, linewidth=1.8, label="SPA")
    ax.axhline(0.05, color=plots.RED, linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Last year of the ten-year window")
    ax.set_ylabel("p-value")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    plots.titles(fig, "Lab Figure 5.2  Does the best rule beat buy-and-hold once the search is counted?",
                 _sub(sim, f"{b['n_rules']:,} rules against buy-and-hold in rolling ten-year windows · stationary "
                           f"bootstrap, mean block {b['block']:g} days · dashed line: 0.05"))
    plots.finish(fig, out / "fig5_2_rolling_pvalues.png")
    plt.close(fig)


def fig_smile(dd, out, sim):
    fig, ax, plt = plots.new_figure(8.0, 5.0)
    ax.scatter(dd["mq"], dd["tq"], s=14, color=plots.BLUE, alpha=0.7)
    g = np.linspace(dd["mq"].min(), dd["mq"].max(), 100)
    co = dd["regression"]["estimate"].to_numpy()
    ax.plot(g, co[0] + co[1] * g + co[2] * g ** 2, color=plots.ORANGE, linewidth=2)
    ax.axhline(0, color=plots.INK_3, linewidth=0.8)
    ax.axvline(0, color=plots.INK_3, linewidth=0.8)
    plots.percent_axis(ax, "x")
    plots.percent_axis(ax, "y")
    ax.set_xlabel("U.S. market excess return in the quarter")
    ax.set_ylabel(f"{dd['factor']} return in the quarter")
    plots.titles(fig, "Lab Figure 5.3  The trend-following smile",
                 _sub(sim, f"Quarterly returns · fitted line: trend = a + b·market + c·market² (c = "
                           f"{co[2]:+.2f}, t = {dd['regression']['t'].iloc[2]:.1f})"))
    plots.finish(fig, out / "fig5_3_smile.png")
    plt.close(fig)


def fig_assets(e, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    s = e["separate"].sort_values("t")
    colors = [plots.BLUE if v > 1.96 else plots.INK_3 for v in s.t]
    ax.barh(s.asset, s.t, color=colors)
    ax.axvline(1.96, color=plots.RED, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=plots.INK_3, linewidth=0.8)
    ax.set_xlabel("t-statistic: this month's scaled return on the past 12 months' scaled return")
    plots.titles(fig, "Lab Figure 5.4  Time-series predictability, asset by asset",
                 _sub(sim, f"Separate regressions (Newey–West t) · pooled slope {e['pooled_slope']:+.3f}, "
                           f"bootstrap p = {e['p_pooled']:.2f} · blue: t > 1.96"))
    plots.finish(fig, out / "fig5_4_assets.png")
    plt.close(fig)


def fig_tsm_tsh(e, out, sim):
    fig, ax, plt = plots.new_figure(9.0, 4.6)
    s = e["strategies"]
    dates = s.index.to_timestamp()
    for c, colr in (("TSM", plots.BLUE), ("TSH", plots.ORANGE)):
        ax.plot(dates, (1 + s[c]).cumprod(), color=colr, linewidth=1.6,
                label=f"{c} (Sharpe {e['strategy_stats'][c]['sharpe']:.2f})")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plots.titles(fig, "Lab Figure 5.5  Time-series momentum against time-series history",
                 _sub(sim, f"TSM: sign of each asset's past 12-month return; TSH: sign of its historical average · "
                           f"alpha of TSM on TSH {e['alpha_tsm_on_tsh_annual']:+.1%} a year (t = {e['alpha_t']:.2f})"))
    plots.finish(fig, out / "fig5_5_tsm_tsh.png")
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


def write_report(out, d, a, b, c, dd, e, sim):
    md = ["# Lab 5 results: Do Simple Trading Rules Beat Buy-and-Hold?", ""]
    if sim:
        md += ["> **Simulated data.** These results test that the code runs; they say nothing about real markets.", ""]
    md += ["**What you will explore:** how trend rules perform against buy-and-hold, and how much of their apparent success remains after correcting for the number of rules tried.",
           ""]
    md += [f"> {n}" for n in d.get("notes", [])] + ([""] if d.get("notes") else [])
    md += ["## Part A: The 26 rules of Brock, Lakonishok and LeBaron (1992)", "",
           f"Index: {d['index_name']}. Brock and coauthors used the Dow Jones price index, without dividends; pass "
           "`--dow-file` to use it. Unconditional mean daily log return: "
           f"{a['unconditional_daily']:.4%}.", ""]
    fmt = {"buy_mean": _num("+.4%"), "sell_mean": _num("+.4%"), "buy_minus_sell": _num("+.4%"), "t": _num(".2f")}
    for name, t in a["tables"].items():
        md += [f"**{name}**", "", _md(t, fmt), ""]
    md += ["![Lab Figure 5.1](fig5_1_rules.png)", ""]
    md += ["## Part B: The search correction", "",
           f"{b['n_rules']:,} rules (moving-average crossovers and breakouts, with bands and holding periods), each "
           "long the market or in Treasury bills, against buy-and-hold; one-way cost per switch "
           f"{b['cost']:.2%}; {b['n_boot']} bootstrap samples.", "",
           _md(b["table"], {"best_excess_annual": _num("+.2%"), "best_t": _num(".2f"), "p_nominal": _num(".4f"),
                            "p_reality_check": _num(".3f"), "p_spa": _num(".3f")}), "",
           "![Lab Figure 5.2](fig5_2_rolling_pvalues.png)", ""]
    md += ["## Part C: Time-series momentum across asset classes (AQR factors)", ""]
    if c is None:
        md += ["Skipped: the AQR file was not available (see the note above).", ""]
    else:
        md += [_md(c["table"], {"mean_annual": _num("+.1%"), "sharpe": _num(".2f"), "t_mean": _num(".2f"),
                                "alpha_annual": _num("+.1%"), "alpha_t": _num(".2f"), "beta": _num("+.2f")}), ""]
    md += ["## Part D: The smile", ""]
    if dd is None:
        md += ["Skipped: the AQR file was not available.", ""]
    else:
        md += [_md(dd["regression"], {"estimate": _num("+.4f"), "t": _num(".2f")}), "",
               f"**The ten worst quarters for U.S. stocks:** market {dd['worst_market_mean']:+.1%}, "
               f"{dd['factor']} {dd['worst_trend_mean']:+.1%} on average.", "",
               _md(dd["worst"].reset_index().rename(columns={"date": "quarter", "index": "quarter"}),
                   {"market": _num("+.1%"), "trend": _num("+.1%")}), "",
               "![Lab Figure 5.3](fig5_3_smile.png)", ""]
    md += ["## Part E: Asset by asset (Huang, Li, Wang and Zhou 2020)", ""]
    if e is None:
        md += ["Skipped: fewer than two assets could be loaded.", ""]
    else:
        md += [f"Assets: {', '.join(e['assets'])}.", "",
               _md(e["separate"], {"slope": _num("+.3f"), "t": _num(".2f")}), "",
               f"Pooled slope {e['pooled_slope']:+.3f}; bootstrap p-value under the null of no predictability "
               f"{e['p_pooled']:.3f} ({e['n_boot']} samples).", "",
               f"Time-series momentum (TSM): {e['strategy_stats']['TSM']['mean_annual']:+.1%} a year, Sharpe "
               f"{e['strategy_stats']['TSM']['sharpe']:.2f}. Time-series history (TSH): "
               f"{e['strategy_stats']['TSH']['mean_annual']:+.1%}, Sharpe {e['strategy_stats']['TSH']['sharpe']:.2f}. "
               f"Correlation {e['corr']:.2f}; alpha of TSM on TSH {e['alpha_tsm_on_tsh_annual']:+.1%} a year "
               f"(t = {e['alpha_t']:.2f}).", "",
               "![Lab Figure 5.4](fig5_4_assets.png)", "", "![Lab Figure 5.5](fig5_5_tsm_tsh.png)", ""]
    md += ["## Questions to think about", "",
           "1. Did the 26 rules beat buy-and-hold before 1987? After?",
           "2. Once the search over many rules is counted, is the best rule better than luck?",
           "3. Does trend following earn a premium across asset classes, and is it time-series momentum or just "
           "each asset's average return?", ""]
    (out / "lab05_report.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["download", "sim"], default="download")
    ap.add_argument("--ff3-daily-file", default=None, help="Fama/French 3 Factors [Daily] zip or CSV")
    ap.add_argument("--ff3-file", default=None, help="Fama/French 3 Factors (monthly) zip or CSV")
    ap.add_argument("--aqr-file", default=None, help="AQR Time Series Momentum: Factors, Monthly (xlsx or CSV)")
    ap.add_argument("--dow-file", default=None, help="optional: Stooq daily Dow Jones CSV (^dji) for Part A")
    ap.add_argument("--fred-dir", default=None, help="folder with FRED CSVs named <SERIES>.csv")
    ap.add_argument("--n-boot", type=int, default=1000, help="Part B bootstrap samples")
    ap.add_argument("--block", type=float, default=10, help="Part B mean block length (days)")
    ap.add_argument("--cost", type=float, default=0.0, help="Part B one-way cost per switch (decimal)")
    ap.add_argument("--fast", action="store_true", help="a smaller rule universe and fewer bootstrap samples")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    return ap.parse_args(argv)


def _fred_files(folder):
    if not folder:
        return {}
    from fintest.fx import CURRENCIES, US_RATE
    f = Path(folder)
    ids = ["GS10", "MCOILWTICO", US_RATE] + [x for v in CURRENCIES.values() for x in (v[0], v[2])]
    found = {i: f / f"{i}.csv" for i in ids if (f / f"{i}.csv").exists()}
    out = {k: v for k, v in found.items() if k in ("GS10", "MCOILWTICO")}
    out["fx"] = {k: v for k, v in found.items() if k not in ("GS10", "MCOILWTICO")}
    return out


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"ff3_daily": args.ff3_daily_file, "ff3": args.ff3_file, "aqr": args.aqr_file, "dow": args.dow_file,
             **_fred_files(args.fred_dir)}
    try:
        d = load(args.source, files, args.sim_seed)
    except DataUnavailable as err:
        raise SystemExit(str(err)) from None
    if d.get("logp_dow") is not None:
        d["logp"], d["index_name"] = d["logp_dow"], "Dow Jones Industrial Average (Stooq, price index)"
    sim = d["simulated"]
    a = part_a(d)
    universe = rule_universe(shorts=(1, 5, 20), longs=(50, 150, 200), breakouts=(20, 50, 200),
                             bands=(0.0, 0.01), holds=(0, 10)) if args.fast else None
    b = part_b(d, n_boot=200 if args.fast else args.n_boot, block=args.block, cost=args.cost, seed=args.seed,
               window_boot=100 if args.fast else 200, universe=universe)
    c, dd = part_c(d), part_d(d)
    e = part_e(d, n_boot=100 if args.fast else 500, seed=args.seed)
    fig_rules(a, out, sim)
    fig_rolling(b, out, sim)
    if dd is not None:
        fig_smile(dd, out, sim)
    if e is not None:
        fig_assets(e, out, sim)
        fig_tsm_tsh(e, out, sim)
    for name, t in a["tables"].items():
        t.to_csv(out / f"table_bll_{name}.csv", index=False)
    b["table"].to_csv(out / "table_reality_check.csv", index=False)
    b["rolling"].to_csv(out / "table_rolling_pvalues.csv", index=False)
    if c is not None:
        c["table"].to_csv(out / "table_aqr.csv", index=False)
    if e is not None:
        e["separate"].to_csv(out / "table_assets.csv", index=False)
    write_report(out, d, a, b, c, dd, e, sim)
    summary = {"simulated": sim, "notes": d.get("notes", []),
               "bll": {k: v.to_dict(orient="records") for k, v in a["tables"].items()},
               "reality_check": b["table"].to_dict(orient="records"),
               "aqr": None if c is None else c["table"].to_dict(orient="records"),
               "smile": None if dd is None else dd["regression"].to_dict(orient="records"),
               "assets": None if e is None else {"pooled_slope": e["pooled_slope"], "p_pooled": e["p_pooled"],
                                                  "strategies": e["strategy_stats"]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for name, t in a["tables"].items():
        print(f"A  {name}: buy-sell positive for {(t.buy_minus_sell > 0).sum()} of 26 rules, |t| > 1.96 for "
              f"{(t.t.abs() > 1.96).sum()}")
    for _, r in b["table"].iterrows():
        print(f"B  {r.period}: best {r.best_rule} {r.best_excess_annual:+.1%}/yr, nominal p {r.p_nominal:.3f}, "
              f"Reality Check p {r.p_reality_check:.3f}, SPA p {r.p_spa:.3f}")
    if c is not None:
        f = c["table"][c["table"].period == "Full sample"].iloc[0]
        print(f"C  {f.factor}: {f.mean_annual:+.1%}/yr, Sharpe {f.sharpe:.2f}, alpha t {f.alpha_t:.2f}")
    if dd is not None:
        print(f"D  smile coefficient {dd['regression'].estimate.iloc[2]:+.2f} (t {dd['regression'].t.iloc[2]:.2f})")
    if e is not None:
        print(f"E  pooled slope {e['pooled_slope']:+.3f}, bootstrap p {e['p_pooled']:.3f}; TSM Sharpe "
              f"{e['strategy_stats']['TSM']['sharpe']:.2f}, TSH {e['strategy_stats']['TSH']['sharpe']:.2f}")
    for n in d.get("notes", []):
        print("NOTE:", n)
    print(f"Report: {out / 'lab05_report.md'}")
    if sim:
        print("NOTE: simulated data. These results test the code only.")
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
