"""Currency and carry-trade tools (Chapter 11).

* Monthly spot rates and three-month interest rates from FRED (Federal Reserve H.10 exchange rates and OECD
  interbank rates), with each exchange rate expressed as U.S. dollars per unit of foreign currency
* Currency excess returns: rx = (i* - i_US)/12 + change in log spot (dollars per foreign unit)
* Fama (1984) regressions and carry portfolios sorted on interest differentials

The interest differential stands in for the forward discount, which covered interest parity makes equal to it
(apart from small deviations since 2008). Rates are annual decimals; returns are monthly log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .credit import monthly_fred
from .data import DataUnavailable

# code: (FRED exchange-rate series, True if FRED quotes foreign units per dollar, FRED 3-month rate series)
CURRENCIES = {
    "JPY": ("EXJPUS", True, "IR3TIB01JPM156N"),
    "GBP": ("EXUSUK", False, "IR3TIB01GBM156N"),
    "CHF": ("EXSZUS", True, "IR3TIB01CHM156N"),
    "CAD": ("EXCAUS", True, "IR3TIB01CAM156N"),
    "AUD": ("EXUSAL", False, "IR3TIB01AUM156N"),
    "NZD": ("EXUSNZ", False, "IR3TIB01NZM156N"),
    "SEK": ("EXSDUS", True, "IR3TIB01SEM156N"),
    "NOK": ("EXNOUS", True, "IR3TIB01NOM156N"),
    "EUR": ("EXUSEU", False, "IR3TIB01EZM156N"),
}
US_RATE = "IR3TIB01USM156N"


def load_fx(files: dict | None = None, currencies=None) -> dict:
    """Spot rates (dollars per foreign unit) and annual interest rates (decimals), monthly. Skips missing series."""
    files = files or {}
    spot, rate, missing = {}, {}, []
    us = monthly_fred(US_RATE, files.get(US_RATE)) / 100
    for code in currencies or CURRENCIES:
        fx_id, invert, r_id = CURRENCIES[code]
        try:
            s = monthly_fred(fx_id, files.get(fx_id))
            r = monthly_fred(r_id, files.get(r_id)) / 100
        except DataUnavailable as e:
            missing.append(f"{code}: {str(e).splitlines()[0]}")
            continue
        spot[code] = 1 / s if invert else s
        rate[code] = r
    if not spot:
        raise DataUnavailable("No currency series could be loaded:\n" + "\n".join(missing))
    return {"spot": pd.DataFrame(spot), "rate": pd.DataFrame(rate), "us": us, "missing": missing}


def excess_returns(spot: pd.DataFrame, rate: pd.DataFrame, us: pd.Series) -> dict:
    """Monthly currency excess returns for a U.S. investor and their two parts.

    Row t holds the return from the end of month t-1 to the end of month t, earned on the interest
    differential known at t-1.
    """
    ds = np.log(spot).diff()
    diff = rate.sub(us.reindex(rate.index), axis=0)
    carry = diff.shift(1) / 12
    return {"rx": carry + ds, "carry": carry, "fx": ds, "diff": diff}


def fama_regressions(ds: pd.DataFrame, diff: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Fama (1984): change in log spot (dollars per foreign unit) on the lagged U.S.-minus-foreign differential.

    Uncovered interest parity implies a slope of 1: a currency with a higher rate than the dollar is expected to
    depreciate by the difference. A slope of 0 means the differential is all excess return; negative means
    high-rate currencies tended to appreciate.
    """
    from .stats import ols_newey_west

    rows = []
    x_all, y_all = [], []
    for c in ds.columns:
        d = pd.concat([ds[c].rename("y"), (-diff[c].shift(1) / 12).rename("x")], axis=1).dropna()
        if len(d) < 60:
            continue
        r = ols_newey_west(d["y"].to_numpy(), d["x"].to_numpy(), lags=lags)
        b, t = float(r["coef"][1, 0]), float(r["t"][1, 0])
        se = b / t if t else float("nan")
        rows.append({"currency": c, "slope": b, "se": se, "t_vs_1": (b - 1) / se, "t_vs_0": t, "months": len(d),
                     "start": str(d.index[0])})
        x_all.append(d["x"].to_numpy() - d["x"].mean())
        y_all.append(d["y"].to_numpy() - d["y"].mean())
    if x_all:
        x, y = np.concatenate(x_all), np.concatenate(y_all)
        b = float((x @ y) / (x @ x))
        rows.append({"currency": "Pooled (demeaned)", "slope": b, "se": float("nan"), "t_vs_1": float("nan"),
                     "t_vs_0": float("nan"), "months": len(x), "start": ""})
    return pd.DataFrame(rows)


def carry_portfolio(rx: pd.DataFrame, diff: pd.DataFrame, n_side: int = 3, cost: float = 0.0) -> pd.DataFrame:
    """Long the n_side highest-rate currencies, short the n_side lowest, equal weights, rebalanced monthly.

    Signals use the differential at the end of the previous month. cost is a proportional cost per unit of
    weight traded (each currency position changed counts once). Returns columns: carry (long-short), high, low,
    dollar (average of all currencies), turnover, n.
    """
    sig = diff.shift(1)
    out = []
    prev_w = pd.Series(0.0, index=rx.columns)
    for t in rx.index:
        s = sig.loc[t].dropna()
        r = rx.loc[t]
        avail = s.index[r.reindex(s.index).notna()]
        s = s[avail]
        if len(s) < 2 * n_side:
            prev_w = pd.Series(0.0, index=rx.columns)
            continue
        order = s.sort_values()
        lo, hi = order.index[:n_side], order.index[-n_side:]
        w = pd.Series(0.0, index=rx.columns)
        w[hi] = 1 / n_side
        w[lo] = -1 / n_side
        turnover = float((w - prev_w).abs().sum())
        ret = float((w * r.fillna(0)).sum()) - cost * turnover
        out.append({"date": t, "carry": ret, "high": float(r[hi].mean()), "low": float(r[lo].mean()),
                    "dollar": float(r[avail].mean()), "turnover": turnover, "n": len(s),
                    "carry_interest": float((w * (diff.shift(1).loc[t].reindex(w.index).fillna(0) / 12)).sum())})
        prev_w = w
    df = pd.DataFrame(out).set_index("date")
    df["carry_fx"] = df["carry"] + cost * df["turnover"] - df["carry_interest"]
    return df


def simulated_fx(n_months: int = 600, seed: int = 0, start: str = "1976-01") -> dict:
    """Fake spot rates and interest rates for nine currencies with a carry premium and crash risk. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    codes = list(CURRENCIES)
    k = len(codes)
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    us = np.zeros(n_months)
    us[0] = 0.05
    levels = np.linspace(-0.04, 0.06, k)
    diff = np.zeros((n_months, k))
    diff[0] = levels
    for t in range(1, n_months):
        us[t] = max(0.0, us[t - 1] + 0.0015 * rng.standard_normal())
        diff[t] = levels + 0.97 * (diff[t - 1] - levels) + 0.002 * rng.standard_normal(k)
    crash = rng.random(n_months) < 0.03
    common = 0.025 * rng.standard_normal(n_months)
    ds = np.zeros((n_months, k))
    for t in range(1, n_months):
        z = diff[t - 1] - diff[t - 1].mean()
        # UIP would be -diff/12; here high-rate currencies depreciate by only 30% of the differential, and crash in risk-off months
        ds[t] = -0.3 * diff[t - 1] / 12 + common[t] + 0.015 * rng.standard_normal(k) - crash[t] * 0.8 * z
    spot = pd.DataFrame(np.exp(np.cumsum(ds, axis=0)), index=idx, columns=codes)
    rate = pd.DataFrame(np.maximum(us[:, None] + diff, -0.01), index=idx, columns=codes)
    vix = pd.Series(18 + 4 * rng.standard_normal(n_months) + 25 * crash, index=idx).clip(lower=9)
    mkt = pd.Series(0.006 + 0.04 * rng.standard_normal(n_months) - 0.08 * crash, index=idx)
    return {"spot": spot, "rate": rate, "us": pd.Series(us, idx), "vix": vix, "mkt": mkt, "missing": []}
