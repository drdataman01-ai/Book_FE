"""Option pricing and option-strategy evaluation (Chapters 18 and 18).

* Black-Scholes-Merton prices, deltas and vegas for European calls and puts
* Monthly put-writing and buy-write (covered call) returns built from an
  index path and an implied volatility
* Variance-swap style payoffs: implied minus realized variance
* Performance measures that are hard to game with option-like payoffs:
  the manipulation-proof performance measure (Goetzmann, Ingersoll, Spiegel
  and Welch 2007), upside and downside betas and capture ratios
* A reader for Cboe index history files (e.g. BXM, PUT)
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import DataUnavailable

# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------


def _d1d2(S, K, T, r, sigma, q=0.0):
    S, K, T, r, sigma, q = (np.asarray(x, float) for x in (S, K, T, r, sigma, q))
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol
    return d1, d1 - vol


def bs_price(S, K, T, r, sigma, kind: str = "call", q=0.0):
    """European option price. r and q are continuously compounded annual rates; T in years."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    S, K, T, r, q = (np.asarray(x, float) for x in (S, K, T, r, q))
    if kind == "call":
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def bs_delta(S, K, T, r, sigma, kind: str = "call", q=0.0):
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    e = np.exp(-np.asarray(q, float) * np.asarray(T, float))
    return e * stats.norm.cdf(d1) if kind == "call" else e * (stats.norm.cdf(d1) - 1)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Change in price for a change of 1.00 (100 vol points) in sigma."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    return np.asarray(S, float) * np.exp(-np.asarray(q, float) * np.asarray(T, float)) * stats.norm.pdf(d1) * \
        np.sqrt(np.asarray(T, float))


def implied_vol(price, S, K, T, r, kind: str = "call", q=0.0, lo=1e-4, hi=5.0) -> float:
    """Black-Scholes implied volatility by bisection (scalar)."""
    f = lambda s: float(bs_price(S, K, T, r, s, kind, q)) - price  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Monthly option-writing strategies
# ---------------------------------------------------------------------------


def monthly_option_writes(ret: pd.Series, rf: pd.Series, iv: pd.Series, put_strike: float = 1.0,
                          call_strike: float = 1.0, iv_shift: float = 0.0) -> pd.DataFrame:
    """Returns of one-month put-writing and buy-write strategies, rolled at each month end.

    ret, rf  daily total returns of the index and the risk-free rate (decimals)
    iv       daily implied volatility in percent (e.g. the VIX), observed at each roll date
    put_strike, call_strike  strike / index level at the roll (1.0 = at the money, 1.02 = 2% out of the money call)
    iv_shift  vol points added to iv when pricing (negative to remove the skew premium in the VIX)

    Put-write (like Cboe PUT): sell a put and hold collateral equal to the strike in bills:
        r = rf_m + P/K * (1 + rf_m) - max(K - S1, 0) / K
    Buy-write (like Cboe BXM): hold the index and sell a call:
        r = S1 - max(S1 - K, 0) + C * (1 + rf_m) - 1          (S0 = 1)
    The option is priced by Black-Scholes on the total-return index (no dividend yield), so the forward is
    consistent with the path used for the payoff. With equal strikes, the two strategies have identical returns:
    a covered call and a cash-secured put with the same strike are the same position (put-call parity).
    """
    df = pd.concat({"ret": ret, "rf": rf.reindex(ret.index).fillna(0.0),
                    "iv": iv.reindex(ret.index).ffill(limit=5)}, axis=1).dropna(subset=["ret"])
    months = df.index.to_period("M")
    ends = df.groupby(months).tail(1).index  # last trading day of each month
    rows = []
    for t0, t1 in zip(ends[:-1], ends[1:]):
        sig = df.at[t0, "iv"]
        if not np.isfinite(sig):
            continue
        seg = df.loc[(df.index > t0) & (df.index <= t1)]
        n = len(seg)
        S1 = float((1 + seg["ret"]).prod())
        rf_m = float((1 + seg["rf"]).prod() - 1)
        T = n / 252
        r_cc = math.log(1 + rf_m) / T
        sigma = max((sig + iv_shift) / 100, 0.01)
        Kp, Kc = put_strike, call_strike
        P = float(bs_price(1.0, Kp, T, r_cc, sigma, "put"))
        C = float(bs_price(1.0, Kc, T, r_cc, sigma, "call"))
        put_w = rf_m + P / Kp * (1 + rf_m) - max(Kp - S1, 0.0) / Kp
        buy_w = S1 - max(S1 - Kc, 0.0) + C * (1 + rf_m) - 1
        rows.append({"date": t1.to_period("M"), "index": S1 - 1, "rf": rf_m, "put_write": put_w, "buy_write": buy_w,
                     "put_premium": P, "call_premium": C, "iv": sig})
    return pd.DataFrame(rows).set_index("date")


def variance_swap_pnl(ret: pd.Series, iv: pd.Series, horizon: int = 21) -> pd.DataFrame:
    """Monthly payoff to a seller of a variance swap struck at the VIX: implied minus realized variance.

    Sampled at month ends, non-overlapping. Implied variance = (VIX/100)^2 * horizon/252; realized = sum of
    squared daily returns over the next `horizon` trading days. Both in decimal variance over the horizon.
    """
    df = pd.concat({"ret": ret, "iv": iv.reindex(ret.index).ffill(limit=5)}, axis=1).dropna(subset=["ret"])
    r2 = df["ret"].to_numpy() ** 2
    c = np.concatenate([[0.0], np.cumsum(r2)])
    n = len(r2)
    fut = np.full(n, np.nan)
    fut[:n - horizon] = c[1 + horizon:n + 1] - c[1:n - horizon + 1]
    df["realized"] = fut
    df["implied"] = (df["iv"] / 100) ** 2 * horizon / 252
    ends = df.groupby(df.index.to_period("M")).tail(1)
    out = ends[["implied", "realized"]].dropna()
    out["pnl"] = out["implied"] - out["realized"]
    out.index = out.index.to_period("M")
    return out


# ---------------------------------------------------------------------------
# Performance measures
# ---------------------------------------------------------------------------


def mppm(r: np.ndarray, rf: np.ndarray, rho: float = 3.0, periods: int = 12) -> float:
    """Manipulation-proof performance measure (Goetzmann, Ingersoll, Spiegel and Welch 2007), annualized.

    Theta = 1 / ((1 - rho) dt) * ln( mean( ((1 + r) / (1 + rf)) ^ (1 - rho) ) ),  dt = 1/periods.
    It is the certainty-equivalent excess return of an investor with relative risk aversion rho, so strategies
    that sell crash insurance cannot raise it simply by adding negative skewness.
    """
    r, rf = np.asarray(r, float), np.asarray(rf, float)
    ok = np.isfinite(r) & np.isfinite(rf)
    x = ((1 + r[ok]) / (1 + rf[ok])) ** (1 - rho)
    return float(periods / (1 - rho) * np.log(x.mean()))


def up_down_betas(y: np.ndarray, m: np.ndarray) -> dict:
    """Betas estimated separately in months when the market's excess return is negative and positive."""
    y, m = np.asarray(y, float), np.asarray(m, float)
    ok = np.isfinite(y) & np.isfinite(m)
    y, m = y[ok], m[ok]
    out = {}
    for name, sel in (("down", m < 0), ("up", m >= 0)):
        X = np.column_stack([np.ones(sel.sum()), m[sel]])
        out[f"beta_{name}"] = float(np.linalg.lstsq(X, y[sel], rcond=None)[0][1])
    return out


def capture_ratios(y: np.ndarray, m: np.ndarray) -> dict:
    """Average strategy return in up and down market months, divided by the market's."""
    y, m = np.asarray(y, float), np.asarray(m, float)
    ok = np.isfinite(y) & np.isfinite(m)
    y, m = y[ok], m[ok]
    up, dn = m > 0, m < 0
    return {"upside_capture": float(y[up].mean() / m[up].mean()), "downside_capture": float(y[dn].mean() / m[dn].mean())}


# ---------------------------------------------------------------------------
# Cboe index files
# ---------------------------------------------------------------------------


def read_cboe_index(path, name: str | None = None) -> pd.Series:
    """Daily index levels from a Cboe history CSV (e.g. BXM_History.csv, PUT_History.csv).

    Skips any preamble, finds the first row whose first field is a date, and uses the last numeric column.
    """
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}. Download the index history from https://www.cboe.com/us/indices/.")
    lines = p.read_text(errors="ignore").splitlines()
    start = None
    for i, line in enumerate(lines):
        first = line.split(",")[0].strip().strip('"')
        if first and not pd.isna(pd.to_datetime(first, errors="coerce")) and any(ch.isdigit() for ch in first) \
                and len(first) >= 6:
            start = i
            break
    if start is None:
        raise DataUnavailable(f"No dated rows found in {p}")
    body = pd.read_csv(io.StringIO("\n".join(lines[start:])), header=None)
    dates = pd.to_datetime(body.iloc[:, 0].astype(str).str.strip().str.strip('"'), errors="coerce")
    vals = body.iloc[:, 1:].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(",", ""), errors="coerce"))
    col = [c for c in vals.columns if vals[c].notna().mean() > 0.9]
    if not col:
        raise DataUnavailable(f"No numeric column found in {p}")
    s = pd.Series(vals[col[-1]].to_numpy(), index=dates).dropna()
    s = s[s.index.notna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(name or p.stem.split("_")[0].upper())


def monthly_returns_from_levels(levels: pd.Series) -> pd.Series:
    m = levels.groupby(levels.index.to_period("M")).last()
    return m.pct_change().dropna()
