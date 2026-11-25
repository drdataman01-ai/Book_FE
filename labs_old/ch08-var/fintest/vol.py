"""Volatility models and forecast evaluation (Chapters 7 and 8).

* EWMA (RiskMetrics), GARCH(1,1) and GJR-GARCH(1,1) with normal or Student-t
  innovations, estimated by maximum likelihood
* Multi-step variance forecasts
* The HAR model of Corsi (2009) on daily squared returns
* Loss functions for variance forecasts: MSE and QLIKE (Patton 2011), and
  Mincer-Zarnowitz regressions
* A FRED reader for series such as the VIX

Returns are daily decimals. Variances are per day unless stated otherwise.
"""

from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

from .data import DEFAULT_CACHE, DataUnavailable

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def ewma_variance(r: np.ndarray, lam: float = 0.94, init: float | None = None) -> np.ndarray:
    """sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t-1]^2: the variance forecast for day t made at the end of t-1."""
    r = np.asarray(r, dtype=float)
    s0 = np.var(r[:250]) if init is None else init
    x = np.concatenate([[s0], (1 - lam) * r[:-1] ** 2])
    return lfilter([1.0], [1.0, -lam], x)


def garch_variance(r: np.ndarray, omega: float, alpha: float, beta: float, gamma: float = 0.0,
                   init: float | None = None) -> np.ndarray:
    """GJR-GARCH(1,1) conditional variance; gamma = 0 gives GARCH(1,1).

    sigma2[t] = omega + (alpha + gamma * 1[r[t-1] < 0]) * r[t-1]^2 + beta * sigma2[t-1]
    """
    r = np.asarray(r, dtype=float)
    s0 = np.var(r) if init is None else init
    shock = (alpha + gamma * (r[:-1] < 0)) * r[:-1] ** 2 + omega
    x = np.concatenate([[s0], shock])
    out = lfilter([1.0], [1.0, -beta], x)
    return out


def _neg_loglik(params, r, dist, gjr):
    if gjr:
        omega, alpha, gamma, beta = params[:4]
        rest = params[4:]
    else:
        omega, alpha, beta = params[:3]
        gamma = 0.0
        rest = params[3:]
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta + 0.5 * gamma >= 0.9999 or alpha + gamma < 0:
        return 1e10
    s2 = garch_variance(r, omega, alpha, beta, gamma)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    if dist == "t":
        nu = rest[0]
        if nu <= 2.05 or nu > 200:
            return 1e10
        z2 = r ** 2 / (s2 * (nu - 2) / nu)
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(math.pi * nu) - 0.5 * np.log(s2 * (nu - 2) / nu)
              - (nu + 1) / 2 * np.log1p(z2 / nu))
    else:
        ll = -0.5 * (np.log(2 * math.pi) + np.log(s2) + r ** 2 / s2)
    return -float(ll.sum())


def fit_garch(r: np.ndarray, gjr: bool = False, dist: str = "normal") -> dict:
    """Maximum-likelihood estimates. Returns omega, alpha, gamma, beta, nu, persistence and long-run variance."""
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    v = np.var(r)
    x0 = [v * 0.05, 0.03, 0.08, 0.90] if gjr else [v * 0.05, 0.08, 0.90]
    if dist == "t":
        x0 = x0 + [8.0]
    res = minimize(_neg_loglik, x0, args=(r, dist, gjr), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-6})
    p = res.x
    if gjr:
        omega, alpha, gamma, beta = p[:4]
    else:
        omega, alpha, beta = p[:3]
        gamma = 0.0
    nu = p[-1] if dist == "t" else float("inf")
    pers = alpha + beta + 0.5 * gamma
    return {"omega": float(omega), "alpha": float(alpha), "gamma": float(gamma), "beta": float(beta), "nu": float(nu),
            "persistence": float(pers), "long_run_var": float(omega / (1 - pers)), "loglik": -float(res.fun),
            "gjr": gjr, "dist": dist}


def garch_forecast_sum(sigma2_next: np.ndarray, params: dict, h: int) -> np.ndarray:
    """Sum of expected variances over days t+1..t+h, given the one-day-ahead variance."""
    V, p = params["long_run_var"], params["persistence"]
    return h * V + (np.asarray(sigma2_next, dtype=float) - V) * (p ** np.arange(h)).sum()


# ---------------------------------------------------------------------------
# HAR
# ---------------------------------------------------------------------------


def har_features(rv: np.ndarray) -> np.ndarray:
    """Daily, weekly (5-day) and monthly (22-day) averages of a daily variance proxy, known at the end of day t."""
    s = pd.Series(rv)
    return np.column_stack([s.to_numpy(), s.rolling(5).mean().to_numpy(), s.rolling(22).mean().to_numpy()])


def future_sum(x: np.ndarray, h: int) -> np.ndarray:
    """y[t] = x[t+1] + ... + x[t+h] (NaN where not observed)."""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    T = len(x)
    out = np.full(T, np.nan)
    out[:T - h] = c[1 + h:T + 1] - c[1:T - h + 1]
    return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def mse_loss(target, forecast) -> np.ndarray:
    return (np.asarray(target, float) - np.asarray(forecast, float)) ** 2


def qlike_loss(target, forecast) -> np.ndarray:
    """QLIKE in the form log(F) + RV/F, robust to noisy proxies (Patton 2011) and defined when RV = 0."""
    f = np.asarray(forecast, float)
    return np.log(f) + np.asarray(target, float) / f


def mincer_zarnowitz(target, forecast) -> dict:
    y, f = np.asarray(target, float), np.asarray(forecast, float)
    ok = np.isfinite(y) & np.isfinite(f)
    A = np.column_stack([np.ones(ok.sum()), f[ok]])
    b, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    e = y[ok] - A @ b
    return {"intercept": float(b[0]), "slope": float(b[1]), "r2": float(1 - e.var() / y[ok].var())}


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def load_fred(series: str, path=None, cache_dir: Path = DEFAULT_CACHE) -> pd.Series:
    """A FRED series as a pandas Series with a DatetimeIndex. Downloads fredgraph.csv and caches it, or reads `path`."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise DataUnavailable(f"File not found: {p}")
        text = p.read_text()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        f = cache_dir / f"{series}.csv"
        if not f.exists():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    f.write_bytes(resp.read())
            except Exception as e:  # noqa: BLE001
                raise DataUnavailable(f"Could not download {url} ({e}). Download the CSV from "
                                      f"https://fred.stlouisfed.org/series/{series} and pass it with the file option.") from e
        text = f.read_text()
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").to_numpy(), index=pd.to_datetime(df[date_col]),
                  name=series)
    return s.dropna()
