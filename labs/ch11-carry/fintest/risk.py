"""Value at Risk, expected shortfall and their backtests (Chapter 8).

Conventions: returns are decimals; VaR and ES are reported as positive
losses. `alpha` is the tail probability, so a 99% VaR has alpha = 0.01, and
the VaR is breached on day t+1 when the loss -r[t+1] exceeds the VaR
forecast made at the end of day t.

* Parametric VaR and ES (normal and standardized Student-t)
* Historical-simulation VaR and ES
* Kupiec (1995) unconditional coverage, Christoffersen (1998) independence
  and conditional coverage tests
* The Basel traffic light for 250-day backtests
* An expected-shortfall backtest on exceedance residuals (McNeil and Frey 2000)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------


def var_normal(sigma, alpha: float = 0.01, mu=0.0):
    """VaR of a normal return with mean mu and standard deviation sigma."""
    return -(np.asarray(mu) + stats.norm.ppf(alpha) * np.asarray(sigma))


def es_normal(sigma, alpha: float = 0.01, mu=0.0):
    """Expected shortfall (average loss beyond the VaR) of a normal return."""
    return -np.asarray(mu) + np.asarray(sigma) * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha


def t_scale(nu: float) -> float:
    """Factor that turns a Student-t variable into one with unit variance."""
    return math.sqrt((nu - 2) / nu)


def var_t(sigma, nu: float, alpha: float = 0.01):
    """VaR when standardized returns are Student-t with nu degrees of freedom and unit variance."""
    return -stats.t.ppf(alpha, nu) * t_scale(nu) * np.asarray(sigma)


def es_t(sigma, nu: float, alpha: float = 0.01):
    """Expected shortfall for standardized Student-t returns with unit variance."""
    q = stats.t.ppf(alpha, nu)
    es_std = stats.t.pdf(q, nu) / alpha * (nu + q ** 2) / (nu - 1)
    return es_std * t_scale(nu) * np.asarray(sigma)


def empirical_var_es(x: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """VaR and ES of a sample of returns: the alpha-quantile of losses and the mean loss beyond it."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return float(-q), float(-tail.mean())


def historical_var_es(r: pd.Series, window: int = 250, alpha: float = 0.01) -> pd.DataFrame:
    """Rolling historical-simulation VaR and ES, made at the end of each day for the next day."""
    v = r.to_numpy(float)
    n = len(v)
    out = np.full((n, 2), np.nan)
    k = max(1, int(math.floor(alpha * window)))  # number of tail observations
    for t in range(window - 1, n):
        w = np.sort(v[t - window + 1:t + 1])
        out[t, 0] = -np.quantile(w, alpha)
        out[t, 1] = -w[:k].mean() if k > 0 else -w[0]
    return pd.DataFrame(out, index=r.index, columns=["var", "es"])


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


def hits(returns: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """1 where the next day's loss exceeds the VaR forecast made the day before; NaN where either is missing.

    `var_forecast[t]` is the forecast made at the end of day t for day t+1.
    """
    r = np.asarray(returns, float)
    v = np.asarray(var_forecast, float)
    h = np.full(len(r), np.nan)
    ok = np.isfinite(r[1:]) & np.isfinite(v[:-1])
    h[1:][ok] = (-r[1:][ok] > v[:-1][ok]).astype(float)
    return h


def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


def kupiec(h: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures test: is the breach rate equal to alpha? LR ~ chi2(1)."""
    h = np.asarray(h, float)
    h = h[np.isfinite(h)]
    n, x = len(h), int(h.sum())
    phat = x / n
    ll0 = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return {"n": n, "breaches": x, "rate": phat, "expected": alpha * n, "lr": lr, "p": float(stats.chi2.sf(lr, 1))}


def christoffersen(h: np.ndarray, alpha: float) -> dict:
    """Christoffersen (1998) independence and conditional-coverage tests.

    Independence: is a breach today more likely after a breach yesterday? LR_ind ~ chi2(1).
    Conditional coverage: LR_cc = LR_uc + LR_ind ~ chi2(2).
    """
    h = np.asarray(h, float)
    h = h[np.isfinite(h)].astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    ll1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr_ind = max(-2 * (ll0 - ll1), 0.0)
    uc = kupiec(h, alpha)
    lr_cc = uc["lr"] + lr_ind
    return {"p01": p01, "p11": p11, "lr_ind": lr_ind, "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2)), "n11": n11}


def basel_zone(breaches_in_250: int) -> str:
    """Basel traffic light for a 99% VaR over 250 days: green 0-4, yellow 5-9, red 10 or more."""
    if breaches_in_250 <= 4:
        return "green"
    return "yellow" if breaches_in_250 <= 9 else "red"


def yearly_zones(h: pd.Series) -> pd.DataFrame:
    """Breaches per calendar year (scaled to 250 days) and the Basel zone for each year."""
    x = h.dropna()
    g = x.groupby(x.index.year)
    df = pd.DataFrame({"breaches": g.sum(), "days": g.count()})
    df = df[df.days >= 200]
    df["per_250"] = df.breaches * 250 / df.days
    df["zone"] = [basel_zone(int(round(v))) for v in df.per_250]
    return df


def es_backtest(returns: np.ndarray, var_forecast: np.ndarray, es_forecast: np.ndarray,
                sigma_forecast: np.ndarray | None = None) -> dict:
    """Backtest ES on breach days (McNeil and Frey 2000).

    On each day the VaR is breached, the exceedance residual is (loss - ES forecast), divided by the volatility
    forecast if one is given. If the ES is right, these residuals have mean zero; a positive mean means the ES
    understated tail losses. Reports a one-sided t-statistic and the ratio of average realized tail loss to average
    predicted ES on breach days.
    """
    r = np.asarray(returns, float)
    v, e = np.asarray(var_forecast, float), np.asarray(es_forecast, float)
    loss, vv, ee = -r[1:], v[:-1], e[:-1]
    ok = np.isfinite(loss) & np.isfinite(vv) & np.isfinite(ee) & (loss > vv)
    resid = loss[ok] - ee[ok]
    if sigma_forecast is not None:
        resid = resid / np.asarray(sigma_forecast, float)[:-1][ok]
    n = len(resid)
    if n < 3:
        return {"n": n, "ratio": float("nan"), "mean_resid": float("nan"), "t": float("nan"), "p": float("nan")}
    t = resid.mean() / (resid.std(ddof=1) / math.sqrt(n))
    return {"n": n, "ratio": float(loss[ok].mean() / ee[ok].mean()), "mean_resid": float(resid.mean()),
            "t": float(t), "p": float(stats.t.sf(t, n - 1))}


def durations(h: np.ndarray) -> np.ndarray:
    """Days between consecutive breaches."""
    h = np.asarray(h, float)
    idx = np.flatnonzero(h == 1)
    return np.diff(idx)
