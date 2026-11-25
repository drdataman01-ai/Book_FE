"""Out-of-sample forecasting and forecast evaluation (Chapter 6).

* walk_forward: expanding or rolling regressions, re-estimated every period,
  with an optional gap (purge) between training targets and the forecast date
  and the Campbell-Thompson (2008) restrictions
* out-of-sample R-squared, Clark-West (2007) and Diebold-Mariano (1995) tests
* proper scoring rules for Gaussian predictive densities: log score and CRPS
* the certainty-equivalent gain of a mean-variance investor

Conventions: forecasts are made at the end of period t for the target paired
with row t (for one-month forecasts, the next month's premium).
"""

from __future__ import annotations

import math

import numpy as np

from .stats import normal_cdf, normal_sf


def lagged_pairs(target: np.ndarray, x: np.ndarray, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Pair predictors at t with the sum of the target over t+1..t+horizon.

    Returns (z, X) of equal length; the last `horizon` rows (whose targets are
    not yet observed) are NaN in z.
    """
    y = np.asarray(target, dtype=float)
    T = len(y)
    z = np.full(T, np.nan)
    for t in range(T - horizon):
        z[t] = y[t + 1:t + 1 + horizon].sum()
    return z, np.asarray(x, dtype=float)


def walk_forward(z: np.ndarray, X: np.ndarray, first: int, window: int | None = None, gap: int = 0,
                 min_train: int = 60, sign: np.ndarray | int | None = None, nonneg: bool = False,
                 coef_fixed: np.ndarray | None = None) -> dict:
    """Recursive out-of-sample forecasts of z[t] from X[t], for t = first, ..., T-1.

    z, X       target and predictors (X may be (T,) or (T, K)); NaNs are skipped
    first      index of the first forecast
    window     None = expanding window; an integer = rolling window of that many pairs
    gap        number of most recent pairs to drop from training. 0 is right for
               one-period targets. For targets summed over h periods, the pairs
               s = t-h+1 .. t-1 have targets that are not yet known at t, so gap
               must be h-1 ("purging"); gap=0 then leaks future returns.
    sign       Campbell-Thompson slope restriction: if the estimated slope has
               the wrong sign, use the benchmark forecast instead (scalar or per-column)
    nonneg     Campbell-Thompson forecast restriction: replace negative forecasts by 0
    coef_fixed use these coefficients (intercept first) instead of estimating;
               with full-sample coefficients this is the classic look-ahead error

    The benchmark is the mean of the same training targets (the historical mean).
    """
    z = np.asarray(z, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T = len(z)
    ok = np.isfinite(z) & np.isfinite(X).all(axis=1)
    f_model = np.full(T, np.nan)
    f_bench = np.full(T, np.nan)
    slopes = np.full((T, X.shape[1]), np.nan)
    for t in range(first, T):
        if not (np.isfinite(z[t]) and np.isfinite(X[t]).all()):
            continue
        hi = t - gap  # training pairs s < hi
        lo = 0 if window is None else max(0, hi - window)
        m = ok[lo:hi]
        if m.sum() < min_train:
            continue
        zt, Xt = z[lo:hi][m], X[lo:hi][m]
        f_bench[t] = zt.mean()
        if coef_fixed is not None:
            b = np.asarray(coef_fixed, dtype=float)
        else:
            A = np.column_stack([np.ones(len(zt)), Xt])
            b = np.linalg.lstsq(A, zt, rcond=None)[0]
        slopes[t] = b[1:]
        f = b[0] + X[t] @ b[1:]
        if sign is not None:
            sg = np.broadcast_to(np.asarray(sign, dtype=float), b[1:].shape)
            if np.any(b[1:] * sg < 0):
                f = f_bench[t]
        if nonneg:
            f = max(f, 0.0)
        f_model[t] = f
    valid = np.isfinite(f_model) & np.isfinite(f_bench) & np.isfinite(z)
    return {"actual": z, "model": f_model, "bench": f_bench, "valid": valid, "slopes": slopes}


def oos_r2(actual, f_model, f_bench) -> float:
    """Campbell-Thompson out-of-sample R-squared: 1 - SSE(model) / SSE(benchmark)."""
    a, m, b = (np.asarray(v, dtype=float) for v in (actual, f_model, f_bench))
    ok = np.isfinite(a) & np.isfinite(m) & np.isfinite(b)
    return float(1 - ((a[ok] - m[ok]) ** 2).sum() / ((a[ok] - b[ok]) ** 2).sum())


def _nw_mean_t(d: np.ndarray, lags: int) -> float:
    d = np.asarray(d, dtype=float)
    n = len(d)
    u = d - d.mean()
    s = (u @ u) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(d.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def clark_west(actual, f_model, f_bench, lags: int = 0) -> dict:
    """Clark and West (2007) test of equal predictive accuracy for nested models.

    H0: the benchmark (historical mean) is as accurate as the larger model.
    One-sided: large positive statistics favor the model. The adjustment
    (f_bench - f_model)^2 removes the noise the larger model adds by estimating
    parameters whose true value is zero under H0.
    """
    a, m, b = (np.asarray(v, dtype=float) for v in (actual, f_model, f_bench))
    ok = np.isfinite(a) & np.isfinite(m) & np.isfinite(b)
    a, m, b = a[ok], m[ok], b[ok]
    f = (a - b) ** 2 - ((a - m) ** 2 - (b - m) ** 2)
    t = _nw_mean_t(f, lags)
    return {"stat": t, "p_value": float(normal_sf(t))}


def diebold_mariano(actual, f1, f2, lags: int = 0, small_sample: bool = True) -> dict:
    """Diebold and Mariano (1995) test with squared-error loss.

    Positive statistics mean forecast 2 is more accurate than forecast 1.
    One-sided p-value for H0: forecast 2 is no more accurate. With
    small_sample=True the Harvey, Leybourne and Newbold (1997) correction is
    applied. For nested models (a model against the historical mean) the test
    is undersized; prefer Clark-West.
    """
    a, x1, x2 = (np.asarray(v, dtype=float) for v in (actual, f1, f2))
    ok = np.isfinite(a) & np.isfinite(x1) & np.isfinite(x2)
    d = (a[ok] - x1[ok]) ** 2 - (a[ok] - x2[ok]) ** 2
    t = _nw_mean_t(d, lags)
    if small_sample:
        n, h = len(d), lags + 1
        t *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"stat": t, "p_value": float(normal_sf(t))}


def cumulative_sse_difference(actual, f_model, f_bench) -> np.ndarray:
    """Running sum of SSE(benchmark) - SSE(model): rising when the model is winning (Goyal and Welch 2008)."""
    a, m, b = (np.asarray(v, dtype=float) for v in (actual, f_model, f_bench))
    d = np.where(np.isfinite(a) & np.isfinite(m) & np.isfinite(b), (a - b) ** 2 - (a - m) ** 2, 0.0)
    return np.cumsum(d)


# ---------------------------------------------------------------------------
# Probabilistic forecasts
# ---------------------------------------------------------------------------


def gaussian_log_score(y, mu, sigma) -> np.ndarray:
    """Log predictive density of y under N(mu, sigma^2); higher is better."""
    y, mu, sigma = (np.asarray(v, dtype=float) for v in (y, mu, sigma))
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((y - mu) / sigma) ** 2


def crps_gaussian(y, mu, sigma) -> np.ndarray:
    """Continuous ranked probability score of N(mu, sigma^2) at y; lower is better (Gneiting and Raftery 2007)."""
    y, mu, sigma = (np.asarray(v, dtype=float) for v in (y, mu, sigma))
    zz = (y - mu) / sigma
    pdf = np.exp(-0.5 * zz ** 2) / math.sqrt(2 * math.pi)
    cdf = np.asarray(normal_cdf(zz), dtype=float)
    return sigma * (zz * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def interval_coverage(y, mu, sigma, level: float = 0.9) -> float:
    """Share of outcomes inside the central `level` interval of N(mu, sigma^2)."""
    from .stats import normal_ppf

    k = normal_ppf(0.5 + level / 2)
    y, mu, sigma = (np.asarray(v, dtype=float) for v in (y, mu, sigma))
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    return float((np.abs(y[ok] - mu[ok]) <= k * sigma[ok]).mean())


# ---------------------------------------------------------------------------
# Economic value
# ---------------------------------------------------------------------------


def rolling_variance(r: np.ndarray, window: int = 60) -> np.ndarray:
    """Variance of r over the `window` periods ending at t (known at the end of t)."""
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    for t in range(window - 1, len(r)):
        w = r[t - window + 1:t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= window // 2:
            out[t] = w.var(ddof=1)
    return out


def mean_variance_weights(forecast, variance, gamma: float = 3.0, lower: float = 0.0, upper: float = 1.5) -> np.ndarray:
    """Equity weight of a mean-variance investor: forecast / (gamma * variance), bounded."""
    w = np.asarray(forecast, dtype=float) / (gamma * np.asarray(variance, dtype=float))
    return np.clip(w, lower, upper)


def certainty_equivalent(excess, rf, weights, gamma: float = 3.0, periods: int = 12) -> float:
    """Annualized certainty-equivalent return of rf + w * excess: mean - gamma/2 * variance."""
    e, r, w = (np.asarray(v, dtype=float) for v in (excess, rf, weights))
    ok = np.isfinite(e) & np.isfinite(r) & np.isfinite(w)
    p = r[ok] + w[ok] * e[ok]
    return float(periods * (p.mean() - gamma / 2 * p.var(ddof=1)))


def expanding_univariate(z: np.ndarray, x: np.ndarray, first: int, min_train: int = 60) -> dict:
    """Fast expanding-window forecasts for one predictor, with no gaps in the data.

    Same result as walk_forward(z, x, first) when z and x have no NaNs before
    the last row, computed from running sums instead of a regression per date.
    """
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    T = len(z)
    n = np.arange(T)  # training pairs for forecast t: s = 0..t-1, so n = t
    cz = np.concatenate([[0.0], np.cumsum(z)])[:T]
    cx = np.concatenate([[0.0], np.cumsum(x)])[:T]
    cxx = np.concatenate([[0.0], np.cumsum(x * x)])[:T]
    cxz = np.concatenate([[0.0], np.cumsum(x * z)])[:T]
    with np.errstate(invalid="ignore", divide="ignore"):
        mz, mx = cz / n, cx / n
        b = (cxz / n - mx * mz) / (cxx / n - mx ** 2)
        f_model = mz + b * (x - mx)
    f_bench = mz.copy()
    keep = (n >= max(first, min_train)) & np.isfinite(z)
    f_model[~keep] = np.nan
    f_bench[~keep] = np.nan
    return {"actual": z, "model": f_model, "bench": f_bench, "valid": keep, "slopes": b}
