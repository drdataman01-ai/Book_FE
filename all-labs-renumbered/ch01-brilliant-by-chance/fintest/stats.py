"""Statistics for noisy returns (Lab 1 subset).

Functions accept monthly decimal returns unless noted. Annualization uses 12
periods per year.
"""

from __future__ import annotations

import math

import numpy as np

PERIODS_PER_YEAR = 12


def annualized_mean(r: np.ndarray, periods: int = PERIODS_PER_YEAR) -> float:
    return float(np.nanmean(r) * periods)


def annualized_vol(r: np.ndarray, periods: int = PERIODS_PER_YEAR) -> float:
    return float(np.nanstd(r, ddof=1) * math.sqrt(periods))


def sharpe_ratio(excess: np.ndarray, periods: int = PERIODS_PER_YEAR, axis: int = 0) -> np.ndarray:
    """Annualized Sharpe ratio of excess returns (along `axis`)."""
    mu = np.nanmean(excess, axis=axis)
    sd = np.nanstd(excess, axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mu / sd * math.sqrt(periods)


def years_to_detect(sharpe: float, t_crit: float = 1.96) -> float:
    """Years of data needed before t = SR * sqrt(T) reaches t_crit (Section 1.2)."""
    return (t_crit / sharpe) ** 2


def sharpe_standard_error(years: float) -> float:
    """Rough standard error of an annual Sharpe ratio estimated from `years` of data."""
    return 1.0 / math.sqrt(years)


def normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def normal_sf(x):
    return 1.0 - normal_cdf(x)


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def bonferroni_threshold(n_tests: int, alpha: float = 0.05, one_sided: bool = True) -> float:
    """t threshold that keeps the chance of any false discovery below alpha."""
    tail = alpha / n_tests if one_sided else alpha / (2 * n_tests)
    return normal_ppf(1 - tail)


def expected_max_normal(n: int, n_sims: int = 20000, seed: int = 0) -> float:
    """Expected maximum of n independent standard normals (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    total, done, batch = 0.0, 0, max(1, 2_000_000 // max(n, 1))
    while done < n_sims:
        k = min(batch, n_sims - done)
        total += rng.standard_normal((k, n)).max(axis=1).sum()
        done += k
    return total / n_sims


def newey_west_lag(n_obs: int) -> int:
    """Rule-of-thumb lag length: floor(4 (T/100)^(2/9))."""
    return int(math.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))


def ols_newey_west(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of each column of y on [1, x] with Newey-West standard errors.

    y  array of shape (T,) or (T, N): N regressions share the regressor x
    x  array of shape (T,) or (T, K)

    Returns coefficients and t-statistics with shape (K+1, N); row 0 is the
    intercept (alpha). Vectorized across the N columns of y, which is what
    makes testing 1,000 strategies fast.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim == 1:
        x = x[:, None]
    t_obs = y.shape[0]
    X = np.column_stack([np.ones(t_obs), x])  # (T, K+1)
    k = X.shape[1]
    lags = newey_west_lag(t_obs) if lags is None else lags

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y  # (K+1, N)
    resid = y - X @ beta  # (T, N)

    # Score contributions g_t = x_t * e_t, shape (T, K+1, N)
    g = X[:, :, None] * resid[:, None, :]
    S = np.einsum("tin,tjn->ijn", g, g)
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = np.einsum("tin,tjn->ijn", g[lag:], g[:-lag])
        S += w * (gamma + np.transpose(gamma, (1, 0, 2)))
    cov = np.einsum("ij,jkn,kl->iln", xtx_inv, S, xtx_inv)
    se = np.sqrt(np.stack([cov[i, i, :] for i in range(k)]))
    return {"coef": beta, "se": se, "t": beta / se, "lags": lags, "n_obs": t_obs}
