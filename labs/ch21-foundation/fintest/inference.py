"""Inference for noisy returns (Chapter 3).

* Stylized facts: moments, autocorrelations, tail counts
* Regression standard errors: classical, White, Newey-West (vectorized across
  many dependent variables that share the same regressors)
* Stationary bootstrap (Politis and Romano 1994)
* GRS test of zero alphas (Gibbons, Ross and Shanken 1989)
* Fama-MacBeth regressions with the Shanken (1992) correction
* Long-horizon regression t-statistics: classical, Newey-West, Hodrick (1992) 1B
"""

from __future__ import annotations

import math

import numpy as np

from .stats import newey_west_lag, normal_sf, ols_newey_west

# ---------------------------------------------------------------------------
# Stylized facts
# ---------------------------------------------------------------------------


def moments(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    m, s = r.mean(), r.std(ddof=1)
    z = (r - m) / s
    return {"n": int(r.size), "mean": float(m), "std": float(s),
            "skewness": float((z ** 3).mean()), "kurtosis": float((z ** 4).mean())}


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations at lags 1..nlags."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    denom = (x * x).sum()
    return np.array([(x[k:] * x[:-k]).sum() / denom for k in range(1, nlags + 1)])


def tail_counts(r: np.ndarray, ks=(3, 4, 5)) -> list[dict]:
    """Observed versus normal-expected counts of moves beyond k standard deviations."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    z = (r - r.mean()) / r.std(ddof=1)
    rows = []
    for k in ks:
        expected = r.size * 2 * float(normal_sf(k))
        rows.append({"threshold_sd": k, "observed": int((np.abs(z) > k).sum()), "normal_expected": expected})
    return rows


# ---------------------------------------------------------------------------
# Regressions with several standard-error choices
# ---------------------------------------------------------------------------


def ols_all_se(Y: np.ndarray, F: np.ndarray, nw_lags: int | None = None) -> dict:
    """Regress each column of Y on [1, F]; return coefficients and three sets of standard errors.

    Y  (T, N) dependent variables, e.g. 25 portfolio excess returns
    F  (T, L) regressors shared by all, e.g. factor returns
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float).T).T if np.ndim(Y) == 1 else np.asarray(Y, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])
    k = X.shape[1]
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ Y
    resid = Y - X @ beta
    s2 = (resid ** 2).sum(axis=0) / (T - k)
    se_classical = np.sqrt(np.outer(np.diag(xtx_inv), s2))
    white = ols_newey_west(Y, F, lags=0)
    lags = newey_west_lag(T) if nw_lags is None else nw_lags
    nw = ols_newey_west(Y, F, lags=lags)
    return {"coef": beta, "resid": resid, "se_classical": se_classical, "se_white": white["se"],
            "se_nw": nw["se"], "nw_lags": lags}


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(T: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of time indices 0..T-1 in blocks of geometric length (mean `mean_block`).

    Blocks wrap around the end of the sample (Politis and Romano 1994).
    """
    p = 1.0 / mean_block
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(T)
    new_block = rng.random(T) < p
    starts = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % T
    return idx


def bootstrap_coefficients(Y: np.ndarray, F: np.ndarray, n_boot: int = 2000, mean_block: float = 12,
                           seed: int = 0) -> np.ndarray:
    """Stationary-bootstrap draws of regression coefficients, shape (n_boot, 1+L, N).

    Rows of (Y, F) are resampled together, preserving the relation between
    portfolios and factors and any short-run dependence within blocks.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    T = Y.shape[0]
    X = np.column_stack([np.ones(T), F])
    out = np.empty((n_boot, X.shape[1], Y.shape[1]))
    for b in range(n_boot):
        i = stationary_bootstrap_indices(T, mean_block, rng)
        Xb, Yb = X[i], Y[i]
        out[b] = np.linalg.solve(Xb.T @ Xb, Xb.T @ Yb)
    return out


# ---------------------------------------------------------------------------
# GRS test
# ---------------------------------------------------------------------------


def grs_test(Y: np.ndarray, F: np.ndarray) -> dict:
    """Gibbons-Ross-Shanken test that all N intercepts are zero.

    Uses maximum-likelihood (divide-by-T) covariance estimates, as in the
    original paper. Returns the statistic, its F(N, T-N-L) p-value, and the
    average absolute alpha.
    """
    from scipy import stats as st

    Y = np.asarray(Y, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    T, N = Y.shape
    L = F.shape[1]
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.solve(X.T @ X, X.T @ Y)
    alpha = B[0]
    E = Y - X @ B
    sigma = E.T @ E / T
    fbar = F.mean(axis=0)
    omega = (F - fbar).T @ (F - fbar) / T
    quad_a = float(alpha @ np.linalg.solve(sigma, alpha))
    quad_f = float(fbar @ np.linalg.solve(omega, fbar))
    stat = (T / N) * ((T - N - L) / (T - L - 1)) * quad_a / (1 + quad_f)
    p = float(st.f.sf(stat, N, T - N - L))
    return {"grs": float(stat), "p_value": p, "df1": N, "df2": T - N - L,
            "mean_abs_alpha_annual": float(np.abs(alpha).mean() * 12)}


# ---------------------------------------------------------------------------
# Fama-MacBeth
# ---------------------------------------------------------------------------


def fama_macbeth(Y: np.ndarray, F: np.ndarray, names: list[str] | None = None, nw_lags: int = 0) -> dict:
    """Two-pass Fama-MacBeth regression.

    Pass 1: full-sample time-series betas of each test asset on the factors.
    Pass 2: each month, a cross-sectional regression of returns on [1, betas].
    Reports average slopes (risk premia), Fama-MacBeth standard errors
    (optionally Newey-West on the slope series), and Shanken-corrected
    standard errors. For traded factors, compare each premium with the
    factor's own average return.
    """
    Y = np.asarray(Y, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    T, N = Y.shape
    L = F.shape[1]
    names = names or [f"f{i + 1}" for i in range(L)]
    X = np.column_stack([np.ones(T), F])
    betas = np.linalg.solve(X.T @ X, X.T @ Y)[1:].T  # (N, L)
    Z = np.column_stack([np.ones(N), betas])  # (N, 1+L)
    lam_t = np.linalg.solve(Z.T @ Z, Z.T @ Y.T).T  # (T, 1+L)
    lam = lam_t.mean(axis=0)
    if nw_lags:
        se_fm = ols_newey_west(lam_t, np.zeros((T, 0)), lags=nw_lags)["se"][0]
    else:
        se_fm = lam_t.std(axis=0, ddof=1) / math.sqrt(T)
    sigma_f = np.atleast_2d(np.cov(F, rowvar=False, ddof=0))
    lam_f = lam[1:]
    c = float(lam_f @ np.linalg.solve(sigma_f, lam_f))
    var_sh = (1 + c) * se_fm ** 2
    var_sh[1:] += np.diag(sigma_f) / T
    fitted = Z @ lam
    avg = Y.mean(axis=0)
    r2 = 1 - ((avg - fitted) ** 2).sum() / ((avg - avg.mean()) ** 2).sum()
    labels = ["intercept"] + list(names)
    return {"labels": labels, "lambda": lam, "se_fm": se_fm, "se_shanken": np.sqrt(var_sh),
            "t_fm": lam / se_fm, "t_shanken": lam / np.sqrt(var_sh), "shanken_c": c,
            "factor_means": np.concatenate([[np.nan], F.mean(axis=0)]), "cross_sectional_r2": float(r2),
            "betas": betas}


# ---------------------------------------------------------------------------
# Long-horizon regressions
# ---------------------------------------------------------------------------


def long_horizon_tstats(r: np.ndarray, X: np.ndarray, k: int, nw_lags: int | None = None) -> dict:
    """t-statistics for the slope in regressions of k-period future returns on x_t.

    r  (T,) one-period returns
    X  (T, R) R candidate predictors (e.g. R simulated series), x_t known at t
    k  horizon: the dependent variable is r_{t+1} + ... + r_{t+k}

    Returns slopes, R-squared and three t-statistics per predictor:
      classical    OLS formula, ignores overlap
      newey_west   HAC with `nw_lags` lags (default k)
      hodrick      Hodrick (1992) 1B standard errors under the null of no
                   predictability, which avoid overlapping errors
    """
    r = np.asarray(r, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T = r.size
    n = T - k
    csum = np.concatenate([[0.0], np.cumsum(r)])
    y = csum[1 + k:T + 1] - csum[1:n + 1]  # y_t = r_{t+1} + ... + r_{t+k}, t = 0..n-1
    x = X[:n]
    xd = x - x.mean(axis=0)
    yd = y - y.mean()
    sxx = (xd ** 2).sum(axis=0)
    b = (xd * yd[:, None]).sum(axis=0) / sxx
    e = yd[:, None] - xd * b
    r2 = 1 - (e ** 2).sum(axis=0) / (yd ** 2).sum()
    se_cl = np.sqrt((e ** 2).sum(axis=0) / (n - 2) / sxx)

    lags = k if nw_lags is None else nw_lags
    g = xd * e
    S = (g ** 2).sum(axis=0)
    for lag in range(1, lags + 1):
        S += 2 * (1 - lag / (lags + 1)) * (g[lag:] * g[:-lag]).sum(axis=0)
    se_nw = np.sqrt(np.maximum(S, 0)) / sxx

    # Hodrick 1B: sum_t xd_t y_t = sum_s eps_s * (sum of xd_t over t in [s-k, s-1] within 0..n-1)
    eps = r - r[1:].mean()
    cx = np.vstack([np.zeros((1, xd.shape[1])), np.cumsum(xd, axis=0)])  # cx[j] = sum xd[0..j-1]
    s_idx = np.arange(1, T)
    lo = np.clip(s_idx - k, 0, n)
    hi = np.clip(s_idx, 0, n)
    window = cx[hi] - cx[lo]  # (T-1, R)
    S_h = ((eps[s_idx, None] * window) ** 2).sum(axis=0)
    se_h = np.sqrt(S_h) / sxx
    return {"slope": b, "r2": r2, "t_classical": b / se_cl, "t_newey_west": b / se_nw, "t_hodrick": b / se_h,
            "n_obs": n, "nw_lags": lags}


def ar1_predictors(T: int, n_series: int, phi: float, rng: np.random.Generator,
                   shocks: np.ndarray | None = None, rho: float = 0.0) -> np.ndarray:
    """Simulate n_series AR(1) predictors of length T, x_t = phi x_{t-1} + v_t.

    If `shocks` (standardized one-period return shocks, length T) and `rho`
    are given, each predictor's innovation v_{t+1} has correlation rho with
    the return shock in period t+1, as for valuation ratios (Stambaugh 1999).
    With rho = 0 the predictors are independent of returns by construction.
    """
    v = rng.standard_normal((T, n_series))
    if shocks is not None and rho != 0.0:
        z = np.asarray(shocks, dtype=float)[:, None]
        v = rho * z + math.sqrt(1 - rho ** 2) * v
    x = np.empty((T, n_series))
    x[0] = v[0] / math.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + v[t]
    return x
