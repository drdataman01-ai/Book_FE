"""Stochastic discount factors estimated from many assets (Chapter 22).

* Readers for the Jensen-Kelly-Pedersen (JKP) factor returns and their theme clusters (jkpfactors.com)
* Five characteristics of each asset, known at the end of each month, ranked across assets
* Characteristic-managed portfolios: linear, random-feature and linear-attention (Kelly, Kuznetsov,
  Malamud and Xu 2025, Equation w_t = (X_t W X_t') X_t lambda)
* The shrinkage SDF of Kozak, Nagel and Santosh (2020): w = (Sigma + gamma I)^-1 mu, gamma by cross-validation
* Walk-forward estimation: estimate on the previous `window` months, hold for the next month

Every SDF portfolio return at month t+1 uses weights computed from data up to the end of month t.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataUnavailable

CHARACTERISTICS = ["ret_1", "ret_3", "ret_12", "vol_12", "avg_36"]


# ---------------------------------------------------------------------------
# JKP data
# ---------------------------------------------------------------------------


def read_jkp_factors(path, weighting: str | None = None) -> pd.DataFrame:
    """JKP factor returns as a wide table (months x factors), decimals.

    Accepts the long CSV from jkpfactors.com (columns include name, date, ret and, often, location, weighting)
    or an already wide CSV with a date column. With several weighting schemes in the file, `weighting` picks
    one (for example 'vw_cap'); by default the first one listed is used.
    """
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}")
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    if {"name", "date", "ret"} <= set(df.columns):
        if "location" in df.columns:
            locs = df["location"].astype(str).str.lower()
            if (locs == "usa").any():
                df = df[locs == "usa"]
        if "weighting" in df.columns:
            w = weighting or df["weighting"].iloc[0]
            df = df[df["weighting"] == w]
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M")
        wide = df.pivot_table(index="date", columns="name", values="ret", aggfunc="mean").sort_index()
    else:
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M")
        wide = df.set_index(date_col).apply(pd.to_numeric, errors="coerce").sort_index()
    wide.index.name = "date"
    if wide.abs().stack().median() > 0.5:  # given in percent
        wide = wide / 100
    return wide


def read_jkp_clusters(path) -> pd.Series:
    """Factor name -> theme cluster, from the JKP cluster file (first column: factor, a column named like
    'cluster' or the second column: theme)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    name_col = next((c for c in df.columns if c in ("characteristic", "name", "factor")), df.columns[0])
    clus_col = next((c for c in df.columns if "cluster" in c or "theme" in c), df.columns[1])
    return pd.Series(df[clus_col].astype(str).str.strip().to_numpy(), index=df[name_col].astype(str).str.strip())


# ---------------------------------------------------------------------------
# Characteristics
# ---------------------------------------------------------------------------


def characteristics(R: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Five characteristics of each asset known at the end of month t: its return over the past 1, 3 and 12
    months, its volatility over the past 12 months and its average return over the past 36 months."""
    return {"ret_1": R,
            "ret_3": R.rolling(3, min_periods=3).sum(),
            "ret_12": R.rolling(12, min_periods=12).sum(),
            "vol_12": R.rolling(12, min_periods=12).std(),
            "avg_36": R.rolling(36, min_periods=24).mean()}


def rank_standardize(x: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks mapped to [-0.5, 0.5]; missing values set to 0 (the median)."""
    out = np.zeros_like(x, dtype=float)
    ok = np.isfinite(x)
    n = ok.sum()
    if n > 1:
        r = pd.Series(x[ok]).rank(method="average").to_numpy()
        out[ok] = (r - 1) / (n - 1) - 0.5
    return out


def characteristic_array(R: pd.DataFrame) -> np.ndarray:
    """(T, N, 5) ranked characteristics; row t is known at the end of month t."""
    ch = characteristics(R)
    T, N = R.shape
    X = np.zeros((T, N, len(CHARACTERISTICS)))
    for k, name in enumerate(CHARACTERISTICS):
        v = ch[name].to_numpy()
        for t in range(T):
            X[t, :, k] = rank_standardize(v[t])
    return X


# ---------------------------------------------------------------------------
# Weight maps: characteristics at t -> portfolio weights (per parameter) at t
# ---------------------------------------------------------------------------


def features_linear(X_t: np.ndarray) -> np.ndarray:
    """No attention: w_i = x_i' lambda. Returns (N, 5)."""
    return X_t


def random_feature_map(n_features: int = 64, n_char: int = 5, scale: float = 2.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    Om = rng.normal(0, scale, (n_char, n_features // 2))

    def f(X_t: np.ndarray) -> np.ndarray:
        z = X_t @ Om
        return np.hstack([np.sin(z), np.cos(z)]) / math.sqrt(n_features)
    return f


def features_linear_attention(X_t: np.ndarray) -> np.ndarray:
    """Linear attention: w_i = sum_{a,b,c} theta_abc x_ia S_bc with S = X'X / N, the products that make up
    w = (X W X') X lambda with theta_abc = W_ab lambda_c. Returns (N, 125)."""
    N, K = X_t.shape
    S = X_t.T @ X_t / N
    return (X_t[:, :, None, None] * S[None, None, :, :]).reshape(N, K * K * K)


def attention_scores(X_t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """A[i, j] = sum_abc theta_abc x_ia x_jb x_jc / N: how much asset j's characteristics move asset i's weight.
    Rows sum to asset i's weight."""
    N, K = X_t.shape
    Th = theta.reshape(K, K, K)
    left = np.einsum("ia,abc->ibc", X_t, Th)
    return np.einsum("ibc,jb,jc->ij", left, X_t, X_t) / N


# ---------------------------------------------------------------------------
# Shrinkage tangency with cross-validated penalty
# ---------------------------------------------------------------------------


def ridge_tangency(F: np.ndarray, gamma: float) -> np.ndarray:
    mu = F.mean(axis=0)
    S = np.cov(F, rowvar=False) if F.shape[1] > 1 else np.atleast_2d(F.var(ddof=1))
    return np.linalg.solve(S + gamma * np.eye(len(mu)), mu)


def _sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else -np.inf


def cv_gamma(F: np.ndarray, grid=(1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0), folds: int = 5) -> float:
    """Penalty chosen to maximize the Sharpe ratio of the held-out-fold SDF returns (contiguous folds).
    The grid is relative to the average variance of the managed portfolios."""
    T = F.shape[0]
    scale = float(np.mean(F.var(axis=0, ddof=1))) or 1e-6
    edges = np.linspace(0, T, folds + 1).astype(int)
    best, best_g = -np.inf, grid[0] * scale
    for g in grid:
        oos = []
        for k in range(folds):
            test = np.zeros(T, bool)
            test[edges[k]:edges[k + 1]] = True
            w = ridge_tangency(F[~test], g * scale)
            oos.append(F[test] @ w)
        s = _sharpe(np.concatenate(oos))
        if s > best:
            best, best_g = s, g * scale
    return best_g


# ---------------------------------------------------------------------------
# Walk-forward SDF
# ---------------------------------------------------------------------------


def managed_returns(X: np.ndarray, Rnext: np.ndarray, fmap) -> np.ndarray:
    """F[t] = Phi(X_t)' R_{t+1}: returns of the characteristic-managed portfolios formed at t, earned in t+1.
    Missing returns count as zero (the asset is not held)."""
    T = X.shape[0]
    R0 = np.nan_to_num(Rnext)
    first = fmap(X[0])
    out = np.full((T, first.shape[1]), np.nan)
    for t in range(T):
        out[t] = fmap(X[t]).T @ R0[t] / X.shape[1]
    return out


def walk_forward_sdf(F: np.ndarray, window: int = 60, cv: bool = True, gamma_fixed: float | None = None,
                     start: int | None = None) -> dict:
    """Out-of-sample SDF portfolio returns. F[t] is the managed return formed at t and earned in t+1 (known at the
    end of t+1). The coefficients used at t are estimated on F[t-window .. t-1], all known by the end of t."""
    T, K = F.shape
    ret = np.full(T, np.nan)
    thetas = np.full((T, K), np.nan)
    gammas = np.full(T, np.nan)
    s = window if start is None else max(start, window)
    for t in range(s, T):
        Fw = F[t - window:t]
        Fw = Fw[np.isfinite(Fw).all(axis=1)]
        if len(Fw) < window * 0.8 or not np.isfinite(F[t]).all():
            continue
        g = gamma_fixed if gamma_fixed is not None else (cv_gamma(Fw) if cv else 1e-3 * np.mean(Fw.var(axis=0)))
        th = ridge_tangency(Fw, g)
        nrm = np.sqrt(th @ np.cov(Fw, rowvar=False) @ th) if K > 1 else abs(th[0]) * Fw.std()
        th = th / nrm if nrm > 0 else th  # unit in-sample volatility, so weights are comparable over time
        thetas[t] = th
        gammas[t] = g
        ret[t] = F[t] @ th
    return {"ret": ret, "theta": thetas, "gamma": gammas}


def asset_weights(X: np.ndarray, thetas: np.ndarray, fmap) -> np.ndarray:
    """(T, N) weights on the assets implied by the coefficients at each date."""
    T, N, _ = X.shape
    W = np.full((T, N), np.nan)
    for t in range(T):
        if np.isfinite(thetas[t]).all():
            W[t] = fmap(X[t]) @ thetas[t] / N
    return W


def turnover(W: np.ndarray) -> np.ndarray:
    """Monthly change in weights, with weights scaled to unit gross exposure each month."""
    g = np.nansum(np.abs(W), axis=1, keepdims=True)
    Wn = np.where(g > 0, W / g, np.nan)
    return np.r_[np.nan, np.nansum(np.abs(np.diff(Wn, axis=0)), axis=1)]
