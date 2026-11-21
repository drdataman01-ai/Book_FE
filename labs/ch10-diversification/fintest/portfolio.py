"""Portfolio construction and out-of-sample evaluation (Chapter 9).

Weight rules (each takes a T x N window of excess returns and returns N weights summing to 1):
* equal weight (1/N)
* mean-variance tangency from sample moments, unconstrained and long-only
* minimum variance from the sample covariance, unconstrained and long-only
* minimum variance with Ledoit-Wolf (2004) shrinkage toward a scaled identity
* risk parity (inverse volatility) and equal risk contribution
* hierarchical risk parity (Lopez de Prado 2016)

Plus a walk-forward backtest with turnover and costs, the certainty-equivalent return, and the Jobson-Korkie test
of a Sharpe-ratio difference with the Memmel (2003) correction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward mu * I. Returns (covariance, intensity)."""
    X = np.asarray(X, float)
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = np.sum((S - F) ** 2)
    b2_bar = (np.sum(np.sum(Xc ** 2, axis=1) ** 2) - T * np.sum(S ** 2)) / T ** 2  # (1/T^2) sum ||x x' - S||^2
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S, float(delta)


# ---------------------------------------------------------------------------
# Weight rules
# ---------------------------------------------------------------------------


def w_equal(X):
    return np.full(X.shape[1], 1 / X.shape[1])


def _long_only_qp(cov, mu=None, gamma=None):
    N = cov.shape[0]
    if mu is None:
        f = lambda w: w @ cov @ w  # noqa: E731
    else:
        f = lambda w: -(w @ mu) + 0.5 * gamma * (w @ cov @ w)  # noqa: E731
    res = minimize(f, np.full(N, 1 / N), method="SLSQP", bounds=[(0, 1)] * N,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def w_min_variance(X, long_only=False, cov=None):
    cov = np.cov(X, rowvar=False) if cov is None else cov
    if long_only:
        return _long_only_qp(cov)
    ones = np.ones(cov.shape[0])
    w = np.linalg.solve(cov, ones)
    return w / w.sum()


def w_min_variance_lw(X):
    cov, _ = ledoit_wolf(X)
    return w_min_variance(X, cov=cov)


def w_tangency(X, long_only=False, gamma=5.0, max_gross=5.0):
    """Sample mean-variance portfolio. Unconstrained: tangency weights scaled to sum to 1 (gross exposure capped).
    Long-only: maximize mu'w - gamma/2 w'Sw with weights in [0, 1] summing to 1."""
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    if long_only:
        return _long_only_qp(cov, mu, gamma)
    w = np.linalg.solve(cov, mu)
    s = w.sum()
    if abs(s) < 1e-8 or s < 0:  # tangency undefined or on the inefficient branch: fall back to minimum variance
        return w_min_variance(X, cov=cov)
    w = w / s
    g = np.abs(w).sum()
    if g > max_gross:  # shrink toward 1/N until gross exposure is acceptable
        e = w_equal(X)
        lam = (max_gross - 1) / (g - 1)
        w = e + lam * (w - e)
    return w


def w_inverse_vol(X):
    v = X.std(axis=0, ddof=1)
    w = 1 / v
    return w / w.sum()


def w_equal_risk(X):
    """Equal risk contribution (Maillard, Roncalli and Teiletche 2010), solved by Newton-type iteration."""
    cov = np.cov(X, rowvar=False)
    N = cov.shape[0]
    f = lambda y: 0.5 * y @ cov @ y - np.sum(np.log(y)) / N  # noqa: E731
    res = minimize(f, w_inverse_vol(X), method="L-BFGS-B", bounds=[(1e-10, None)] * N)
    w = res.x
    return w / w.sum()


def w_hrp(X):
    """Hierarchical risk parity (Lopez de Prado 2016): cluster on correlation distance, then bisect with inverse
    variance allocations."""
    cov = np.cov(X, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(len(order))

    def cluster_var(idx):
        c = cov[np.ix_(idx, idx)]
        ivp = 1 / np.diag(c)
        ivp /= ivp.sum()
        return ivp @ c @ ivp

    clusters = [list(order)]
    while clusters:
        new = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            a, b = c[:h], c[h:]
            va, vb = cluster_var(a), cluster_var(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            new += [a, b]
        clusters = new
    return w / w.sum()


RULES = {
    "1/N": w_equal,
    "Mean-variance (sample)": lambda X: w_tangency(X, False),
    "Mean-variance, long-only": lambda X: w_tangency(X, True),
    "Minimum variance (sample)": lambda X: w_min_variance(X, False),
    "Minimum variance, long-only": lambda X: w_min_variance(X, True),
    "Minimum variance, Ledoit–Wolf": w_min_variance_lw,
    "Risk parity (inverse volatility)": w_inverse_vol,
    "Equal risk contribution": w_equal_risk,
    "Hierarchical risk parity": w_hrp,
}


# ---------------------------------------------------------------------------
# Backtest and statistics
# ---------------------------------------------------------------------------


def walk_forward(R: pd.DataFrame, rules: dict, window: int = 120, every: int = 1, cost: float = 0.0) -> dict:
    """Out-of-sample monthly excess returns for each rule.

    At each rebalancing date t, weights are estimated from the previous `window` months (t-window .. t-1) and held for
    `every` months, drifting with returns in between. Turnover is the sum of absolute differences between the new
    weights and the drifted old ones; cost is charged per unit of turnover.
    """
    X = R.to_numpy(float)
    T, N = X.shape
    res = {}
    for name, rule in rules.items():
        rets, tos, ws = [], [], []
        w_cur = None
        for t in range(window, T):
            to = 0.0
            if (t - window) % every == 0:
                w_new = rule(X[t - window:t])
                to = float(np.abs(w_new - w_cur).sum()) if w_cur is not None else 0.0
                w_cur = w_new
            rets.append(float(w_cur @ X[t]) - cost * to)
            tos.append(to)
            ws.append(w_cur.copy())
            g = w_cur * (1 + X[t])
            if abs(g.sum()) > 1e-8:
                w_cur = g / g.sum()
        res[name] = {"ret": pd.Series(rets, index=R.index[window:]), "turnover": np.array(tos), "weights": np.array(ws)}
    return res


def certainty_equivalent(r: np.ndarray, gamma: float = 1.0) -> float:
    """Annualized mean minus gamma/2 times variance (DeMiguel, Garlappi and Uppal 2009)."""
    r = np.asarray(r, float)
    return float(12 * (r.mean() - 0.5 * gamma * r.var(ddof=1)))


def sharpe_diff_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Jobson-Korkie test, with Memmel's correction, that two Sharpe ratios are equal. Returns z and two-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = len(a)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sab = np.cov(a, b)[0, 1]
    var = (2 * sa ** 2 * sb ** 2 - 2 * sa * sb * sab + 0.5 * ma ** 2 * sb ** 2 + 0.5 * mb ** 2 * sa ** 2
           - ma * mb / (sa * sb) * sab ** 2) / T
    z = (sb * ma - sa * mb) / math.sqrt(var) if var > 0 else float("nan")
    return {"z": float(z), "p": float(2 * norm.sf(abs(z)))}


def summarize(res: dict, bench: str = "1/N", gamma: float = 1.0) -> pd.DataFrame:
    rows = []
    b = res[bench]["ret"].to_numpy()
    for name, r in res.items():
        x = r["ret"].to_numpy()
        w = np.exp(np.cumsum(np.log1p(x)))
        test = sharpe_diff_test(x, b) if name != bench else {"z": float("nan"), "p": float("nan")}
        rows.append({"rule": name, "mean_annual": 12 * x.mean(), "vol_annual": x.std(ddof=1) * math.sqrt(12),
                     "sharpe": x.mean() / x.std(ddof=1) * math.sqrt(12), "ce": certainty_equivalent(x, gamma),
                     "turnover_monthly": float(r["turnover"][1:].mean()) if len(r["turnover"]) > 1 else 0.0,
                     "max_weight": float(np.abs(r["weights"]).max()), "min_weight": float(r["weights"].min()),
                     "max_drawdown": float((w / np.maximum.accumulate(w) - 1).min()), "sharpe_vs_1N_z": test["z"],
                     "p": test["p"], "months": len(x)})
    return pd.DataFrame(rows)


def simulated_assets(n_months: int = 1188, n_assets: int = 10, seed: int = 0, start: str = "1926-07") -> pd.DataFrame:
    """Fake monthly excess returns from a one-factor model with similar true Sharpe ratios. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.7, 1.3, n_assets)
    idio = rng.uniform(0.02, 0.05, n_assets)
    f = 0.006 + 0.045 * rng.standard_t(5, n_months) / math.sqrt(5 / 3)
    X = beta[None, :] * f[:, None] + idio[None, :] * rng.standard_normal((n_months, n_assets))
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    return pd.DataFrame(X, index=idx, columns=[f"A{i + 1}" for i in range(n_assets)])
