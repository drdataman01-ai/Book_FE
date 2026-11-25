"""Multiple testing and backtest overfitting.

* p-value corrections: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli
* Probabilistic and deflated Sharpe ratios (Bailey and Lopez de Prado 2012, 2014)
* Effective number of trials from the correlation of strategy returns
* Probability of backtest overfitting by combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado and Zhu 2017)

All Sharpe ratios here are per period (not annualized) unless a function
says otherwise.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .stats import normal_cdf, normal_ppf, normal_sf

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# p-value corrections. Each returns a boolean array: True = rejected (discovery).
# ---------------------------------------------------------------------------


def one_sided_pvalues(t: np.ndarray) -> np.ndarray:
    """p-values for H0: mean <= 0 against mean > 0."""
    return np.asarray(normal_sf(np.asarray(t, dtype=float)), dtype=float)


def bonferroni(p: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p <= alpha / p.size


def holm(p: np.ndarray, alpha: float) -> np.ndarray:
    """Holm (1979) step-down procedure; controls the family-wise error rate."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    reject = np.zeros(n, dtype=bool)
    for k, i in enumerate(order):
        if p[i] <= alpha / (n - k):
            reject[i] = True
        else:
            break
    return reject


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up procedure; controls the false discovery rate."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        k = np.flatnonzero(passed).max()
        reject[order[:k + 1]] = True
    return reject


def benjamini_yekutieli(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Yekutieli (2001): BH with q divided by the harmonic sum; valid under any dependence."""
    n = np.asarray(p).size
    return benjamini_hochberg(p, q / np.sum(1.0 / np.arange(1, n + 1)))


def apply_corrections(t: np.ndarray, alpha: float = 0.025) -> dict[str, np.ndarray]:
    """Discoveries under each method, for one-sided tests at level `alpha`.

    alpha = 0.025 one-sided is equivalent to the usual two-sided 5% test,
    so the uncorrected threshold is t > 1.96.
    """
    t = np.asarray(t, dtype=float)
    p = one_sided_pvalues(t)
    return {
        "No correction (t > 1.96)": p <= alpha,
        "t > 3": t > 3.0,
        "Benjamini-Hochberg": benjamini_hochberg(p, alpha),
        "Benjamini-Yekutieli": benjamini_yekutieli(p, alpha),
        "Holm": holm(p, alpha),
        "Bonferroni": bonferroni(p, alpha),
    }


# ---------------------------------------------------------------------------
# Probabilistic and deflated Sharpe ratios
# ---------------------------------------------------------------------------


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0,
                         kurt: float = 3.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds `sr_benchmark`.

    `kurt` is ordinary (not excess) kurtosis, 3 for a normal distribution.
    """
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(normal_cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom))


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe ratio among n_trials with no skill.

    sr_variance is the variance of the estimated Sharpe ratios across trials.
    """
    if n_trials <= 1:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(sr_variance) * ((1 - g) * normal_ppf(1 - 1 / n_trials)
                                     + g * normal_ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sr: float, n_obs: int, n_trials: float, sr_variance: float, skew: float = 0.0,
                    kurt: float = 3.0) -> dict:
    """Deflated Sharpe ratio of the best of n_trials strategies."""
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return {"sr0": sr0, "dsr": probabilistic_sharpe(sr, sr0, n_obs, skew, kurt)}


def effective_trials(returns: np.ndarray, method: str = "variance", share: float = 0.90) -> float:
    """Effective number of independent strategies from their return correlations.

    method="variance"       number of principal components needed to explain
                            `share` of the variance of the correlation matrix
                            (the default; about 0.9 N for uncorrelated noise)
    method="participation"  participation ratio (sum lambda)^2 / sum lambda^2;
                            N for uncorrelated strategies, 1 for perfectly
                            correlated ones, and very small whenever one
                            common component dominates

    Both are simple stand-ins for the clustering approach Bailey and Lopez de
    Prado recommend. The effective number of trials is a judgment, not a
    measurement, and the deflated Sharpe ratio is sensitive to it.
    """
    R = np.asarray(returns, dtype=float)
    keep = R.std(axis=0) > 0
    C = np.corrcoef(R[:, keep], rowvar=False)
    lam = np.sort(np.clip(np.linalg.eigvalsh(C), 0, None))[::-1]
    if method == "participation":
        return float(lam.sum() ** 2 / (lam ** 2).sum())
    if method == "variance":
        return float(np.searchsorted(np.cumsum(lam) / lam.sum(), share) + 1)
    raise ValueError("method must be 'variance' or 'participation'")


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------


def cscv_pbo(returns: np.ndarray, n_blocks: int = 16, max_splits: int | None = None,
             seed: int = 0) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    returns  (T, N) per-period returns of N strategies over the same T periods
    n_blocks even number of equal blocks S; every choice of S/2 blocks is a
             training set and the rest the test set (C(S, S/2) splits)

    For each split, the strategy with the best training Sharpe ratio is ranked
    among all strategies in the test set; its relative rank w in (0, 1) gives
    the logit ln(w / (1 - w)). PBO is the share of splits with logit < 0.
    """
    R = np.asarray(returns, dtype=float)
    T, N = R.shape
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    L = T // n_blocks
    R = R[:L * n_blocks].reshape(n_blocks, L, N)
    s1 = R.sum(axis=1)  # (S, N)
    s2 = (R ** 2).sum(axis=1)
    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)))
    if max_splits is not None and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = combos[rng.choice(len(combos), size=max_splits, replace=False)]
    M = np.zeros((len(combos), n_blocks))
    M[np.arange(len(combos))[:, None], combos] = 1.0
    n_half = L * n_blocks / 2

    def sharpe(mask):
        mean = (mask @ s1) / n_half
        var = (mask @ s2) / n_half - mean ** 2
        return mean / np.sqrt(np.maximum(var, 1e-18))

    sr_in, sr_out = sharpe(M), sharpe(1.0 - M)
    best = sr_in.argmax(axis=1)
    rows = np.arange(len(combos))
    out_best = sr_out[rows, best]
    rank = (sr_out < out_best[:, None]).sum(axis=1) + 1  # 1 = worst, N = best
    w = rank / (N + 1)
    logits = np.log(w / (1 - w))
    return {"pbo": float((logits < 0).mean()), "logits": logits, "n_splits": len(combos),
            "sr_in_best": sr_in[rows, best], "sr_out_best": out_best,
            "median_sr_out": np.median(sr_out, axis=1)}
