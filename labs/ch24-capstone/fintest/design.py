"""Research design tools (Chapter 24).

* years_for_sharpe, power_for_sharpe - how long a sample is needed to detect a Sharpe ratio, analytically
* simulated_power - the same question by simulation, with fat tails and autocorrelation
* months_for_r2 - sample needed to detect a predictive-regression R-squared
* min_bayes_factor, posterior_null - Harvey's (2017) Bayesianized p-values
* specification_grid - run one test under every combination of researcher choices (a specification curve)
* noise_grid_experiment - how often a pure-noise signal looks significant somewhere in a grid of choices
* preregistration - write a dated, hashed pre-registration document
"""

from __future__ import annotations

import hashlib
import itertools
import math
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def years_for_sharpe(sharpe: float, t_crit: float = 1.96, power: float = 0.8) -> float:
    """Years of data for a one-sided test of a zero mean, at critical value t_crit, to reject with the given power
    when the true annualized Sharpe ratio is `sharpe` (normal returns, no autocorrelation): the t-statistic after
    T years is about sharpe x sqrt(T)."""
    z_b = stats.norm.ppf(power)
    return ((t_crit + z_b) / sharpe) ** 2


def power_for_sharpe(sharpe: float, years: float, t_crit: float = 1.96) -> float:
    return float(stats.norm.sf(t_crit - sharpe * math.sqrt(years)))


def simulated_power(sharpe: float, years: float, t_crit: float = 1.96, n_sims: int = 4000, df: float = 5.0,
                    rho: float = 0.0, periods: int = 12, seed: int = 0) -> float:
    """Share of simulated samples whose t-statistic exceeds t_crit. Monthly returns with Student-t(df) shocks and
    AR(1) autocorrelation rho; the t-statistic uses the naive standard error, as many backtests do."""
    rng = np.random.default_rng(seed)
    n = int(round(years * periods))
    mu = sharpe / math.sqrt(periods)
    e = rng.standard_t(df, (n_sims, n)) / math.sqrt(df / (df - 2)) if np.isfinite(df) else rng.standard_normal((n_sims, n))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + math.sqrt(1 - rho ** 2) * e[:, t]
    r = mu + x
    tstat = r.mean(1) / (r.std(1, ddof=1) / math.sqrt(n))
    return float((tstat > t_crit).mean())


def months_for_r2(r2: float, t_crit: float = 1.96, power: float = 0.8) -> float:
    """Observations for the slope of a one-predictor regression to be significant with the given power when the
    population R-squared is r2: the t-statistic is about sqrt(T x r2 / (1 - r2))."""
    z_b = stats.norm.ppf(power)
    return (t_crit + z_b) ** 2 * (1 - r2) / r2


def multiple_testing_threshold(n_tests: int, alpha: float = 0.05, method: str = "bonferroni") -> float:
    """t-statistic needed for two-sided family-wise significance across n independent tests."""
    if method == "bonferroni":
        return float(stats.norm.isf(alpha / 2 / n_tests))
    if method == "sidak":
        return float(stats.norm.isf((1 - (1 - alpha) ** (1 / n_tests)) / 2))
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Bayesianized p-values
# ---------------------------------------------------------------------------


def min_bayes_factor(t: float) -> float:
    """Minimum Bayes factor exp(-t^2 / 2): the most the data can favour the alternative over the null."""
    return math.exp(-t * t / 2)


def posterior_null(t: float, prior_odds_null: float) -> float:
    """Lower bound on the probability that the null is true after seeing t, given prior odds (null : alternative)."""
    b = min_bayes_factor(t) * prior_odds_null
    return b / (1 + b)


# ---------------------------------------------------------------------------
# Specification curves and forking paths
# ---------------------------------------------------------------------------


def specification_grid(func, grid: dict) -> pd.DataFrame:
    """Call func(**choice) for every combination in `grid` ({name: [options]}); func returns a dict with at least
    'estimate' and 't'. Returns one row per specification, sorted by estimate."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        choice = dict(zip(keys, combo))
        res = func(**choice)
        rows.append({**{k: str(v) for k, v in choice.items()}, **res})
    return pd.DataFrame(rows).sort_values("estimate").reset_index(drop=True)


def noise_grid_experiment(n_months: int = 720, n_assets: int = 100, n_trials: int = 300, seed: int = 0) -> dict:
    """Researchers test a signal that is pure noise, under 36 specifications: 3 sample starts x 3 portfolio cuts
    (tercile, quintile, decile) x 2 weightings (equal, by a random 'size') x 2 January treatments. Returns the share of
    trials in which the pre-chosen specification is significant, and in which at least one specification is."""
    rng = np.random.default_rng(seed)
    pre, any_sig, max_t = [], [], []
    starts = (0, n_months // 3, n_months // 2)
    for _ in range(n_trials):
        R = 0.01 + 0.06 * rng.standard_normal((n_months, n_assets))
        sig = rng.standard_normal((n_months, n_assets))           # the 'signal': unrelated to future returns
        size = np.exp(rng.normal(0, 1.5, n_assets))
        jan = (np.arange(n_months) % 12) == 0
        ts = []
        for s0 in starts:
            for q in (3, 5, 10):
                for vw in (False, True):
                    for nojan in (False, True):
                        rank = sig[s0:-1].argsort(1).argsort(1) / (n_assets - 1)
                        hi, lo = rank >= 1 - 1 / q, rank < 1 / q
                        w = size if vw else np.ones(n_assets)
                        rr = R[s0 + 1:]
                        lh = (rr * hi * w).sum(1) / (hi * w).sum(1) - (rr * lo * w).sum(1) / (lo * w).sum(1)
                        if nojan:
                            lh = lh[~jan[s0 + 1:]]
                        ts.append(lh.mean() / (lh.std(ddof=1) / math.sqrt(len(lh))))
        ts = np.array(ts)
        pre.append(abs(ts[0]) > 1.96)
        any_sig.append((np.abs(ts) > 1.96).any())
        max_t.append(np.abs(ts).max())
    return {"pre_specified_rate": float(np.mean(pre)), "any_spec_rate": float(np.mean(any_sig)),
            "max_abs_t": np.array(max_t), "n_specs": 36, "n_trials": n_trials}


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


TEMPLATE = """# Pre-registration: {title}

Date written: {written}
Author(s): {authors}

## 1. Claim
{claim}

## 2. Hypotheses
{hypotheses}

## 3. Data
{data}

## 4. Sample and exclusions (fixed before looking at results)
{sample}

## 5. Primary test
{primary}

## 6. Secondary tests and robustness grid
{secondary}

## 7. Power
{power}

## 8. Decision rules: what result would count as support, as failure, as inconclusive
{decision}

## 9. What will be reported regardless of outcome
{reporting}

## 10. Deviations
Any deviation from this plan will be listed and justified in the final report.

---
SHA-256 of the text above this line: {digest}
"""


def preregistration(fields: dict) -> str:
    """Fill the template and append a SHA-256 hash of the plan, so that later changes are detectable. Publish the hash
    (or the file) somewhere time-stamped, such as a public repository commit or OSF, before running the main test."""
    f = {"written": date.today().isoformat(), "authors": "", **fields}
    body = TEMPLATE.split("---")[0].format(**f)
    digest = hashlib.sha256(body.encode()).hexdigest()
    return body + "---\nSHA-256 of the text above this line: " + digest + "\n"


def verify_preregistration(text: str) -> bool:
    body, _, tail = text.partition("---\nSHA-256 of the text above this line: ")
    return hashlib.sha256(body.encode()).hexdigest() == tail.strip()
