"""Correlation in calm and stressed markets (Chapter 10).

* bond_returns_from_yields - monthly total returns on a constant-maturity par bond from a yield series
* exceedance_corr, normal_exceedance - Longin-Solnik / Ang-Chen exceedance correlations and their
  bivariate-normal benchmark
* forbes_rigobon - correct a conditional correlation for the higher variance of the conditioning variable
* tail_dependence, normal_tail_dependence - probability that one asset is in its worst p-tail when the other is
* avg_pairwise_corr, rolling_corr, diversification_ratio
* simulated_assets - fake data with calm and stress regimes and an inflation-dependent stock-bond correlation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bond returns
# ---------------------------------------------------------------------------


def par_bond_price(coupon, yld, years):
    """Price per 1 of face of a bond paying semiannual coupons (annual rate `coupon`) at annual yield `yld`."""
    coupon, yld, years = (np.asarray(a, float) for a in (coupon, yld, years))
    n = 2 * years
    h = yld / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(h) > 1e-12, (1 - (1 + h) ** (-n)) / h, n)
    return coupon / 2 * annuity + (1 + h) ** (-n)


def bond_returns_from_yields(y: pd.Series, maturity: float = 10.0) -> pd.Series:
    """Monthly total return on a par bond: buy a `maturity`-year bond at par at the end of month t-1 (coupon equal
    to that month's yield), sell it at the end of month t, a month shorter, at month t's yield. `y` is in percent."""
    yy = y.astype(float) / 100
    prev = yy.shift(1)
    price = par_bond_price(prev, yy, maturity - 1 / 12)
    r = pd.Series(price - 1 + prev / 12, index=y.index, name="bond")
    return r.dropna()


# ---------------------------------------------------------------------------
# Conditional correlations
# ---------------------------------------------------------------------------


def exceedance_corr(x, y, q: float, min_obs: int = 15) -> float:
    """Correlation of x and y over observations where both are below their q-quantile (q < 0.5) or both above it
    (q > 0.5). Longin and Solnik (2001); Ang and Chen (2002)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    qx, qy = np.quantile(x, q), np.quantile(y, q)
    m = (x <= qx) & (y <= qy) if q < 0.5 else (x >= qx) & (y >= qy)
    if m.sum() < min_obs:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def exceedance_curve(x, y, qs, min_obs: int = 15) -> np.ndarray:
    return np.array([exceedance_corr(x, y, q, min_obs) for q in qs])


def normal_exceedance(rho: float, qs, n: int = 400_000, seed: int = 0) -> np.ndarray:
    """Exceedance correlations a bivariate normal with correlation rho would produce (computed by simulation)."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return exceedance_curve(z1, z2, qs, min_obs=30)


def forbes_rigobon(rho_c, var_high, var_low):
    """Forbes and Rigobon (2002): the unconditional correlation implied by a correlation rho_c measured in a period
    when the conditioning variable's variance is var_high instead of var_low."""
    delta = np.asarray(var_high, float) / np.asarray(var_low, float) - 1
    rho_c = np.asarray(rho_c, float)
    return rho_c / np.sqrt(1 + delta * (1 - rho_c ** 2))


def tail_dependence(x, y, p: float = 0.1) -> float:
    """P(y in its worst p-tail | x in its worst p-tail)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    bx, by = x <= np.quantile(x, p), y <= np.quantile(y, p)
    return float((bx & by).sum() / max(bx.sum(), 1))


def normal_tail_dependence(rho: float, p: float = 0.1, n: int = 400_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return tail_dependence(z1, z2, p)


def avg_pairwise_corr(R: pd.DataFrame) -> float:
    c = R.corr().to_numpy()
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def rolling_corr(x: pd.Series, y: pd.Series, window: int = 36) -> pd.Series:
    return x.rolling(window, min_periods=window).corr(y)


def diversification_ratio(R: pd.DataFrame, w) -> float:
    """Weighted average of asset volatilities divided by portfolio volatility (1 = no diversification)."""
    w = np.asarray(w, float)
    S = R.cov().to_numpy()
    return float(w @ np.sqrt(np.diag(S)) / np.sqrt(w @ S @ w))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulated_assets(n_months: int = 1188, seed: int = 0, start: str = "1926-07") -> dict:
    """Fake monthly excess returns for US stocks, international stocks, ten industries, a 10-year bond and oil, plus
    inflation. Markets switch between calm and stress regimes; in stress, volatility triples and the true equity
    correlation rises modestly. The stock-bond correlation is positive when inflation is high and negative when it is
    low. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    # regime: Markov chain, stress about 15% of months
    stress = np.zeros(n_months, bool)
    for t in range(1, n_months):
        p = 0.75 if stress[t - 1] else 0.04
        stress[t] = rng.random() < p
    # inflation: persistent AR(1) around 3% with a high-inflation episode
    infl = np.zeros(n_months)
    infl[0] = 0.03
    for t in range(1, n_months):
        infl[t] = 0.03 + 0.97 * (infl[t - 1] - 0.03) + 0.004 * rng.standard_normal()
    infl += np.where((np.arange(n_months) > n_months * 0.4) & (np.arange(n_months) < n_months * 0.55), 0.04, 0.0)
    vol_m = np.where(stress, 0.09, 0.035)
    mkt = 0.006 + vol_m * rng.standard_normal(n_months) - np.where(stress, 0.01, 0.0)
    rho_intl = np.where(stress, 0.75, 0.6)
    z = rng.standard_normal(n_months)
    intl = 0.004 + vol_m * 1.05 * (rho_intl * (mkt - mkt.mean()) / vol_m + np.sqrt(1 - rho_intl ** 2) * z)
    inds = {}
    for i in range(10):
        beta = 0.7 + 0.06 * i
        inds[f"Ind{i + 1}"] = 0.001 + beta * mkt + vol_m * 0.55 * rng.standard_normal(n_months)
    sb = np.clip((infl - 0.03) * 12, -0.5, 0.6)          # stock-bond correlation driven by inflation
    zb = rng.standard_normal(n_months)
    bond = 0.0015 + 0.02 * (sb * (mkt - 0.006) / vol_m + np.sqrt(1 - sb ** 2) * zb)
    oil = 0.002 + 0.08 * (0.25 * (mkt - 0.006) / vol_m + np.sqrt(1 - 0.25 ** 2) * rng.standard_normal(n_months))
    df = pd.DataFrame({"US stocks": mkt, "Intl stocks": intl, "Bonds": bond, "Oil": oil}, index=idx)
    df.loc[df.index < pd.Period("1990-07", "M"), "Intl stocks"] = np.nan
    df.loc[df.index < pd.Period("1953-05", "M"), "Bonds"] = np.nan
    df.loc[df.index < pd.Period("1946-02", "M"), "Oil"] = np.nan
    df.index.name = "date"
    return {"assets": df, "industries": pd.DataFrame(inds, index=idx), "inflation": pd.Series(infl, index=idx,
            name="inflation"), "stress": pd.Series(stress, index=idx), "simulated": True}
