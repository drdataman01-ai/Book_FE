"""Machine-learning return forecasts on a panel of portfolios (Chapter 20).

* build_panel - characteristics from returns, size and firm counts, rank-normalized each month (as in Gu, Kelly
  and Xiu 2020), with next month's excess return as the target
* make_models - OLS, OLS on three characteristics, elastic net, gradient-boosted trees, neural network
* walk_forward - expanding-window training with a validation block for tuning, refit every few years
* r2_oos, diebold_mariano, long_short, permutation_importance
* random_features_timing - Kelly, Malamud and Zhou (2024) style market timing with random Fourier features
* simulated_panel - fake portfolios with a small, partly nonlinear predictable component. TESTING ONLY.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

CHARS = ["size", "nfirms", "bm", "rev_1m", "mom_12_2", "lrev_36_13", "vol_12m", "beta_60m", "ivol_60m", "max_12m"]
MACRO = ["mkt_12m", "mktvol_12m"]
OLS3 = ["size", "bm", "mom_12_2"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _rank_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of each row mapped to [-1, 1]; missing values set to 0 (the median)."""
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1)
    scaled = (r.mul(n, axis=0) - 0.5).div(n, axis=0) * 2 - 1
    return scaled.fillna(0.0)


def build_panel(ret: pd.DataFrame, mkt: pd.Series, size: pd.DataFrame | None = None,
                nfirms: pd.DataFrame | None = None, bm: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long panel: one row per (month t, portfolio) with characteristics known at the end of t and the target
    `ret_next`, the excess return in month t+1. `ret` and `mkt` are monthly excess returns."""
    ret = ret.astype(float)
    feats = {}
    if size is not None:
        feats["size"] = np.log(size.where(size > 0))
    if nfirms is not None:
        feats["nfirms"] = np.log(nfirms.where(nfirms > 0))
    if bm is not None:
        feats["bm"] = np.log(bm.where(bm > 0))
    lr = np.log1p(ret)
    feats["rev_1m"] = ret
    feats["mom_12_2"] = np.expm1(lr.shift(1).rolling(11, min_periods=9).sum())
    feats["lrev_36_13"] = np.expm1(lr.shift(12).rolling(24, min_periods=18).sum())
    feats["vol_12m"] = ret.rolling(12, min_periods=10).std()
    feats["max_12m"] = ret.rolling(12, min_periods=10).max()
    m = mkt.reindex(ret.index)
    cov = ret.rolling(60, min_periods=24).cov(m)
    var = m.rolling(60, min_periods=24).var()
    beta = cov.div(var, axis=0)
    feats["beta_60m"] = beta
    resid = ret - beta.mul(m, axis=0)
    feats["ivol_60m"] = resid.rolling(60, min_periods=24).std()
    ranked = {k: _rank_scale(v) for k, v in feats.items()}
    for k in CHARS:
        if k not in ranked:
            ranked[k] = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    macro = pd.DataFrame({"mkt_12m": np.expm1(np.log1p(m).rolling(12).sum()), "mktvol_12m": m.rolling(12).std()})
    rows = []
    target = ret.shift(-1)
    valid = feats["beta_60m"].notna() & feats["lrev_36_13"].notna() & target.notna() & ret.notna()
    for col in ret.columns:
        df = pd.DataFrame({k: ranked[k][col] for k in CHARS}, index=ret.index)
        df[MACRO] = macro
        df["ret_next"] = target[col]
        df["asset"] = col
        rows.append(df[valid[col]])
    panel = pd.concat(rows).dropna(subset=MACRO + ["ret_next"])
    panel.index.name = "date"
    panel = panel.sort_index(kind="stable")
    # next month's return relative to the average asset that month: the cross-sectional part a long-short book earns
    panel["ret_next_dm"] = panel["ret_next"] - panel.groupby(level=0)["ret_next"].transform("mean")
    return panel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def make_models(fast: bool = False, seeds=(0, 1, 2)):
    """{name: (factory(params, seed) -> estimator, grid of params, feature list or None for all)}."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def enet(p, s):
        return make_pipeline(StandardScaler(), ElasticNet(alpha=p["alpha"], l1_ratio=0.5, max_iter=5000))

    def nn(p, s):
        # inputs and target standardized: small returns otherwise give the network vanishing gradients
        net = MLPRegressor(hidden_layer_sizes=(32, 16, 8), alpha=p["alpha"], learning_rate_init=1e-3, batch_size=1024,
                           max_iter=30 if fast else 60, early_stopping=True, validation_fraction=0.15,
                           n_iter_no_change=5, random_state=s)
        return TransformedTargetRegressor(make_pipeline(StandardScaler(), net), transformer=StandardScaler())

    seeds = seeds[:1] if fast else seeds
    return {
        "OLS-3 (size, value, momentum)": (lambda p, s: LinearRegression(), [{}], OLS3, [0]),
        "OLS (all)": (lambda p, s: LinearRegression(), [{}], None, [0]),
        "Elastic net": (enet, [{"alpha": a} for a in (1e-5, 1e-4, 1e-3)], None, [0]),
        "Gradient-boosted trees": (lambda p, s: HistGradientBoostingRegressor(
            max_depth=p["depth"], learning_rate=0.05, max_iter=p["iters"], min_samples_leaf=200, l2_regularization=1.0,
            random_state=s), [{"depth": d, "iters": it} for d in (1, 2) for it in (50, 150)], None, seeds),
        "Neural network": (nn, [{"alpha": a} for a in (1e-1, 1.0, 10.0)], None, seeds),
    }


def r2_oos(y, f) -> float:
    """Pooled out-of-sample R-squared against a forecast of zero (Gu, Kelly and Xiu 2020). With the demeaned target,
    zero is the forecast that every asset earns the month's average return."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(1 - np.sum((y - f) ** 2) / np.sum(y ** 2))


def _fit(factory, params, seeds, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _Ensemble([factory(params, s).fit(X, y) for s in seeds])


def walk_forward(panel: pd.DataFrame, models: dict, test_start: str = "1975-01", refit_years: int = 5,
                 val_years: int = 10, features=None, target: str = "ret_next_dm", verbose: bool = False) -> dict:
    """Expanding-window forecasts. At each refit date, models are tuned on the last `val_years` before it (trained
    on everything earlier), then refit on all data up to the refit date and used for the next `refit_years`.
    The default target is next month's return minus that month's cross-sectional average (`ret_next_dm`), so the
    models compete on the cross-section, not on timing the market; use target="ret_next" for raw returns.
    Returns {"pred": DataFrame of forecasts, "params": chosen parameters, "fitted": last fitted models}."""
    allf = features or CHARS + MACRO
    dates = panel.index
    start = pd.Period(test_start, "M")
    last = dates.max()
    out = pd.DataFrame(index=panel.index[dates >= start])
    out["asset"] = panel.loc[dates >= start, "asset"].to_numpy()
    out["y"] = panel.loc[dates >= start, target].to_numpy()
    chosen, fitted = {}, {}
    preds = {n: np.full(len(out), np.nan) for n in models}
    pos_dates = out.index
    t0 = start
    while t0 <= last:
        t1 = t0 + 12 * refit_years
        # training targets must be realized by the refit date: row t has target in t+1, so use rows < t0 - 1
        train = panel[dates < t0 - 1]
        v0 = t0 - 1 - 12 * val_years
        fit_part, val_part = train[train.index < v0 - 1], train[train.index >= v0]
        test_mask = (pos_dates >= t0) & (pos_dates < t1)
        Xtest = panel.loc[(dates >= t0) & (dates < t1)]
        for name, (factory, grid, feats, seeds) in models.items():
            cols = feats or allf
            best = grid[0]
            if len(grid) > 1:
                scores = []
                for p in grid:
                    m = _fit(factory, p, seeds[:1], fit_part[cols].to_numpy(), fit_part[target].to_numpy())
                    scores.append(r2_oos(val_part[target], m.predict(val_part[cols].to_numpy())))
                best = grid[int(np.argmax(scores))]
            m = _fit(factory, best, seeds, train[cols].to_numpy(), train[target].to_numpy())
            preds[name][test_mask] = m.predict(Xtest[cols].to_numpy())
            chosen.setdefault(name, []).append({"refit": str(t0), **best})
            fitted[name] = (m, cols)
            if verbose:
                print(f"  {t0} {name}: {best}")
        t0 = t1
    for n in models:
        out[n] = preds[n]
    return {"pred": out, "params": chosen, "fitted": fitted}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _nw_t(x, lags: int = 6) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    mu = x.mean()
    e = x - mu
    s = e @ e / T
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / T
    se = math.sqrt(max(s, 1e-30) / T)
    return float(mu), float(mu / se)


def diebold_mariano(pred: pd.DataFrame, a: str, b: str) -> float:
    """t-statistic that model a has lower squared error than model b, using monthly cross-sectional averages of the
    loss difference (positive favours a)."""
    d = (pred["y"] - pred[b]) ** 2 - (pred["y"] - pred[a]) ** 2
    return _nw_t(d.groupby(level=0).mean().to_numpy())[1]


def long_short(pred: pd.DataFrame, model: str, frac: float = 0.1, cost: float = 0.0) -> pd.DataFrame:
    """Each month, equal-weight the top and bottom `frac` of assets by forecast. Returns monthly long, short,
    long-short return (net of `cost` per unit of turnover) and turnover (sum of absolute weight changes)."""
    g = pred[["asset", model, "y"]].dropna(subset=[model]).copy()
    if g.empty:
        return pd.DataFrame(columns=["long", "short", "ls", "turnover"], dtype=float)
    g["date"] = g.index
    g = g.reset_index(drop=True)
    grp = g.groupby("date")[model]
    n = grp.transform("size")
    k = np.maximum(1, np.round(frac * n)).astype(int)
    hi = grp.rank(method="first", ascending=False) <= k
    lo = grp.rank(method="first", ascending=True) <= k
    g["w"] = np.where(hi, 1.0 / k, 0.0) - np.where(lo, 1.0 / k, 0.0)
    long = g[hi].groupby("date")["y"].mean()
    short = g[lo].groupby("date")["y"].mean()
    W = g.pivot_table(index="date", columns="asset", values="w", aggfunc="sum").fillna(0.0)
    to = W.diff().abs().sum(axis=1)
    to.iloc[0] = W.iloc[0].abs().sum()
    out = pd.DataFrame({"long": long, "short": short})
    out["turnover"] = to.reindex(out.index)
    out["ls"] = out["long"] - out["short"] - cost * out["turnover"]
    out.index.name = "date"
    return out[["long", "short", "ls", "turnover"]]


def ls_stats(ls: pd.DataFrame) -> dict:
    if len(ls) < 3:
        return {"mean_monthly": float("nan"), "t": float("nan"), "sharpe": float("nan"), "turnover": float("nan")}
    m, t = _nw_t(ls["ls"].to_numpy())
    return {"mean_monthly": m, "t": t, "sharpe": float(ls["ls"].mean() / ls["ls"].std() * math.sqrt(12)),
            "turnover": float(ls["turnover"].iloc[1:].mean())}


def permutation_importance(model, cols, panel: pd.DataFrame, n_repeats: int = 2, seed: int = 0,
                           target: str = "ret_next_dm") -> pd.Series:
    """Drop in out-of-sample R-squared when each feature is shuffled across rows within the same month."""
    rng = np.random.default_rng(seed)
    X = panel[cols].to_numpy().copy()
    y = panel[target].to_numpy()
    base = r2_oos(y, model.predict(X))
    groups = pd.Series(np.arange(len(panel)), index=panel.index).groupby(level=0).indices
    out = {}
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            for idx in groups.values():
                Xp[idx, j] = X[rng.permutation(idx), j]
            drops.append(base - r2_oos(y, model.predict(Xp)))
        out[c] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Random-feature market timing (Kelly, Malamud and Zhou 2024)
# ---------------------------------------------------------------------------


def random_features_timing(X: pd.DataFrame, y: pd.Series, n_features=(2, 6, 12, 60, 120, 600, 1200), window: int = 12,
                           ridge=(1e-3, 1e-1, 10.0), gamma: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """Rolling-window ridge regressions of next month's return on P random Fourier features of the predictors.
    Predictors are standardized with expanding statistics (no look-ahead). Returns one row per (P, ridge) with
    out-of-sample R-squared, the timing strategy's mean, Sharpe ratio and alpha t-statistic against the market."""
    rng = np.random.default_rng(seed)
    Z = (X - X.expanding(36).mean()) / X.expanding(36).std()
    Z = Z.shift(0).dropna()
    yy = y.shift(-1).reindex(Z.index)
    ok = yy.notna()
    Z, yy = Z[ok].to_numpy(), yy[ok].to_numpy()
    T, K = Z.shape
    Pmax = max(n_features)
    W = rng.standard_normal((K, Pmax // 2)) * gamma
    S = np.c_[np.sin(Z @ W), np.cos(Z @ W)]
    rows = []
    for P in n_features:
        F = S[:, np.r_[0:P // 2, Pmax // 2:Pmax // 2 + P // 2]] / math.sqrt(P)
        for z in ridge:
            f = np.full(T, np.nan)
            for t in range(window, T):
                A, b = F[t - window:t], yy[t - window:t]
                # ridge in the dual form: beta = A' (A A' + z I)^-1 b, cheap when P > window
                beta = A.T @ np.linalg.solve(A @ A.T + z * np.eye(window), b)
                f[t] = F[t] @ beta
            m = np.isfinite(f)
            strat = f[m] * yy[m]
            xm = yy[m]
            Xr = np.c_[np.ones(m.sum()), xm]
            coef = np.linalg.lstsq(Xr, strat, rcond=None)[0]
            res = strat - Xr @ coef
            se = math.sqrt(res @ res / (len(res) - 2) * np.linalg.inv(Xr.T @ Xr)[0, 0])
            rows.append({"P": P, "complexity_P_over_T": P / window, "ridge": z, "r2_oos": r2_oos(yy[m], f[m]),
                         "timing_sharpe": float(strat.mean() / strat.std() * math.sqrt(12)),
                         "alpha_t": float(coef[0] / se), "months": int(m.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulated_panel(n_assets: int = 150, n_months: int = 1188, seed: int = 0, start: str = "1926-07",
                    signal: float = 1.0) -> dict:
    """Fake monthly excess returns, average size, firm counts and book-to-market for `n_assets` portfolios. Expected
    returns depend on lagged value and momentum (linear), and on an interaction of size with momentum and a
    threshold in volatility (nonlinear). With signal=1 the predictable share of return variance is roughly 1%.
    TESTING ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    N, T = n_assets, n_months
    mkt = 0.006 + 0.045 * rng.standard_t(5, T) / math.sqrt(5 / 3)
    beta = rng.uniform(0.7, 1.4, N)
    logsize = rng.normal(0, 1.5, N)
    logbm = rng.normal(-0.3, 0.5, N)
    idio = rng.uniform(0.02, 0.06, N)
    R = np.zeros((T, N))
    size = np.zeros((T, N))
    bmv = np.zeros((T, N))
    for t in range(T):
        logsize = 0.995 * logsize + 0.08 * rng.standard_normal(N)
        logbm = 0.99 * logbm + 0.05 * rng.standard_normal(N) - 0.003
        size[t], bmv[t] = np.exp(logsize + 5), np.exp(logbm)
        mu = np.zeros(N)
        if t >= 13:
            mom = np.prod(1 + R[t - 12:t - 1], axis=0) - 1
            vol = R[t - 12:t].std(axis=0)
            zr = lambda v: (v.argsort().argsort() / (N - 1)) * 2 - 1  # noqa: E731
            zs, zb, zm, zv = zr(size[t - 1]), zr(bmv[t - 1]), zr(mom), zr(vol)
            mu = signal * (0.0015 * zb + 0.0012 * zm + 0.0025 * zm * (zs < 0) - 0.003 * (zv > 0.6))
        R[t] = mu + beta * mkt[t] + idio * rng.standard_normal(N)
    cols = [f"P{i + 1:03d}" for i in range(N)]
    nf = np.maximum(1, np.round(np.exp(5 - 0.3 * (np.log(size) - 5) + rng.normal(0, 0.1, (T, N)))))
    df = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
    return {"ret": df(R), "mkt": pd.Series(mkt, index=idx, name="mkt"), "size": df(size), "nfirms": df(nf),
            "bm": df(bmv), "large": [c for c, s in zip(cols, size.mean(axis=0)) if s >= np.median(size.mean(axis=0))],
            "simulated": True}
