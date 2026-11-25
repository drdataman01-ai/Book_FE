"""Zero-shot time-series foundation models against classical forecasts (Chapter 20).

* Classical one-step forecasters: historical mean, naive (last value), AR(p) chosen by BIC, HAR-type (1, 3 and 12
  month averages), GARCH(1,1) for realized variance
* Adapters for foundation models that are installed separately: Chronos / Chronos-Bolt (`chronos-forecasting`) and
  TimesFM (`timesfm`). Each adapter imports its package only when used and explains what to install if it is missing.
* Two clearly labelled test doubles, which are NOT foundation models: a stand-in that averages classical forecasts
  (to exercise the pipeline) and a "memorizing" model that has seen the answers before a cutoff date (to show what
  training-data contamination looks like in an evaluation)
* rolling_origin - one-step forecasts from every origin in the evaluation window, context limited to the past
* scores - MSE ratios, out-of-sample R-squared, Diebold-Mariano tests, interval coverage, pinball loss, split
  before and after a cutoff date
* simulated_series - fake monthly series with known dynamics. TESTING ONLY.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Classical forecasters: f(history) -> (point, sd) ; sd may be NaN
# ---------------------------------------------------------------------------


def f_mean(x: np.ndarray):
    return float(x.mean()), float(x.std(ddof=1))


def f_naive(x: np.ndarray):
    d = np.diff(x)
    return float(x[-1]), float(np.sqrt(np.mean(d ** 2))) if len(d) > 1 else float("nan")


def _ar_fit(x: np.ndarray, p: int):
    y = x[p:]
    X = np.column_stack([np.ones(len(y))] + [x[p - k:len(x) - k] for k in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, e


def f_ar(x: np.ndarray, max_p: int = 12, window: int = 360):
    """AR(p) by OLS on the last `window` observations, p from 1 to max_p chosen by BIC."""
    x = x[-window:]
    best = None
    for p in range(1, max_p + 1):
        if len(x) < 3 * p + 20:
            break
        b, e = _ar_fit(x, p)
        n = len(e)
        bic = n * math.log(e @ e / n) + (p + 1) * math.log(n)
        if best is None or bic < best[0]:
            best = (bic, p, b, e)
    _, p, b, e = best
    f = b[0] + b[1:] @ x[::-1][:p]
    return float(f), float(e.std(ddof=p + 1))


def f_har(x: np.ndarray, window: int = 360):
    """Regression on the last value and the 3- and 12-month averages (a monthly version of Corsi's HAR)."""
    x = x[-window:]
    s = pd.Series(x)
    F = np.column_stack([s, s.rolling(3).mean(), s.rolling(12).mean()])
    y = x[1:]
    X = np.column_stack([np.ones(len(y)), F[:-1]])
    ok = np.isfinite(X).all(axis=1)
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    e = y[ok] - X[ok] @ b
    return float(b[0] + b[1:] @ F[-1]), float(e.std(ddof=4))


CLASSICAL = {"Historical mean": f_mean, "Naive (last value)": f_naive, "AR (BIC)": f_ar, "HAR-type (1, 3, 12)": f_har}


def garch_logvar_forecasts(ret: pd.Series, logrv: pd.Series, origins, refit_every: int = 60) -> pd.DataFrame:
    """GARCH(1,1) on monthly returns, mapped to a forecast of next month's log realized variance by adding the average
    gap between log realized variance and log GARCH variance over the estimation sample (no look-ahead)."""
    from fintest.vol import fit_garch, garch_variance

    r = ret.reindex(logrv.index)
    out, params = {}, None
    for k, t in enumerate(origins):
        hist = r.loc[:t].dropna().to_numpy()
        if params is None or k % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = fit_garch(hist)
        rc = hist - hist.mean()
        s2 = garch_variance(np.r_[rc, 0.0], params["omega"], params["alpha"], params["beta"])
        gap = np.nanmean(logrv.loc[:t].to_numpy()[-len(rc):] - np.log(s2[:len(rc)]))
        out[t] = math.log(s2[-1]) + gap
    return pd.DataFrame({"GARCH(1,1)": pd.Series(out)})


# ---------------------------------------------------------------------------
# Foundation-model adapters (optional packages)
# ---------------------------------------------------------------------------


class ModelUnavailable(RuntimeError):
    pass


class ChronosForecaster:
    """Chronos or Chronos-Bolt via the `chronos-forecasting` package (pip install chronos-forecasting).
    Written for the BaseChronosPipeline.predict_quantiles interface; check the package documentation if it fails."""

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", context: int = 512):
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise ModelUnavailable(f"{model_id}: install with `pip install chronos-forecasting` ({e})") from e
        self.torch = torch
        self.name = model_id.split("/")[-1]
        self.context = context
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)

    def predict(self, contexts: list[np.ndarray]):
        ctx = [self.torch.tensor(c[-self.context:], dtype=self.torch.float32) for c in contexts]
        q, mean = self.pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=list(QUANTILES))
        q = q.numpy()[:, 0, :]
        return q[:, 1], q          # median as the point forecast, plus the 10/50/90 quantiles


class TimesFMForecaster:
    """TimesFM via the `timesfm` package. Tries the 2.5 interface first, then the 1.x/2.0 interface. The package's
    interface has changed between releases; check its documentation if both fail."""

    def __init__(self, context: int = 512):
        try:
            import timesfm
        except ImportError as e:
            raise ModelUnavailable(f"TimesFM: install with `pip install timesfm` ({e})") from e
        self.context = context
        self.name = "timesfm"
        self.mode = None
        try:
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            m.compile(timesfm.ForecastConfig(max_context=context, max_horizon=16, normalize_inputs=True,
                                             use_continuous_quantile_head=True))
            self.model, self.mode, self.name = m, "2.5", "timesfm-2.5-200m"
        except Exception:  # noqa: BLE001
            try:
                m = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32,
                                                                    horizon_len=1, context_len=context),
                                    checkpoint=timesfm.TimesFmCheckpoint(
                                        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
                self.model, self.mode, self.name = m, "1.x", "timesfm-2.0-500m"
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailable(f"TimesFM is installed but could not be loaded with either interface ({e})")

    def predict(self, contexts: list[np.ndarray]):
        ctx = [np.asarray(c[-self.context:], float) for c in contexts]
        if self.mode == "2.5":
            point, quant = self.model.forecast(horizon=1, inputs=ctx)
            point, quant = np.asarray(point)[:, 0], np.asarray(quant)[:, 0, :]
            # quantile head: mean followed by deciles 0.1..0.9
            q = quant[:, [1, 5, 9]] if quant.shape[1] >= 10 else None
            return point, q
        point, quant = self.model.forecast(ctx, freq=[1] * len(ctx))
        point, quant = np.asarray(point)[:, 0], np.asarray(quant)[:, 0, :]
        q = quant[:, [1, 5, 9]] if quant.shape[1] >= 10 else None
        return point, q


class StandIn:
    """TEST DOUBLE, NOT A FOUNDATION MODEL: the average of the naive, AR and HAR-type forecasts. It lets the pipeline
    run where no foundation model is installed."""

    name = "Stand-in (not a foundation model)"

    def predict(self, contexts):
        pts = []
        for c in contexts:
            pts.append(np.mean([f_naive(c)[0], f_ar(c)[0], f_har(c)[0]]))
        return np.array(pts), None


class Memorizer:
    """DEMONSTRATION OF CONTAMINATION, NOT A FOUNDATION MODEL: before `cutoff` it has 'seen' the answer and returns
    0.6 x truth + 0.4 x AR forecast; after the cutoff it falls back to the AR forecast."""

    name = "Memorizing model (contamination demo)"

    def __init__(self, truth: pd.Series, cutoff: str):
        self.truth, self.cutoff = truth, pd.Period(cutoff, "M")

    def predict_at(self, contexts, origins):
        out = []
        for c, t in zip(contexts, origins):
            ar = f_ar(c)[0]
            nxt = t + 1
            out.append(0.6 * self.truth.get(nxt, ar) + 0.4 * ar if nxt < self.cutoff else ar)
        return np.array(out), None


def load_foundation_models(names, context: int = 512):
    """Instantiate the requested adapters; returns (models, messages)."""
    models, msgs = [], []
    for n in names:
        try:
            if n.startswith("chronos"):
                models.append(ChronosForecaster(f"amazon/{n}", context))
            elif n.startswith("timesfm"):
                models.append(TimesFMForecaster(context))
            elif n == "standin":
                models.append(StandIn())
            else:
                msgs.append(f"unknown model {n}")
        except ModelUnavailable as e:
            msgs.append(str(e))
    return models, msgs


# ---------------------------------------------------------------------------
# Rolling-origin evaluation
# ---------------------------------------------------------------------------


def rolling_origin(series: pd.Series, eval_start: str, models=(), memorizer: Memorizer | None = None,
                   context: int = 512, min_history: int = 120, batch: int = 64) -> pd.DataFrame:
    """One-step forecasts for every origin t from eval_start - 1 to the second-last month: the forecast for t+1 uses
    only observations up to t. Returns a DataFrame indexed by target month with the actual value, each model's point
    forecast, classical standard deviations (`sd:` columns) and foundation-model quantiles (`q10:`/`q90:` columns)."""
    s = series.dropna()
    start = pd.Period(eval_start, "M")
    origins = [t for t in s.index[:-1] if t + 1 >= start and s.loc[:t].size >= min_history]
    rows = {t + 1: {"actual": s.get(t + 1, np.nan)} for t in origins}
    ctxs = {t: s.loc[:t].to_numpy()[-context:] for t in origins}
    for name, f in CLASSICAL.items():
        for t in origins:
            p, sd = f(s.loc[:t].to_numpy())
            rows[t + 1][name] = p
            rows[t + 1][f"sd:{name}"] = sd
    for m in models:
        for i in range(0, len(origins), batch):
            chunk = origins[i:i + batch]
            point, q = m.predict([ctxs[t] for t in chunk])
            for j, t in enumerate(chunk):
                rows[t + 1][m.name] = float(point[j])
                if q is not None:
                    rows[t + 1][f"q10:{m.name}"], rows[t + 1][f"q90:{m.name}"] = float(q[j, 0]), float(q[j, -1])
    if memorizer is not None:
        point, _ = memorizer.predict_at([ctxs[t] for t in origins], origins)
        for j, t in enumerate(origins):
            rows[t + 1][memorizer.name] = float(point[j])
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index = pd.PeriodIndex(out.index, freq="M")
    out.index.name = "date"
    return out


def model_columns(fc: pd.DataFrame) -> list[str]:
    return [c for c in fc.columns if c != "actual" and ":" not in c]


def _dm(actual, f1, f2, lags: int = 3) -> float:
    d = (actual - f2) ** 2 - (actual - f1) ** 2   # positive: f1 better
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return float("nan")
    e = d - d.mean()
    s = e @ e / T
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / T
    return float(d.mean() / math.sqrt(max(s, 1e-30) / T))


def scores(fc: pd.DataFrame, bench: str = "AR (BIC)") -> pd.DataFrame:
    """Per model: MSE, MSE relative to the benchmark (below 1 is better), out-of-sample R-squared against the
    historical mean, Diebold-Mariano t against the benchmark (positive: model better), share of months with the
    smaller error, and 80% interval coverage where the model provides an interval."""
    a = fc["actual"].to_numpy()
    mse_b = np.mean((a - fc[bench].to_numpy()) ** 2)
    mse_m = np.mean((a - fc["Historical mean"].to_numpy()) ** 2)
    rows = []
    for m in model_columns(fc):
        f = fc[m].to_numpy()
        mse = np.mean((a - f) ** 2)
        cov = float("nan")
        if f"q10:{m}" in fc:
            cov = float(np.mean((a >= fc[f"q10:{m}"]) & (a <= fc[f"q90:{m}"])))
        elif f"sd:{m}" in fc:
            sd = fc[f"sd:{m}"].to_numpy()
            cov = float(np.mean(np.abs(a - f) <= 1.2816 * sd))
        rows.append({"model": m, "mse": mse, "mse_ratio_vs_" + bench: mse / mse_b, "r2_vs_mean": 1 - mse / mse_m,
                     "dm_t_vs_" + bench: float("nan") if m == bench else _dm(a, f, fc[bench].to_numpy()),
                     "win_rate_vs_" + bench: float(np.mean(np.abs(a - f) < np.abs(a - fc[bench].to_numpy()))),
                     "coverage_80": cov, "months": len(a)})
    return pd.DataFrame(rows)


def split_scores(fc: pd.DataFrame, cutoff: str, bench: str = "AR (BIC)") -> pd.DataFrame:
    c = pd.Period(cutoff, "M")
    out = []
    for label, part in (("before cutoff", fc[fc.index < c]), ("after cutoff", fc[fc.index >= c])):
        if len(part) < 6:
            continue
        s = scores(part, bench)[["model", "mse_ratio_vs_" + bench, "months"]]
        s.insert(0, "window", label)
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulated_series(n_months: int = 1200, seed: int = 0, start: str = "1926-07") -> dict:
    """Fake monthly series with known dynamics: iid returns (nothing to forecast beyond the mean), log realized
    variance with long memory (a HAR-type process), inflation as AR(2), output growth as AR(1), changes in a yield and
    an unemployment rate as near white noise. TESTING ONLY."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_months, freq="M")
    T = n_months
    lrv = np.full(T, -6.0)
    for t in range(12, T):
        lrv[t] = -6.0 * 0.05 + 0.45 * lrv[t - 1] + 0.3 * lrv[t - 3:t].mean() + 0.2 * lrv[t - 12:t].mean() \
            + 0.45 * rng.standard_normal()
    ret = 0.006 + np.exp(lrv / 2) * rng.standard_normal(T) * 1.0
    infl = np.zeros(T)
    for t in range(2, T):
        infl[t] = 0.0003 + 0.5 * infl[t - 1] + 0.3 * infl[t - 2] + 0.002 * rng.standard_normal()
    ip = np.zeros(T)
    for t in range(1, T):
        ip[t] = 0.002 + 0.3 * ip[t - 1] + 0.01 * rng.standard_normal()
    dy = 0.25 * rng.standard_normal(T) + 0.1 * np.r_[0, rng.standard_normal(T - 1)]
    du = np.r_[0, 0.15 * rng.standard_normal(T - 1)]
    du[1:] += 0.2 * du[:-1]
    S = lambda v, n: pd.Series(v, index=idx, name=n)  # noqa: E731
    return {"series": {"Market excess return": S(ret, "ret"), "Log realized variance": S(lrv, "lrv"),
                       "Inflation (CPI, monthly)": S(infl, "infl"), "Industrial production growth": S(ip, "ip"),
                       "Unemployment rate change": S(du, "du"), "10-year yield change": S(dy, "dy")},
            "market_return": S(ret, "ret"), "simulated": True}
