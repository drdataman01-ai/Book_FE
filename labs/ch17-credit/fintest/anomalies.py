"""Tools for measuring published anomalies (Chapter 16).

* Readers for the Open Source Asset Pricing data (Chen and Zimmermann 2022):
  long-short portfolio returns in wide or long format, and the signal
  documentation with each paper's sample and publication years
* In-sample, post-sample and post-publication statistics per signal
* A pooled McLean-Pontiff (2016) style regression of normalized returns on
  post-sample and post-publication indicators, with month-clustered errors
* Empirical Bayes shrinkage of in-sample means (a simple version of the
  hierarchical approach of Jensen, Kelly and Pedersen 2023)

Data: https://www.openassetpricing.com (portfolio returns and SignalDoc.csv).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataUnavailable


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    s = s.astype(str).str.strip()
    if s.str.fullmatch(r"\d{6}").all():
        return pd.PeriodIndex([f"{v[:4]}-{v[4:]}" for v in s], freq="M")
    return pd.to_datetime(s).dt.to_period("M")


def read_ls_returns(path) -> pd.DataFrame:
    """Monthly long-short returns, one column per signal, in decimals.

    Accepts the wide file (a date column then one column per signal, e.g.
    PredictorLSretWide.csv) or the long format (signalname, port, date, ret),
    keeping port == 'LS'. Open Source Asset Pricing reports returns in percent;
    values are converted to decimals when they look like percentages.
    """
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}. Download portfolio returns from https://www.openassetpricing.com.")
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if {"signalname", "port", "date", "ret"} <= set(cols):
        df = df[df[cols["port"]].astype(str).str.upper() == "LS"]
        df = df.assign(_d=_to_period(df[cols["date"]]))
        wide = df.pivot_table(index="_d", columns=cols["signalname"], values=cols["ret"], aggfunc="mean")
    else:
        date_col = next((c for c in df.columns if c.lower() in ("date", "yyyymm", "month")), df.columns[0])
        wide = df.set_index(_to_period(df[date_col])).drop(columns=[date_col])
        wide = wide.apply(pd.to_numeric, errors="coerce")
    wide.index.name = "date"
    wide = wide.sort_index()
    vals = wide.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) and np.median(np.abs(vals)) > 0.2:  # percent
        wide = wide / 100.0
    return wide


def read_signal_doc(path) -> pd.DataFrame:
    """Signal documentation: acronym, publication year, sample start and end years."""
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}. Download SignalDoc.csv from https://www.openassetpricing.com.")
    df = pd.read_csv(p)
    low = {c.lower().replace(" ", ""): c for c in df.columns}
    need = {"acronym": "signal", "year": "pub_year", "samplestartyear": "sample_start", "sampleendyear": "sample_end"}
    missing = [k for k in need if k not in low]
    if missing:
        raise DataUnavailable(f"SignalDoc is missing columns {missing}; found {list(df.columns)}")
    out = df[[low[k] for k in need]].copy()
    out.columns = list(need.values())
    for c in ("pub_year", "sample_start", "sample_end"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna().drop_duplicates("signal").set_index("signal")


def _mean_t(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    if len(x) < 12:
        return float("nan"), float("nan"), len(x)
    m, s = x.mean(), x.std(ddof=1)
    return float(m), float(m / (s / math.sqrt(len(x)))) if s > 0 else float("nan"), len(x)


def signal_periods(ret: pd.DataFrame, doc: pd.DataFrame) -> pd.DataFrame:
    """Mean and t-statistic per signal in-sample, post-sample (before publication) and post-publication."""
    rows = []
    years = ret.index.year
    for sig in ret.columns:
        if sig not in doc.index:
            continue
        d = doc.loc[sig]
        x = ret[sig].to_numpy()
        ins = (years >= d.sample_start) & (years <= d.sample_end)
        post_s = (years > d.sample_end) & (years <= d.pub_year)
        post_p = years > d.pub_year
        mi, ti, ni = _mean_t(x[ins])
        ms, ts, ns = _mean_t(x[post_s])
        mp, tp, n_p = _mean_t(x[post_p])
        rows.append({"signal": sig, "pub_year": int(d.pub_year), "sample_start": int(d.sample_start),
                     "sample_end": int(d.sample_end), "mean_in": mi, "t_in": ti, "n_in": ni,
                     "mean_post_sample": ms, "t_post_sample": ts, "n_post_sample": ns,
                     "mean_post_pub": mp, "t_post_pub": tp, "n_post_pub": n_p})
    return pd.DataFrame(rows).set_index("signal")


def mclean_pontiff(ret: pd.DataFrame, doc: pd.DataFrame, stats: pd.DataFrame) -> dict:
    """Pooled regression of returns scaled by each signal's in-sample mean on post-sample and post-publication dummies.

    Coefficients are the average proportional change in returns: -0.3 means returns fell by 30% of their in-sample
    mean. Standard errors are clustered by month. Only signals with a positive in-sample mean are used.
    """
    recs = []
    years = ret.index.year
    for sig in stats.index:
        mi = stats.loc[sig, "mean_in"]
        if not (np.isfinite(mi) and mi > 0):
            continue
        d = doc.loc[sig]
        x = ret[sig].to_numpy() / mi
        ok = np.isfinite(x) & (years >= d.sample_start)
        recs.append(pd.DataFrame({"y": x[ok], "post_sample": ((years > d.sample_end) & (years <= d.pub_year))[ok],
                                  "post_pub": (years > d.pub_year)[ok], "month": np.asarray(ret.index)[ok]}))
    panel = pd.concat(recs, ignore_index=True)
    X = np.column_stack([np.ones(len(panel)), panel["post_sample"].astype(float), panel["post_pub"].astype(float)])
    y = panel["y"].to_numpy()
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((3, 3))
    for _, idx in panel.groupby("month").indices.items():
        g = X[idx].T @ e[idx]
        meat += np.outer(g, g)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    return {"coef": dict(zip(["in_sample", "post_sample", "post_pub"], b)),
            "t": dict(zip(["in_sample", "post_sample", "post_pub"], b / se)), "n_obs": len(y),
            "n_signals": len(recs)}


def empirical_bayes(means: np.ndarray, ses: np.ndarray) -> dict:
    """Shrink noisy means toward their common average.

    Model: observed mean m_i ~ N(mu_i, s_i^2) and true mean mu_i ~ N(mu0, tau^2). The prior mean and variance are
    estimated by the method of moments; each posterior mean is m_i shrunk toward mu0 by s_i^2 / (s_i^2 + tau^2).
    """
    m, s = np.asarray(means, float), np.asarray(ses, float)
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    mu0 = float(np.mean(m[ok]))
    tau2 = max(float(np.var(m[ok], ddof=1) - np.mean(s[ok] ** 2)), 1e-12)
    shrink = s ** 2 / (s ** 2 + tau2)
    post_mean = mu0 + (1 - shrink) * (m - mu0)
    post_sd = np.sqrt(1 / (1 / s ** 2 + 1 / tau2))
    from .stats import normal_cdf

    p_pos = np.asarray(normal_cdf(post_mean / post_sd), dtype=float)
    return {"mu0": mu0, "tau": math.sqrt(tau2), "shrink": shrink, "post_mean": post_mean, "post_sd": post_sd,
            "p_positive": p_pos}


def simulated_anomalies(n_signals: int = 200, share_true: float = 0.3, true_mean: float = 0.003, vol: float = 0.035,
                        post_pub_decay: float = 0.3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fake anomaly returns and documentation, with publication only after significant in-sample results.

    Each candidate signal has a true monthly mean of 0 or `true_mean`; after publication, true means fall by
    `post_pub_decay`. Only candidates whose in-sample t exceeds 2 are "published", which builds in selection bias.
    Returns (returns, doc, true in-sample means of the published signals). FOR TESTING AND TEACHING ONLY.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1926-07", "2024-12", freq="M", name="date")
    years = idx.year.to_numpy()
    cols, docs, truth = {}, [], []
    k = 0
    while len(cols) < n_signals and k < n_signals * 20:
        k += 1
        mu = true_mean if rng.random() < share_true else 0.0
        start = int(rng.integers(1950, 1985))
        end = start + int(rng.integers(15, 30))
        pub = end + int(rng.integers(1, 5))
        if pub > 2018:
            continue
        r = mu + vol * rng.standard_normal(len(idx))
        r[years > pub] -= post_pub_decay * mu
        r[years < start - 10] = np.nan
        ins = (years >= start) & (years <= end)
        x = r[ins]
        if x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) <= 2.0:
            continue
        name = f"sig{len(cols):03d}"
        cols[name] = r
        docs.append({"Acronym": name, "Year": pub, "SampleStartYear": start, "SampleEndYear": end})
        truth.append(mu)
    ret = pd.DataFrame(cols, index=idx)
    doc = pd.DataFrame(docs).rename(columns={"Acronym": "signal", "Year": "pub_year", "SampleStartYear": "sample_start",
                                             "SampleEndYear": "sample_end"}).set_index("signal")
    return ret, doc, np.array(truth)
