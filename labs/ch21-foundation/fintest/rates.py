"""Term-structure tools (Chapter 12).

* A reader for the Federal Reserve's zero-coupon Treasury yield curve
  (Gurkaynak, Sack and Wright 2007; file feds200628.csv)
* Log bond prices, forward rates and one-year holding-period excess returns
  from annual zero-coupon yields
* The regressions of Fama and Bliss (1987), Campbell and Shiller (1991) and
  Cochrane and Piazzesi (2005)

Notation (Cochrane and Piazzesi 2005): maturities n in years; y(n) is the
continuously compounded n-year zero yield (decimal); p(n) = -n y(n) the log
price; f(n) = p(n-1) - p(n) the forward rate for year n-1 to n; the one-year
log holding return on an n-year bond is r(n)_{t+1} = p(n-1)_{t+1} - p(n)_t, and
the excess return rx(n) = r(n) - y(1)_t. With monthly data and a one-year
holding period, observations overlap by 11 months.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_CACHE, DataUnavailable
from .stats import ols_newey_west

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


def read_gsw(path=None, cache_dir: Path = DEFAULT_CACHE, max_n: int = 10) -> pd.DataFrame:
    """Month-end zero-coupon yields (decimal, continuously compounded) for maturities 1..max_n years.

    Reads the Federal Reserve's feds200628.csv (downloaded and cached if no path is given). Uses the SVENYxx
    columns; returns a DataFrame indexed by monthly Period with columns 1..max_n.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise DataUnavailable(f"File not found: {p}")
        text = p.read_text(errors="ignore")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        f = cache_dir / "feds200628.csv"
        if not f.exists():
            try:
                req = urllib.request.Request(GSW_URL, headers={"User-Agent": "fintest (textbook lab)"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    f.write_bytes(resp.read())
            except Exception as e:  # noqa: BLE001
                raise DataUnavailable(f"Could not download {GSW_URL} ({e}). Download it from "
                                      "https://www.federalreserve.gov/data/nominal-yield-curve.htm and pass its path.") from e
        text = f.read_text(errors="ignore")
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().strip('"').lower().startswith("date")), None)
    if start is None:
        raise DataUnavailable("No header row starting with 'Date' in the yield-curve file.")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    cols = {}
    for n in range(1, max_n + 1):
        c = f"SVENY{n:02d}"
        if c not in df.columns:
            raise DataUnavailable(f"Column {c} missing; found {list(df.columns)[:12]}...")
        cols[n] = pd.to_numeric(df[c], errors="coerce") / 100
    y = pd.DataFrame(cols)
    m = y.groupby(y.index.to_period("M")).last()
    m.index.name = "date"
    return m


def term_structure(y: pd.DataFrame, h: int = 12) -> dict:
    """Log prices, forwards, forward spreads and one-year excess returns from monthly annual-maturity yields.

    y  columns 1..N (years), monthly rows. h = number of rows in one year (12 for monthly data).
    Returns DataFrames with columns n: p, f (n >= 2), fs = f(n) - y(1), rx (n >= 2; row t holds the return from t
    to t+h), and the realized future short-rate change and long-yield change used by the regressions.
    """
    N = max(y.columns)
    p = -y.mul(pd.Series({n: n for n in y.columns}), axis=1)
    f = pd.DataFrame({n: p[n - 1] - p[n] for n in range(2, N + 1)})
    fs = f.sub(y[1], axis=0)
    rx = pd.DataFrame({n: p[n - 1].shift(-h) - p[n] - y[1] for n in range(2, N + 1)})
    return {"y": y, "p": p, "f": f, "fs": fs, "rx": rx}


def _reg(yv: pd.Series, X: pd.DataFrame | pd.Series, lags: int) -> dict:
    df = pd.concat([yv.rename("y"), X], axis=1).dropna()
    Xc = df.drop(columns="y").to_numpy()
    r = ols_newey_west(df["y"].to_numpy(), Xc, lags=lags)
    e = df["y"].to_numpy() - np.column_stack([np.ones(len(df)), Xc]) @ r["coef"][:, 0]
    return {"coef": r["coef"][:, 0], "t": r["t"][:, 0], "se": r["coef"][:, 0] / r["t"][:, 0],
            "r2": float(1 - e.var() / df["y"].var()), "n": len(df), "start": str(df.index[0]), "end": str(df.index[-1])}


def fama_bliss(ts: dict, lags: int = 18) -> pd.DataFrame:
    """Fama-Bliss (1987): excess return rx(n) on the forward spread f(n) - y(1). The expectations hypothesis implies
    a slope of 0; a slope of 1 means forward spreads forecast excess returns one for one."""
    rows = []
    for n in ts["rx"].columns:
        r = _reg(ts["rx"][n], ts["fs"][n].rename("fs"), lags)
        rows.append({"n": n, "slope": r["coef"][1], "se": r["se"][1], "t_vs_0": r["t"][1],
                     "t_vs_1": (r["coef"][1] - 1) / r["se"][1], "r2": r["r2"], "months": r["n"]})
    return pd.DataFrame(rows)


def fama_bliss_short(ts: dict, h: int = 12, lags: int = 18) -> pd.DataFrame:
    """Fama-Bliss companion regression: the change in the one-year rate over the next n-1 years,
    y(1)_{t+(n-1)} - y(1)_t, on the forward spread f(n)_t - y(1)_t. Expectations hypothesis: slope 1."""
    rows = []
    y1 = ts["y"][1]
    for n in ts["fs"].columns:
        k = (n - 1) * h
        dy = y1.shift(-k) - y1
        r = _reg(dy, ts["fs"][n].rename("fs"), max(lags, int(1.5 * k)))
        rows.append({"n": n, "horizon_years": n - 1, "slope": r["coef"][1], "se": r["se"][1],
                     "t_vs_1": (r["coef"][1] - 1) / r["se"][1], "r2": r["r2"], "months": r["n"]})
    return pd.DataFrame(rows)


def campbell_shiller(ts: dict, h: int = 12, lags: int = 18) -> pd.DataFrame:
    """Campbell-Shiller (1991): y(n-1)_{t+1} - y(n)_t on (y(n)_t - y(1)_t)/(n-1). Expectations hypothesis: slope 1."""
    rows = []
    y = ts["y"]
    for n in [c for c in y.columns if c >= 2]:
        dep = y[n - 1].shift(-h) - y[n]
        x = ((y[n] - y[1]) / (n - 1)).rename("spread")
        r = _reg(dep, x, lags)
        rows.append({"n": n, "slope": r["coef"][1], "se": r["se"][1], "t_vs_1": (r["coef"][1] - 1) / r["se"][1],
                     "r2": r["r2"], "months": r["n"]})
    return pd.DataFrame(rows)


def cochrane_piazzesi(ts: dict, maturities=(2, 3, 4, 5), lags: int = 18) -> dict:
    """Cochrane-Piazzesi (2005): average excess return across maturities on y(1) and forwards f(2)..f(5).

    Returns the coefficients (the 'tent'), t-statistics, R2 and the fitted factor (the CP factor), plus the R2 of
    each maturity's excess return on the factor alone.
    """
    mats = list(maturities)
    rxbar = ts["rx"][mats].mean(axis=1)
    X = pd.concat([ts["y"][1].rename("y1")] + [ts["f"][n].rename(f"f{n}") for n in mats], axis=1)
    r = _reg(rxbar, X, lags)
    factor = r["coef"][0] + X.to_numpy() @ r["coef"][1:]
    factor = pd.Series(factor, index=X.index, name="cp")
    each = []
    for n in mats:
        rr = _reg(ts["rx"][n], factor, lags)
        each.append({"n": n, "loading": rr["coef"][1], "t": rr["t"][1], "r2": rr["r2"]})
    return {"names": ["const"] + list(X.columns), "coef": r["coef"], "t": r["t"], "r2": r["r2"], "factor": factor,
            "rxbar": rxbar, "X": X, "each": pd.DataFrame(each), "months": r["n"]}


def simulated_yields(n_months: int = 780, max_n: int = 10, seed: int = 0, start: str = "1961-07") -> pd.DataFrame:
    """Fake annual-maturity zero yields from an AR(1) short rate plus a time-varying term premium. TESTING ONLY.

    y(n)_t = average expected one-year rate over the next n years + tp_t * (1 - exp(-n/3)) + noise, so the
    expectations hypothesis is violated whenever the premium tp_t moves.
    """
    rng = np.random.default_rng(seed)
    phi, mu, sd = 0.985, 0.045, 0.004
    rho_tp, sd_tp = 0.96, 0.0025
    r1, tp = np.zeros(n_months), np.zeros(n_months)
    r1[0], tp[0] = mu, 0.01
    for t in range(1, n_months):
        r1[t] = mu + phi * (r1[t - 1] - mu) + sd * rng.standard_normal()
        tp[t] = 0.01 + rho_tp * (tp[t - 1] - 0.01) + sd_tp * rng.standard_normal()
    phi_y = phi ** 12  # one-year persistence of the one-year rate
    cols = {}
    for n in range(1, max_n + 1):
        k = np.arange(n)
        avg = mu + (r1 - mu)[:, None] * (phi_y ** k)[None, :]
        cols[n] = avg.mean(axis=1) + tp * (1 - np.exp(-(n - 1) / 3)) + 0.0003 * rng.standard_normal(n_months)
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    return pd.DataFrame(cols, index=idx)
