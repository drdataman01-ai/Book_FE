"""Credit-spread tools (Chapter 17).

* Expected-loss spreads from default probabilities and recovery rates
* The default rate implied if a whole spread were expected loss (break-even)
* Readers for monthly FRED series and the Gilchrist-Zakrajsek excess bond premium
* Simulated spreads with a counter-cyclical risk premium, for testing

Conventions: yields, spreads and default rates are annual decimals. With an annual default probability PD and
recovery rate R (fraction of face value recovered), the spread that exactly compensates a risk-neutral investor for
expected losses on a bond with continuously compounded yields is -ln(1 - PD (1 - R)), close to PD (1 - R).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataUnavailable
from .vol import load_fred


def expected_loss_spread(pd_annual, recovery):
    """Spread that compensates exactly for expected default losses: -ln(1 - PD (1 - R))."""
    return -np.log1p(-np.asarray(pd_annual, float) * (1 - np.asarray(recovery, float))) + 0.0


def breakeven_default_rate(spread, recovery):
    """Annual default rate at which expected losses would absorb the whole spread."""
    return (1 - np.exp(-np.asarray(spread, float))) / (1 - np.asarray(recovery, float))


def monthly_fred(series: str, path=None) -> pd.Series:
    """A FRED series as month-end values with a monthly PeriodIndex."""
    s = load_fred(series, path)
    m = s.groupby(s.index.to_period("M")).last()
    m.index.name = "date"
    return m.rename(series)


def read_ebp(path) -> pd.DataFrame:
    """The Gilchrist-Zakrajsek credit spread and excess bond premium (Federal Reserve 'ebp_csv.csv').

    Expects a date column and columns named like gz_spread and ebp (percentage points); returns decimals.
    """
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}")
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    dcol = next((c for c in df.columns if "date" in c), df.columns[0])
    df.index = pd.to_datetime(df[dcol]).dt.to_period("M")
    keep = [c for c in df.columns if c in ("gz_spread", "ebp")]
    if not keep:
        raise DataUnavailable(f"No gz_spread or ebp column in {p}; found {list(df.columns)}")
    out = df[keep].apply(pd.to_numeric, errors="coerce") / 100
    out.index.name = "date"
    return out


def simulated_credit(n_months: int = 1284, seed: int = 0, start: str = "1919-01") -> pd.DataFrame:
    """Fake monthly Aaa, Baa and 10-year Treasury yields, recessions and industrial production. TESTING ONLY.

    Recessions follow a two-state Markov chain. The Baa-Aaa spread is expected loss (0.15%) plus a risk premium that
    jumps in recessions and decays, and the premium forecasts lower output growth.
    """
    rng = np.random.default_rng(seed)
    rec = np.zeros(n_months, int)
    for t in range(1, n_months):
        p = 0.10 if rec[t - 1] else 0.015
        rec[t] = 1 - rec[t - 1] if rng.random() < p else rec[t - 1]
    tsy = np.zeros(n_months)
    tsy[0] = 0.04
    prem = np.zeros(n_months)
    for t in range(1, n_months):
        tsy[t] = max(0.005, tsy[t - 1] + 0.0003 * (0.05 - tsy[t - 1]) * 10 + 0.0018 * rng.standard_normal())
        target = 0.012 if rec[t] else 0.004
        prem[t] = prem[t - 1] + 0.15 * (target - prem[t - 1]) + 0.0008 * rng.standard_normal()
    prem = np.maximum(prem, 0.0)
    aaa = tsy + 0.006 + 0.3 * prem + 0.0005 * rng.standard_normal(n_months)
    baa = aaa + 0.0015 + prem + 0.0005 * rng.standard_normal(n_months)
    g = 0.0025 - 0.012 * rec - 0.5 * (np.roll(prem, 3) - 0.006) + 0.008 * rng.standard_normal(n_months)
    ip = 50 * np.exp(np.cumsum(g))
    idx = pd.period_range(start, periods=n_months, freq="M", name="date")
    df = pd.DataFrame({"AAA": aaa, "BAA": baa, "GS10": tsy, "USREC": rec, "INDPRO": ip}, index=idx)
    df.loc[df.index < pd.Period("1953-04", "M"), "GS10"] = np.nan
    return df
