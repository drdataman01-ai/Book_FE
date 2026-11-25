"""Equity-premium predictors: the Goyal-Welch data (Chapter 6 onward).

Amit Goyal maintains the monthly data used by Goyal and Welch (2008) and
Goyal, Welch and Zafirov (2024): the S&P 500 index, its dividends and
earnings, interest rates, bond returns and other variables that have been
proposed as predictors of the equity premium. The file is published on his
website (https://sites.google.com/view/agoyal145) as a spreadsheet ("Updated
data" under Goyal and Welch 2008). Download it, save it in data/ and pass its
path, or export its "Monthly" sheet to CSV.

The links on that page point to Google Drive, so this module does not
download the file itself.

build_predictors() turns the raw columns into the variables of Goyal and
Welch (2008), Table 1: the dividend-price ratio, earnings-price ratio and so
on. Every predictor in the returned table is dated by the month in which it
is known; the target is the equity premium of the *next* month.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_CACHE, DataUnavailable

GOYAL_PAGE = "https://sites.google.com/view/agoyal145"

# name: (description, sign expected by the original papers: +1 = higher value -> higher premium)
PREDICTORS = {
    "dp": ("Dividend-price ratio (log)", +1),
    "dy": ("Dividend yield (log)", +1),
    "ep": ("Earnings-price ratio (log)", +1),
    "de": ("Dividend payout ratio (log)", +1),
    "svar": ("Stock variance", +1),
    "bm": ("Book-to-market ratio", +1),
    "ntis": ("Net equity expansion", -1),
    "tbl": ("Treasury bill rate", -1),
    "lty": ("Long-term yield", -1),
    "ltr": ("Long-term bond return", +1),
    "tms": ("Term spread", +1),
    "dfy": ("Default yield spread", +1),
    "dfr": ("Default return spread", +1),
    "infl": ("Inflation (one-month publication lag)", -1),
}

# Year of an early, widely cited paper proposing each predictor (as credited in
# Goyal and Welch 2008). Used for before/after-publication comparisons.
PUBLICATION_YEAR = {
    "dp": 1988, "dy": 1988, "ep": 1988, "de": 1998, "svar": 2006, "bm": 1997, "ntis": 2007,
    "tbl": 1987, "lty": 1987, "ltr": 1987, "tms": 1989, "dfy": 1989, "dfr": 1989, "infl": 1977,
}

_REQUIRED = ["yyyymm", "index", "d12", "e12", "b/m", "tbl", "aaa", "baa", "lty", "ntis", "rfree", "infl", "ltr",
             "corpr", "svar"]


def _find_file(path) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise DataUnavailable(f"File not found: {p}")
        return p
    if DEFAULT_CACHE.exists():
        cands = sorted([p for p in DEFAULT_CACHE.iterdir() if p.suffix.lower() in (".xlsx", ".xls", ".csv")
                        and ("predictor" in p.name.lower() or "goyal" in p.name.lower())])
        if cands:
            return cands[-1]
    raise DataUnavailable(
        "The Goyal-Welch predictor data was not found.\n"
        f"Download it from Amit Goyal's website ({GOYAL_PAGE}): under Goyal and Welch (2008), 'Updated data'.\n"
        "Save the spreadsheet in data/ with 'predictor' or 'goyal' in its name, or pass --data-file PATH.\n"
        "A CSV export of the 'Monthly' sheet also works.\n"
        "To test the code without it, use --source sim (results are then not evidence about markets)."
    )


def read_goyal_welch(path=None) -> pd.DataFrame:
    """The raw monthly sheet, with lower-case column names and a monthly PeriodIndex."""
    p = _find_file(path)
    if p.suffix.lower() == ".csv":
        raw = pd.read_csv(p)
    else:
        book = pd.ExcelFile(p)
        sheet = next((s for s in book.sheet_names if s.strip().lower() == "monthly"), None)
        if sheet is None:
            raise DataUnavailable(f"No 'Monthly' sheet in {p.name}; sheets: {book.sheet_names}")
        raw = book.parse(sheet)
    return parse_goyal_welch(raw)


def parse_goyal_welch(raw: pd.DataFrame | str) -> pd.DataFrame:
    if isinstance(raw, str):
        raw = pd.read_csv(io.StringIO(raw))
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise DataUnavailable(f"Missing columns in the Goyal-Welch file: {missing}. Found: {list(df.columns)}")
    df = df[df["yyyymm"].notna()]
    ym = pd.to_numeric(df["yyyymm"], errors="coerce").astype("Int64")
    df = df[ym.notna()]
    idx = pd.PeriodIndex([pd.Period(year=int(v) // 100, month=int(v) % 100, freq="M") for v in ym.dropna()],
                         name="date")
    out = df.drop(columns=["yyyymm"]).copy()
    for c in out.columns:  # the index is sometimes stored as text with thousands separators
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    out.index = idx
    return out.sort_index()


def build_predictors(raw: pd.DataFrame) -> pd.DataFrame:
    """Equity premium (`eqp`, simple, decimal) and the 14 predictors of Goyal and Welch (2008).

    Row t holds the equity premium realized in month t and the predictors as
    known at the end of month t. To forecast, pair predictors at t with the
    premium at t+1 (forecast.lagged_pairs does this).
    """
    r = raw
    if "crsp_spvw" in r.columns and r["crsp_spvw"].notna().any():
        ret = r["crsp_spvw"].copy()
        fallback = (r["index"] + r["d12"] / 12) / r["index"].shift(1) - 1
        ret = ret.fillna(fallback)
    else:
        ret = (r["index"] + r["d12"] / 12) / r["index"].shift(1) - 1
    out = pd.DataFrame(index=r.index)
    out["eqp"] = ret - r["rfree"]
    out["rf"] = r["rfree"]
    out["dp"] = np.log(r["d12"]) - np.log(r["index"])
    out["dy"] = np.log(r["d12"]) - np.log(r["index"].shift(1))
    out["ep"] = np.log(r["e12"]) - np.log(r["index"])
    out["de"] = np.log(r["d12"]) - np.log(r["e12"])
    out["svar"] = r["svar"]
    out["bm"] = r["b/m"]
    out["ntis"] = r["ntis"]
    out["tbl"] = r["tbl"]
    out["lty"] = r["lty"]
    out["ltr"] = r["ltr"]
    out["tms"] = r["lty"] - r["tbl"]
    out["dfy"] = r["baa"] - r["aaa"]
    out["dfr"] = r["corpr"] - r["ltr"]
    out["infl"] = r["infl"].shift(1)  # the CPI is published with a delay of about a month
    out.attrs["simulated"] = False
    return out


def load_predictors(path=None) -> pd.DataFrame:
    return build_predictors(read_goyal_welch(path))


def simulated_predictors(n_months: int = 1188, seed: int = 0, start: str = "1926-01", beta_dp: float = 0.0032,
                         rho: float = -0.9) -> pd.DataFrame:
    """Fake predictor data with known truth. FOR TESTING CODE ONLY.

    The dividend-price ratio is a persistent AR(1) whose innovations are
    negatively correlated (rho) with return shocks, as in real data, and it
    truly predicts the premium: a one-standard-deviation rise in dp raises the
    next month's expected premium by beta_dp (with the default settings the
    true monthly R-squared is about 0.5%). Every other
    predictor is a persistent AR(1) with no predictive power.
    """
    rng = np.random.default_rng(seed)
    T = n_months
    idx = pd.period_range(start=start, periods=T, freq="M", name="date")
    phi, sd_x = 0.985, 0.4
    e_ret = rng.standard_t(5, T) / np.sqrt(5 / 3)
    e_x = rho * e_ret + np.sqrt(1 - rho ** 2) * rng.standard_normal(T)
    dp = np.empty(T)
    dp[0] = -3.4
    for t in range(1, T):
        dp[t] = -3.4 * (1 - phi) + phi * dp[t - 1] + sd_x * np.sqrt(1 - phi ** 2) * e_x[t]
    eqp = np.empty(T)
    eqp[0] = 0.005
    eqp[1:] = 0.005 + beta_dp * (dp[:-1] + 3.4) / sd_x + 0.045 * e_ret[1:]
    out = pd.DataFrame(index=idx)
    out["eqp"] = eqp
    out["rf"] = 0.003
    out["dp"] = dp
    names = [k for k in PREDICTORS if k != "dp"]
    for j, name in enumerate(names):
        x = np.empty(T)
        x[0] = 0
        ph = 0.95 if name in ("ltr", "dfr", "svar") else 0.98
        z = rng.standard_normal(T)
        for t in range(1, T):
            x[t] = ph * x[t - 1] + np.sqrt(1 - ph ** 2) * z[t]
        out[name] = x
    out.attrs["simulated"] = True
    return out
