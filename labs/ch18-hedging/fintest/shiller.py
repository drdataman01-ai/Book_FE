"""Robert Shiller's long-run U.S. stock market data (Chapter 6).

Shiller publishes monthly S&P Composite prices, dividends, earnings, the CPI,
the 10-year Treasury yield and the cyclically adjusted price-earnings ratio
(CAPE) from 1871, as the spreadsheet ie_data.xls (sheet "Data") on his
website, http://www.econ.yale.edu/~shiller/data.htm. This module reads that
sheet, or a CSV export of it, without downloading anything.

Dates in the sheet are decimals in which the two digits after the point are
the month: 1871.01 is January and 1871.1 is October.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataUnavailable

SHILLER_PAGE = "http://www.econ.yale.edu/~shiller/data.htm"

_WANTED = {"date": "date", "p": "price", "d": "dividend", "e": "earnings", "cpi": "cpi", "rate gs10": "gs10",
           "real total return price": "real_tr_price", "cape": "cape"}


def _norm(x) -> str:
    return " ".join(str(x).replace("\n", " ").split()).strip().lower()


def parse_shiller(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse a sheet read with header=None: find the header row (the one naming CAPE) and the data under it."""
    header_row = None
    for i in range(min(len(raw), 40)):
        cells = [_norm(c) for c in raw.iloc[i].tolist()]
        if "cape" in cells and "date" in cells:
            header_row = i
            break
    if header_row is None:
        raise DataUnavailable("Could not find the header row (with 'Date' and 'CAPE') in the Shiller data.")
    names = [_norm(c) for c in raw.iloc[header_row].tolist()]
    cols = {}
    for j, n in enumerate(names):
        if n in _WANTED and _WANTED[n] not in cols:
            cols[_WANTED[n]] = j
    missing = [v for v in ("date", "price", "dividend", "earnings", "cpi", "cape") if v not in cols]
    if missing:
        raise DataUnavailable(f"Shiller data is missing columns {missing}; found {names}")
    body = raw.iloc[header_row + 1:, list(cols.values())]
    body.columns = list(cols.keys())
    body = body.apply(pd.to_numeric, errors="coerce")
    body = body[body["date"].notna() & (body["date"] > 1800)]
    years = np.floor(body["date"].to_numpy() + 1e-9).astype(int)
    months = np.rint((body["date"].to_numpy() - years) * 100).astype(int)
    ok = (months >= 1) & (months <= 12)
    body = body[ok]
    body.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in zip(years[ok], months[ok])],
                                name="date")
    out = body.drop(columns="date").sort_index()
    out = out[~out.index.duplicated()]
    out.attrs["simulated"] = False
    return out


def read_shiller(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(f"File not found: {p}. Download ie_data.xls from {SHILLER_PAGE}.")
    if p.suffix.lower() == ".csv":
        raw = pd.read_csv(p, header=None)
    else:
        book = pd.ExcelFile(p)
        sheet = next((s for s in book.sheet_names if s.strip().lower() == "data"), book.sheet_names[0])
        raw = book.parse(sheet, header=None)
    return parse_shiller(raw)


def parse_shiller_csv_text(text: str) -> pd.DataFrame:
    return parse_shiller(pd.read_csv(io.StringIO(text), header=None))


def real_returns(df: pd.DataFrame) -> pd.Series:
    """Monthly real total log returns from the real total return price (or from price, dividends and CPI)."""
    if "real_tr_price" in df.columns and df["real_tr_price"].notna().sum() > 1:
        return np.log(df["real_tr_price"]).diff()
    nominal = (df["price"] + df["dividend"] / 12) / df["price"].shift(1)
    return np.log(nominal) - np.log(df["cpi"] / df["cpi"].shift(1))
