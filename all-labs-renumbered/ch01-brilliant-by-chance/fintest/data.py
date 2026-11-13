"""Data access for the book's labs: the Kenneth R. French Data Library.

The library publishes factor returns as zipped CSV files. This module
downloads a file once, caches it under data/, and parses it into a pandas
DataFrame of decimal returns (the files report percent).

Three ways to get data:

1. load_french_factors()                     download (or use the cache)
2. load_french_factors(path="my_file.zip")   a file you downloaded yourself
3. simulated_market(...)                     fake data with realistic moments,
                                              for testing code only

The loader never falls back to simulated data on its own: a lab that
quietly used fake data would produce fake conclusions.

Data source: Kenneth R. French Data Library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Please cite the library when you publish results that use it.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
FTP_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASETS = {
    # name: (zip file, frequency)
    "ff3_monthly": ("F-F_Research_Data_Factors_CSV.zip", "M"),
    "ff3_daily": ("F-F_Research_Data_Factors_daily_CSV.zip", "D"),
    "momentum_monthly": ("F-F_Momentum_Factor_CSV.zip", "M"),
}

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"


class DataUnavailable(RuntimeError):
    """Raised when the data cannot be downloaded or found."""


def _download(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fintest/0.1 (textbook lab)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_text(raw: bytes) -> str:
    """Return the CSV text from a zip archive or a plain CSV file."""
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")] or zf.namelist()
            raw = zf.read(names[0])
    return raw.decode("latin-1")


def parse_french_csv(text: str, freq: str = "M") -> pd.DataFrame:
    """Parse the first table in a French library CSV file.

    The files contain a few lines of description, a header row such as
    ',Mkt-RF,SMB,HML,RF', then rows keyed by date (YYYYMM for monthly,
    YYYYMMDD for daily). A second, annual table follows after a blank line;
    parsing stops when the date key no longer has the expected length.
    Values are percent and are converted to decimals; -99.99 and -999 are
    the library's missing-value codes.
    """
    key_len = {"M": 6, "D": 8}[freq]
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 2 and cells[0] == "" and any(re.search(r"[A-Za-z]", c) for c in cells[1:]):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("could not find the header row in the French data file")
    columns = [c.strip() for c in lines[header_idx].split(",")[1:]]

    dates, rows = [], []
    for line in lines[header_idx + 1:]:
        cells = [c.strip() for c in line.split(",")]
        if not cells or not cells[0].isdigit() or len(cells[0]) != key_len:
            if rows:
                break
            continue
        dates.append(cells[0])
        rows.append([float(c) if c else np.nan for c in cells[1:len(columns) + 1]])
    if not rows:
        raise ValueError("no data rows found in the French data file")

    df = pd.DataFrame(rows, columns=columns)
    df = df.replace([-99.99, -999.0], np.nan) / 100.0
    if freq == "M":
        df.index = pd.PeriodIndex([f"{d[:4]}-{d[4:]}" for d in dates], freq="M")
    else:
        df.index = pd.to_datetime(dates, format="%Y%m%d")
    df.index.name = "date"
    return df


def load_french_factors(
    dataset: str = "ff3_monthly",
    path: str | Path | None = None,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a French library dataset as decimal returns.

    path      a zip or CSV you downloaded yourself (skips the download)
    refresh   download again even if a cached copy exists
    """
    filename, freq = DATASETS[dataset]
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise DataUnavailable(f"{p} does not exist")
        return parse_french_csv(_read_text(p.read_bytes()), freq)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / filename
    if refresh or not cached.exists():
        url = FTP_BASE + filename
        try:
            cached.write_bytes(_download(url))
        except Exception as e:  # noqa: BLE001
            raise DataUnavailable(
                f"Could not download {url} ({type(e).__name__}: {e}).\n"
                f"Download the file manually from {LIBRARY_URL}\n"
                f"and pass it with --data-file, or save it as {cached}.\n"
                "To test the code without real data, use --source sim (results will be fake)."
            ) from e
    return parse_french_csv(_read_text(cached.read_bytes()), freq)


def simulated_market(n_months: int = 1188, seed: int = 0, start: str = "1926-07") -> pd.DataFrame:
    """Fake monthly market data with realistic first and second moments.

    Monthly excess returns are drawn from a Student-t distribution (fat
    tails) scaled to a mean of 0.65% and a standard deviation of 5.3%,
    roughly matching U.S. history; the risk-free rate is a constant 0.28%.
    FOR TESTING CODE ONLY: results from this data are not evidence.
    """
    rng = np.random.default_rng(seed)
    dof = 5
    shocks = rng.standard_t(dof, size=n_months) / np.sqrt(dof / (dof - 2))
    mkt_rf = 0.0065 + 0.053 * shocks
    idx = pd.period_range(start=start, periods=n_months, freq="M")
    df = pd.DataFrame({"Mkt-RF": mkt_rf, "RF": 0.0028}, index=idx)
    df.index.name = "date"
    df.attrs["simulated"] = True
    return df
