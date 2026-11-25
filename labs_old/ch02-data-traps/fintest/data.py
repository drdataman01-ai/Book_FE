"""Data access for the book's labs: the Kenneth R. French Data Library.

The library publishes returns as zipped CSV files. Some files hold one table
(the factor files); others hold several, one after another, each introduced
by a title line such as "Average Value Weighted Returns -- Monthly". This
module downloads a file once, caches it under data/, and parses tables into
pandas DataFrames of decimal returns (the files report percent).

Ways to get data:

1. load_french_factors() / load_french_tables()   download (or use the cache)
2. pass path="my_file.zip"                         a file you downloaded yourself
3. simulated_* functions                           fake data, for testing code only

The loaders never fall back to simulated data on their own: a lab that
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
    # name: (zip file, frequency, title shown on the library page)
    "ff3_monthly": ("F-F_Research_Data_Factors_CSV.zip", "M", "Fama/French 3 Factors"),
    "ff3_daily": ("F-F_Research_Data_Factors_daily_CSV.zip", "D", "Fama/French 3 Factors [Daily]"),
    "momentum_monthly": ("F-F_Momentum_Factor_CSV.zip", "M", "Momentum Factor (Mom)"),
    "size_portfolios": ("Portfolios_Formed_on_ME_CSV.zip", "M", "Portfolios Formed on Size"),
}

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"


class DataUnavailable(RuntimeError):
    """Raised when the data cannot be downloaded or found."""


def _download(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fintest/0.2 (textbook lab)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_text(raw: bytes) -> str:
    """Return the CSV text from a zip archive or a plain CSV file."""
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")] or zf.namelist()
            raw = zf.read(names[0])
    return raw.decode("latin-1")


def _is_header(cells: list[str]) -> bool:
    return len(cells) >= 2 and cells[0] == "" and any(re.search(r"[A-Za-z<>=]", c) for c in cells[1:])


def parse_french_tables(text: str, freq: str = "M") -> dict[str, pd.DataFrame]:
    """Parse every table in a French library CSV whose dates match `freq`.

    Returns {title: DataFrame}, in file order. A table is a header row
    (first cell empty) followed by rows keyed by date: YYYYMM for monthly,
    YYYYMMDD for daily. Annual tables (YYYY keys) are skipped. The title is
    the nearest non-empty line above the header, or "table N" if there is
    none. Values are converted from percent to decimals for return tables;
    tables whose title mentions firms or size (counts, average market cap)
    are left in their original units. -99.99 and -999 mean missing.
    """
    key_len = {"M": 6, "D": 8}[freq]
    lines = text.splitlines()
    tables: dict[str, pd.DataFrame] = {}
    i, n_found = 0, 0
    while i < len(lines):
        cells = [c.strip() for c in lines[i].split(",")]
        if not _is_header(cells):
            i += 1
            continue
        columns = cells[1:]
        title = ""
        for j in range(i - 1, -1, -1):
            if lines[j].strip():
                title = lines[j].strip().strip(",").strip()
                break
        dates, rows = [], []
        k = i + 1
        while k < len(lines):
            row = [c.strip() for c in lines[k].split(",")]
            if not row or not row[0].isdigit():
                break
            if len(row[0]) == key_len:
                dates.append(row[0])
                rows.append([float(c) if c else np.nan for c in row[1:len(columns) + 1]])
            k += 1
        if rows:
            n_found += 1
            if not title or title in tables or "," in title:
                title = title if title and title not in tables and "," not in title else f"table {n_found}"
            df = pd.DataFrame(rows, columns=columns[:len(rows[0])])
            df = df.replace([-99.99, -999.0], np.nan)
            if not re.search(r"firms|size", title, re.IGNORECASE):
                df = df / 100.0
            if freq == "M":
                df.index = pd.PeriodIndex([f"{d[:4]}-{d[4:]}" for d in dates], freq="M")
            else:
                df.index = pd.to_datetime(dates, format="%Y%m%d")
            df.index.name = "date"
            tables[title] = df
        i = max(k, i + 1)
    if not tables:
        raise ValueError("no tables with the expected date format were found in the French data file")
    return tables


def parse_french_csv(text: str, freq: str = "M") -> pd.DataFrame:
    """Parse the first table in a French library CSV file (decimal returns)."""
    return next(iter(parse_french_tables(text, freq).values()))


def _raw_bytes(dataset: str, path, cache_dir: Path, refresh: bool) -> bytes:
    filename, _, title = DATASETS[dataset]
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise DataUnavailable(f"{p} does not exist")
        return p.read_bytes()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / filename
    if refresh or not cached.exists():
        url = FTP_BASE + filename
        try:
            cached.write_bytes(_download(url))
        except Exception as e:  # noqa: BLE001
            raise DataUnavailable(
                f"Could not download {url} ({type(e).__name__}: {e}).\n"
                f"Download '{title}' (CSV) manually from {LIBRARY_URL}\n"
                f"and save it as {cached}, or pass its path on the command line.\n"
                "To test the code without real data, use --source sim (results will be fake)."
            ) from e
    return cached.read_bytes()


def load_french_factors(dataset: str = "ff3_monthly", path: str | Path | None = None,
                        cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Load the first table of a French library dataset as decimal returns."""
    freq = DATASETS[dataset][1]
    return parse_french_csv(_read_text(_raw_bytes(dataset, path, cache_dir, refresh)), freq)


def load_french_tables(dataset: str, path: str | Path | None = None,
                       cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load every table of a French library dataset."""
    freq = DATASETS[dataset][1]
    return parse_french_tables(_read_text(_raw_bytes(dataset, path, cache_dir, refresh)), freq)


def find_table(tables: dict[str, pd.DataFrame], *words: str) -> pd.DataFrame:
    """Return the first table whose title contains all `words` (case-insensitive)."""
    for title, df in tables.items():
        if all(w.lower() in title.lower() for w in words):
            return df
    raise KeyError(f"no table titled with {words}; available: {list(tables)}")


# ---------------------------------------------------------------------------
# Simulated data, for testing code only
# ---------------------------------------------------------------------------


def simulated_market(n_months: int = 1188, seed: int = 0, start: str = "1926-07") -> pd.DataFrame:
    """Fake monthly market data with realistic first and second moments.

    Monthly excess returns are Student-t (fat tails) scaled to a mean of
    0.65% and a standard deviation of 5.3%; the risk-free rate is 0.28%.
    FOR TESTING CODE ONLY: results from this data are not evidence.
    """
    rng = np.random.default_rng(seed)
    dof = 5
    shocks = rng.standard_t(dof, size=n_months) / np.sqrt(dof / (dof - 2))
    idx = pd.period_range(start=start, periods=n_months, freq="M")
    df = pd.DataFrame({"Mkt-RF": 0.0065 + 0.053 * shocks, "RF": 0.0028}, index=idx)
    df.index.name = "date"
    df.attrs["simulated"] = True
    return df


def simulated_daily_market(n_days: int = 25000, seed: int = 0, start: str = "1926-07-01") -> pd.DataFrame:
    """Fake daily market data with volatility clustering (GARCH-like). Testing only."""
    rng = np.random.default_rng(seed)
    omega, a, b = 1.5e-6, 0.08, 0.90
    var = omega / (1 - a - b)
    out = np.empty(n_days)
    for t in range(n_days):
        z = rng.standard_t(6) / np.sqrt(1.5)
        out[t] = 0.00028 + np.sqrt(var) * z
        var = omega + a * (out[t] - 0.00028) ** 2 + b * var
    idx = pd.bdate_range(start=start, periods=n_days)
    df = pd.DataFrame({"Mkt-RF": out, "RF": 0.00011}, index=idx)
    df.index.name = "date"
    df.attrs["simulated"] = True
    return df


def simulated_size_deciles(n_months: int = 1188, seed: int = 0, start: str = "1926-07") -> dict[str, pd.DataFrame]:
    """Fake value- and equal-weighted size-decile returns. Testing only.

    The smallest decile is given a premium in the first 55 years only, so the
    code paths for "before and after publication" have something to find.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start=start, periods=n_months, freq="M")
    mkt = 0.0093 + 0.053 * rng.standard_normal(n_months)
    cols = ["Lo 10"] + [f"Dec {i}" for i in range(2, 10)] + ["Hi 10"]
    vw, ew = {}, {}
    for d, col in enumerate(cols):
        beta = 1.35 - 0.04 * d
        idio = 0.045 - 0.0042 * d
        early = (np.arange(n_months) < 660) * (0.006 if d == 0 else 0.0)
        base = beta * mkt + idio * rng.standard_normal(n_months) + early
        vw[col] = base
        ew[col] = base + (0.004 if d == 0 else 0.0005) + 0.01 * rng.standard_normal(n_months) * (d == 0)
    tables = {
        "Average Value Weighted Returns -- Monthly": pd.DataFrame(vw, index=idx),
        "Average Equal Weighted Returns -- Monthly": pd.DataFrame(ew, index=idx),
    }
    for df in tables.values():
        df.index.name = "date"
        df.attrs["simulated"] = True
    return tables
