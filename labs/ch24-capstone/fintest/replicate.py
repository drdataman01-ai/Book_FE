"""Replication project tools (Chapter 24).

* fingerprint - SHA-256 of every input file, so a reader can confirm they use the same data
* environment - Python and package versions
* RunLog - an append-only JSON-lines log of every run: time, inputs, parameters, outputs, environment
* compare_to_original - reproduced statistics against the published ones, with tolerances and a status per item
* replication_status - an overall label (reproduced, partly reproduced, not reproduced) from the comparison
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fingerprint(paths, base=None) -> dict:
    """SHA-256 and size of each file, keyed by its path relative to `base` when possible."""
    out = {}
    base = Path(base).resolve() if base else None
    for p in map(Path, paths):
        key = str(p)
        if base is not None:
            try:
                key = str(p.resolve().relative_to(base))
            except ValueError:
                pass
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[key] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
        else:
            out[key] = {"sha256": None, "bytes": None, "note": "missing"}
    return out


def environment(packages=("numpy", "pandas", "scipy", "matplotlib")) -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in packages:
        try:
            env[p] = __import__(p).__version__
        except Exception:  # noqa: BLE001
            env[p] = None
    return env


class RunLog:
    """Append one JSON record per run to `path`. Records are never rewritten, so the log shows every attempt,
    including failed and abandoned ones."""

    def __init__(self, path, base=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = Path(base) if base else self.path.resolve().parent.parent   # project root

    def record(self, step: str, inputs=(), params: dict | None = None, outputs: dict | None = None,
               note: str = "") -> dict:
        rec = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "step": step,
               "inputs": fingerprint(inputs, self.base), "params": params or {}, "outputs": outputs or {},
               "environment": environment(), "note": note}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def compare_to_original(original: dict, reproduced: dict, rel_tol: float = 0.10, abs_tol: float = 0.0) -> pd.DataFrame:
    """One row per statistic reported in the original paper. Status:
    'match' if |reproduced - original| <= max(abs_tol, rel_tol x |original|);
    'same sign, different size' if outside tolerance but with the same sign;
    'different sign'; or 'not reproduced' if missing."""
    rows = []
    for k, o in original.items():
        r = reproduced.get(k)
        if r is None:
            status, diff = "not reproduced", None
        else:
            diff = r - o
            tol = max(abs_tol, rel_tol * abs(o))
            if abs(diff) <= tol:
                status = "match"
            elif (r > 0) == (o > 0):
                status = "same sign, different size"
            else:
                status = "different sign"
        rows.append({"statistic": k, "original": o, "reproduced": r, "difference": diff, "status": status})
    return pd.DataFrame(rows)


def replication_status(table: pd.DataFrame) -> str:
    s = table["status"]
    if (s == "match").all():
        return "reproduced"
    if (s == "different sign").any() or (s == "not reproduced").mean() > 0.5:
        return "not reproduced"
    return "partly reproduced"
