"""Searching over many trading rules, and correcting for the search (Chapters 5 and 6).

* Technical trading signals: moving-average crossovers and trading-range breakouts, with
  bands and fixed holding periods (Brock, Lakonishok and LeBaron 1992)
* A rule universe of over a thousand variations (Sullivan, Timmermann and White 1999)
* White's (2000) Reality Check and Hansen's (2005) test for superior predictive ability (SPA),
  both with the stationary bootstrap of Politis and Romano (1994)

Every signal at date t uses prices up to and including t; it is applied to the return of t+1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def _ma(p: np.ndarray, n: int) -> np.ndarray:
    """Trailing n-day average of p (NaN for the first n-1 days)."""
    out = np.full(len(p), np.nan)
    if n > len(p):
        return out
    c = np.cumsum(np.insert(p, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _rolling_extreme(p: np.ndarray, n: int, fn) -> np.ndarray:
    """Max or min of the n prices before day t (excluding t), NaN where unavailable."""
    s = pd.Series(p).shift(1).rolling(n, min_periods=n)
    return (s.max() if fn == "max" else s.min()).to_numpy()


def ma_signal(logp: np.ndarray, short: int, long: int, band: float = 0.0) -> np.ndarray:
    """+1 (buy) when the short average is above the long one by more than `band`, -1 (sell) when below by more
    than `band`, 0 otherwise. Averages are of the price level, as in Brock, Lakonishok and LeBaron."""
    p = np.exp(logp)
    s, l = _ma(p, short), _ma(p, long)
    sig = np.zeros(len(p))
    sig[s > l * (1 + band)] = 1
    sig[s < l * (1 - band)] = -1
    sig[np.isnan(l)] = 0
    return sig


def breakout_signal(logp: np.ndarray, n: int, band: float = 0.0) -> np.ndarray:
    """+1 when the price rises above its previous n-day high by more than `band`, -1 when it falls below its
    previous n-day low by more than `band`, 0 otherwise (a signal on the day it happens)."""
    p = np.exp(logp)
    hi, lo = _rolling_extreme(p, n, "max"), _rolling_extreme(p, n, "min")
    sig = np.zeros(len(p))
    with np.errstate(invalid="ignore"):
        sig[p > hi * (1 + band)] = 1
        sig[p < lo * (1 - band)] = -1
    return sig


def crossings(sig: np.ndarray) -> np.ndarray:
    """Days on which a state signal switches into buy (+1) or sell (-1); 0 elsewhere."""
    out = np.zeros(len(sig))
    prev = np.concatenate([[0.0], sig[:-1]])
    out[(sig == 1) & (prev != 1)] = 1
    out[(sig == -1) & (prev != -1)] = -1
    return out


def hold_after_events(events: np.ndarray, hold: int) -> np.ndarray:
    """Position +1 / -1 for `hold` days after each buy / sell event (the event day included), 0 otherwise.
    A later event replaces an earlier one."""
    T = len(events)
    pos = np.zeros(T)
    idx = np.flatnonzero(events)
    for k, t in enumerate(idx):
        end = min(T, t + hold)
        if k + 1 < len(idx):
            end = min(end, idx[k + 1])
        pos[t:end] = events[t]
    return pos


def long_flat_position(state: np.ndarray) -> np.ndarray:
    """Turn a buy/sell/neutral state into a long-or-cash position, keeping the last decision through neutral days."""
    s = pd.Series(np.where(state == 0, np.nan, state)).ffill().fillna(-1).to_numpy()
    return (s > 0).astype(float)


# ---------------------------------------------------------------------------
# The rule universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    kind: str          # "MA" or "TRB"
    short: int         # short average (MA) or 0 (TRB)
    long: int          # long average (MA) or breakout window (TRB)
    band: float
    hold: int          # 0 = hold until the opposite signal; otherwise fixed days after a signal

    @property
    def name(self) -> str:
        base = f"MA({self.short},{self.long})" if self.kind == "MA" else f"TRB({self.long})"
        return base + f" b={self.band:.1%}" + (f" h={self.hold}" if self.hold else "")


def rule_universe(shorts=(1, 2, 5, 10, 20, 30, 50), longs=(25, 50, 75, 100, 125, 150, 200, 250),
                  breakouts=(5, 10, 15, 20, 25, 50, 100, 150, 200, 250),
                  bands=(0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.05), holds=(0, 5, 10, 25)) -> list[Rule]:
    rules = [Rule("MA", s, l, b, h) for s in shorts for l in longs if s < l for b in bands for h in holds]
    rules += [Rule("TRB", 0, n, b, h) for n in breakouts for b in bands for h in holds]
    return rules


def rule_positions(logp: np.ndarray, rules: list[Rule]) -> np.ndarray:
    """(T, K) long-or-cash positions decided at the close of each day (apply to the next day's return)."""
    T = len(logp)
    out = np.zeros((T, len(rules)), dtype=np.float32)
    cache: dict = {}
    for k, r in enumerate(rules):
        key = (r.kind, r.short, r.long, r.band)
        if key not in cache:
            cache[key] = ma_signal(logp, r.short, r.long, r.band) if r.kind == "MA" else breakout_signal(logp, r.long, r.band)
        state = cache[key]
        if r.hold:
            ev = crossings(state) if r.kind == "MA" else state
            out[:, k] = (hold_after_events(ev, r.hold) > 0)
        elif r.kind == "MA":
            out[:, k] = long_flat_position(state)
        else:
            out[:, k] = long_flat_position(state)
    return out


# ---------------------------------------------------------------------------
# Reality Check and SPA
# ---------------------------------------------------------------------------


def stationary_bootstrap_matrix(T: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, T) resampled time indices in blocks of geometric length (mean `mean_block`), wrapping around
    the end of the sample (Politis and Romano 1994). Vectorized."""
    p = 1.0 / mean_block
    new = rng.random((n_boot, T)) < p
    new[:, 0] = True
    starts = rng.integers(T, size=(n_boot, T))
    pos = np.arange(T)[None, :]
    last_start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    block_start = np.take_along_axis(starts, last_start, axis=1)
    return (block_start + pos - last_start) % T


def bootstrap_means(d: np.ndarray, n_boot: int, mean_block: float, rng: np.random.Generator,
                    chunk: int = 100) -> np.ndarray:
    """(n_boot, K) means of the columns of d under the stationary bootstrap, computed through index counts."""
    T, K = d.shape
    out = np.empty((n_boot, K))
    d64 = np.asarray(d, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = stationary_bootstrap_matrix(T, m, mean_block, rng)
        flat = (np.arange(m)[:, None] * T + idx).ravel()
        counts = np.bincount(flat, minlength=m * T).reshape(m, T).astype(np.float64)
        out[done:done + m] = counts @ d64 / T
        done += m
    return out


def reality_check(d: np.ndarray, n_boot: int = 1000, mean_block: float = 10, seed: int = 0) -> dict:
    """White's Reality Check and Hansen's SPA test of H0: no rule beats the benchmark.

    d is (T, K): each rule's performance minus the benchmark's, period by period (higher is better).
    Returns the best rule, its nominal p-value, and the Reality Check and SPA (consistent) p-values.
    """
    d = np.asarray(d, dtype=float)
    ok = np.isfinite(d).all(axis=1)
    d = d[ok]
    T, K = d.shape
    rng = np.random.default_rng(seed)
    mean = d.mean(axis=0)
    boot = bootstrap_means(d, n_boot, mean_block, rng)
    centred = boot - mean
    # Reality Check (White 2000)
    v = math.sqrt(T) * mean.max()
    v_star = math.sqrt(T) * centred.max(axis=1)
    p_rc = float((v_star >= v).mean())
    # SPA (Hansen 2005): studentize, and recentre only rules that are not clearly worse than the benchmark
    omega = np.sqrt(T) * centred.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)  # a rule identical to the benchmark carries no information
    with np.errstate(divide="ignore", invalid="ignore"):
        t = math.sqrt(T) * mean / omega
    stat = max(0.0, np.nanmax(t))
    thresh = -np.sqrt((omega ** 2 / T) * 2 * math.log(math.log(T)))
    g = np.where(mean >= thresh, mean, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = math.sqrt(T) * (boot - g) / omega
    stat_star = np.maximum(0.0, np.nanmax(z, axis=1))
    p_spa = float((stat_star >= stat).mean())
    best = int(np.argmax(mean))
    sd = d[:, best].std(ddof=1)
    t_best = mean[best] / (sd / math.sqrt(T)) if sd > 0 else float("nan")
    p_nominal = 0.5 * math.erfc(t_best / math.sqrt(2)) if np.isfinite(t_best) else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        t_all = mean / (d.std(axis=0, ddof=1) / math.sqrt(T))
    p_all = np.array([0.5 * math.erfc(x / math.sqrt(2)) if np.isfinite(x) else 1.0 for x in t_all])
    return {"best": best, "best_mean": float(mean[best]), "best_t": float(t_best), "p_nominal": float(p_nominal),
            "p_reality_check": p_rc, "p_spa": p_spa, "n_rules": K, "n_obs": T,
            "n_nominal_5pct": int((p_all < 0.05).sum())}
