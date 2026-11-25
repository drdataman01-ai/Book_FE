"""Holding periods, stopping rules and the disposition effect (Chapter 13).

A selling rule turns a stock's fixed-period returns into an investor's
holding-period return over a random horizon, the stopping time tau. This
module provides:

* Rule                 a selling rule checked at each month end
* exit_times()         when each position is sold, and why
* simulate_paths()     geometric Brownian motion price paths
* summarize()          the distribution of holding-period outcomes
* entries_from_panel() every (asset, entry month) position in a return panel
* excess_per_year()    a rule's log return per year held, against the asset's
                       unconditional mean, with a block bootstrap
* disposition_test()   Odean-style test: sold winners against held losers
* simulate_taxes()     after-tax wealth of selling rules with taxes and costs

Every rule decides using only prices up to the month in which it sells:
changing prices after the sale cannot change the decision (tests check this).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REASONS = ("target", "stop", "even", "cap")


@dataclass(frozen=True)
class Rule:
    """A selling rule, checked at each month end against the purchase price.

    kind
      "fixed"     sell after `cap` months
      "target"    sell when the gain reaches `up`; otherwise sell after `cap` months
      "stop"      sell when the loss reaches `down`; otherwise after `cap` months
      "bracket"   whichever of the target and the stop comes first
      "get_even"  sell when the gain reaches `up`; once the loss has reached
                  `down`, sell as soon as the price is back to the purchase price
    up, down      decimal thresholds (0.2 = 20%); `down` is a positive number
    cap           the longest holding period, in months
    """

    name: str
    kind: str
    up: float = math.inf
    down: float = math.inf
    cap: int = 120

    def __post_init__(self):
        if self.kind not in {"fixed", "target", "stop", "bracket", "get_even"}:
            raise ValueError(f"unknown rule kind {self.kind!r}")
        if self.cap < 1:
            raise ValueError("cap must be at least one month")


def standard_rules(up: float = 0.2, down: float = 0.2, cap: int = 120, fixed: int = 12) -> list[Rule]:
    """The five rules used in the lab, plus buy-and-hold for the whole cap."""
    return [
        Rule(f"Fixed {fixed} months", "fixed", cap=fixed),
        Rule(f"Hold {cap} months", "fixed", cap=cap),
        Rule(f"Profit target +{up:.0%}", "target", up=up, cap=cap),
        Rule(f"Stop-loss -{down:.0%}", "stop", down=down, cap=cap),
        Rule(f"Bracket +{up:.0%} / -{down:.0%}", "bracket", up=up, down=down, cap=cap),
        Rule(f"Get even (+{up:.0%}; after -{down:.0%}, sell at cost)", "get_even", up=up, down=down, cap=cap),
    ]


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True in each row, or the row length if there is none."""
    has = mask.any(axis=1)
    return np.where(has, mask.argmax(axis=1), mask.shape[1])


def exit_times(rel: np.ndarray, rule: Rule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """When each position is sold under `rule`.

    rel   array (n, L): rel[j, k-1] is the gain of position j at the end of
          month k after purchase, P_k / P_0 - 1. NaN is not allowed inside
          the first `cap` months.
    Returns (tau, hpr, reason): months held (1..cap), the holding-period
    return at the sale, and the reason code for the sale.
    """
    rel = np.asarray(rel, dtype=float)
    if rel.ndim != 2:
        raise ValueError("rel must be two-dimensional")
    n, L = rel.shape
    cap = min(rule.cap, L)
    r = rel[:, :cap]
    if np.isnan(r).any():
        raise ValueError("rel has missing values inside the holding window")
    never = np.full(n, cap)
    t_up = _first_true(r >= rule.up) if rule.kind in {"target", "bracket", "get_even"} else never
    t_dn = _first_true(r <= -rule.down) if rule.kind in {"stop", "bracket", "get_even"} else never
    if rule.kind == "get_even":
        k = np.arange(cap)[None, :]
        t_even = _first_true((k > t_dn[:, None]) & (r >= 0.0))
        t_stop = never
    else:
        t_even = never
        t_stop = t_dn if rule.kind in {"stop", "bracket"} else never
    t = np.minimum.reduce([t_up, t_stop, t_even, np.full(n, cap - 1)])
    tau = t + 1
    hpr = r[np.arange(n), t]
    reason = np.full(n, "cap", dtype=object)
    reason[t_even == t] = "even"
    reason[t_stop == t] = "stop"
    reason[t_up == t] = "target"
    return tau, hpr, reason


def simulate_paths(n: int, months: int, mu: float = 0.0, sigma: float = 0.30, seed: int = 0) -> np.ndarray:
    """Monthly gains P_k / P_0 - 1 of geometric Brownian motion, shape (n, months).

    `mu` is the annual expected (arithmetic) return. With mu = 0 the price is
    a martingale: its expected value one month ahead is today's price.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12
    steps = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal((n, months))
    return np.expm1(np.cumsum(steps, axis=1))


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / s**3) if s > 0 else 0.0


def summarize(hpr: np.ndarray, tau: np.ndarray, reason: np.ndarray) -> dict:
    """Distribution of holding-period outcomes for one rule.

    per_year_log    total log return divided by total years held: the growth
                    rate of money while it is invested under the rule
    naive_annual    average of each position's return annualized on its own,
                    (1 + R)^(12 / tau) - 1: the figure a brochure would quote
    """
    hpr = np.asarray(hpr, dtype=float)
    tau = np.asarray(tau, dtype=float)
    reason = np.asarray(reason)
    q = np.quantile(hpr, [0.05, 0.25, 0.5, 0.75, 0.95])
    losers_at_cap = (reason == "cap") & (hpr < 0)
    return {
        "n": int(len(hpr)),
        "mean_hpr": float(hpr.mean()),
        "se_mean_hpr": float(hpr.std(ddof=1) / math.sqrt(len(hpr))),
        "median_hpr": float(q[2]),
        "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]), "q95": float(q[4]),
        "win_rate": float((hpr > 0).mean()),
        "mean_gain_if_win": float(hpr[hpr > 0].mean()) if (hpr > 0).any() else float("nan"),
        "mean_loss_if_loss": float(hpr[hpr <= 0].mean()) if (hpr <= 0).any() else float("nan"),
        "skewness": _skew(hpr),
        "mean_months": float(tau.mean()),
        "share_at_cap": float((reason == "cap").mean()),
        "share_losers_at_cap": float(losers_at_cap.mean()),
        "per_year_log": float(np.log1p(hpr).sum() / (tau.sum() / 12)),
        "naive_annual": float(np.mean((1 + hpr) ** (12 / tau) - 1)),
    }


# ---------------------------------------------------------------------------
# Real data: a panel of portfolios
# ---------------------------------------------------------------------------


def log_prices(returns: pd.DataFrame) -> np.ndarray:
    """Log price index, shape (T+1, N), starting at 0; NaN before an asset's first return."""
    lr = np.log1p(returns.to_numpy(dtype=float))
    lp = np.vstack([np.zeros(lr.shape[1]), np.cumsum(np.nan_to_num(lr), axis=0)])
    first = np.argmax(np.isfinite(lr), axis=0)
    for j, f in enumerate(first):
        lp[:f, j] = np.nan  # no price before the first observed return
    lp[1:][~np.isfinite(lr)] = np.nan
    return lp


def entries_from_panel(returns: pd.DataFrame, cap: int, step: int = 1) -> dict:
    """Every position bought at a month end with `cap` complete months of data after it.

    Returns {"asset": i, "start": e, "rel": (n, cap)}, where `start` indexes
    the log-price array of log_prices() (0 = before the first month).
    """
    lp = log_prices(returns)
    T1, N = lp.shape
    k = np.arange(1, cap + 1)
    starts, assets = [], []
    for e in range(0, T1 - cap, step):
        window = lp[e:e + cap + 1]
        ok = np.isfinite(window).all(axis=0)
        assets.append(np.nonzero(ok)[0])
        starts.append(np.full(ok.sum(), e))
    asset = np.concatenate(assets)
    start = np.concatenate(starts)
    rel = np.expm1(lp[start[:, None] + k[None, :], asset[:, None]] - lp[start, asset][:, None])
    return {"asset": asset, "start": start, "rel": rel, "log_prices": lp}


def mean_log_return(returns: pd.DataFrame) -> np.ndarray:
    """Each asset's average monthly log return over the whole sample."""
    return np.nanmean(np.log1p(returns.to_numpy(dtype=float)), axis=0)


def excess_per_year(pos: dict, tau: np.ndarray, hpr: np.ndarray, g: np.ndarray, block: int = 60,
                    n_boot: int = 400, seed: int = 0) -> dict:
    """A rule's log return per year held, minus what the assets earned on average per year.

    For position j in asset i held tau_j months, the shortfall is
    log(1 + R_j) - g_i * tau_j, where g_i is asset i's average monthly log
    return over the whole sample. The estimate is 12 * sum(shortfall) /
    sum(tau). Standard errors come from a moving-block bootstrap over entry
    months, because positions bought in nearby months share returns.
    """
    d = np.log1p(hpr) - g[pos["asset"]] * tau
    start = pos["start"]
    months = np.unique(start)
    sd = pd.Series(d).groupby(start).sum().reindex(months).to_numpy()
    st = pd.Series(tau.astype(float)).groupby(start).sum().reindex(months).to_numpy()
    est = 12 * sd.sum() / st.sum()
    rng = np.random.default_rng(seed)
    m = len(months)
    block = max(1, min(block, m))
    n_blocks = int(math.ceil(m / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.integers(0, m - block + 1, n_blocks)[:, None] + np.arange(block)[None, :]).ravel()[:m]
        boots[b] = 12 * sd[idx].sum() / st[idx].sum()
    se = float(boots.std(ddof=1))
    return {"excess_per_year": float(est), "se": se, "t": float(est / se) if se > 0 else float("nan"),
            "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))}


def disposition_test(pos: dict, rule: Rule, horizon: int = 12, nw_lags: int | None = None) -> dict:
    """Do the winners a disposition investor sells beat the losers the investor keeps?

    Uses `rule` (normally a profit target with no stop) on every position.
    In each month s, the sold winners are positions sold that month at the
    target; the held losers are positions still open at the end of month s
    and below their purchase price. For each, the outcome is the asset's log
    return over the next `horizon` months, which the investor no longer
    earns (sold) or still bears (held). The test statistic is the average,
    across months, of (mean sold winner) - (mean held loser), with Newey-West
    standard errors for the overlap of horizons.
    """
    from fintest.stats import ols_newey_west

    lp = pos["log_prices"]
    T1 = lp.shape[0]
    fut = np.full_like(lp, np.nan)
    fut[:T1 - horizon] = lp[horizon:] - lp[:T1 - horizon]
    cnt_all = np.isfinite(fut).sum(axis=1)
    ew = np.where(cnt_all > 0, np.nansum(fut, axis=1) / np.maximum(cnt_all, 1), np.nan)  # equal-weighted, by month

    tau, hpr, reason = exit_times(pos["rel"], rule)
    n, cap = pos["rel"].shape
    k = np.arange(1, cap + 1)[None, :]
    s = pos["start"][:, None] + k  # price index of the end of month k
    a = np.broadcast_to(pos["asset"][:, None], s.shape)
    f = fut[s, a]
    sold_w = (k == tau[:, None]) & (reason == "target")[:, None]
    held_l = (k < tau[:, None]) & (pos["rel"] < 0)
    out = {}
    for name, mask in (("sold_winners", sold_w), ("held_losers", held_l)):
        mm = mask & np.isfinite(f)
        cnt = np.bincount(s[mm], minlength=T1).astype(float)
        tot = np.bincount(s[mm], weights=f[mm], minlength=T1)
        out[name] = (cnt, tot)
    cw, tw = out["sold_winners"]
    cl, tl = out["held_losers"]
    both = (cw > 0) & (cl > 0) & np.isfinite(ew)
    mw = tw[both] / cw[both]
    ml = tl[both] / cl[both]
    diff = mw - ml
    lags = horizon - 1 if nw_lags is None else nw_lags
    nw = ols_newey_west(diff, np.zeros((len(diff), 0)), lags=lags) if len(diff) > horizon else None
    res = {
        "months": int(both.sum()),
        "month_index": np.nonzero(both)[0],
        "sold_winners_mean": float(mw.mean()),
        "held_losers_mean": float(ml.mean()),
        "ew_mean": float(ew[both].mean()),
        "diff_mean": float(diff.mean()),
        "diff_t": float(nw["t"][0, 0]) if nw else float("nan"),
        "diff_series": diff,
        "n_sold_winners": int(cw.sum()),
        "n_held_loser_months": int(cl.sum()),
        "horizon": horizon,
    }
    return res


# ---------------------------------------------------------------------------
# Taxes and costs
# ---------------------------------------------------------------------------


def simulate_taxes(kind: str, years: int = 30, n: int = 20000, mu: float = 0.08, sigma: float = 0.30,
                   tax: float = 0.20, cost: float = 0.001, up: float = 0.2, down: float = 0.2,
                   every: int = 12, seed: int = 0) -> dict:
    """After-tax, after-cost wealth of an investor who follows one selling rule for `years`.

    kind  "hold"     buy once and sell at the end
          "every"    sell and buy again every `every` months
          "winners"  sell when the gain reaches `up` (the disposition investor)
          "losers"   sell when the loss reaches `down` (tax-loss harvesting)
    Each sale pays the one-way `cost` and tax at rate `tax` on the realized
    gain; a realized loss earns a tax credit at the same rate, used at once.
    The proceeds buy a new position (another draw of the same price process),
    paying `cost` again. Everything is sold, and taxed, at the end.
    """
    if kind not in {"hold", "every", "winners", "losers"}:
        raise ValueError(f"unknown kind {kind!r}")
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12
    months = years * 12
    value = np.full(n, 1.0 - cost)  # market value of the position
    basis = np.ones(n)  # amount paid, including the cost of buying
    age = np.zeros(n, dtype=int)
    sales = np.zeros(n)
    for m in range(months):
        value *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n))
        age += 1
        if m == months - 1:
            break
        gain = value / basis - 1
        if kind == "hold":
            sell = np.zeros(n, dtype=bool)
        elif kind == "every":
            sell = age >= every
        elif kind == "winners":
            sell = gain >= up
        else:
            sell = gain <= -down
        if sell.any():
            proceeds = value[sell] * (1 - cost)
            cash = proceeds - tax * (proceeds - basis[sell])
            basis[sell] = cash
            value[sell] = cash * (1 - cost)
            age[sell] = 0
            sales[sell] += 1
    proceeds = value * (1 - cost)
    wealth = proceeds - tax * (proceeds - basis)
    growth = np.log(wealth) / years
    return {"kind": kind, "mean_wealth": float(wealth.mean()), "median_wealth": float(np.median(wealth)),
            "annualized_mean_wealth": float(wealth.mean() ** (1 / years) - 1),
            "mean_log_growth": float(growth.mean()), "sales_per_year": float(sales.mean() / years),
            "wealth": wealth}
