"""Trading costs, turnover and capacity (Chapter 14).

* Cost accounting for long-short portfolios: monthly cost, net return and
  break-even cost from one-sided turnover
* A simulated stock universe with value, momentum and short-term reversal
  signals built the way the Fama-French factors are built, used to measure
  turnover mechanically and to test cost-mitigation rules
* Buy/hold spreads (banding)
* Square-root market impact and strategy capacity

Units: returns, costs and turnover are decimals per month unless a function
says otherwise. "One-way cost" is the cost of buying (or of selling) one
dollar of stock: half the effective bid-ask spread plus commissions and impact.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


def monthly_cost(turnover: float, one_way_cost: float, sides: int = 2) -> float:
    """Trading cost per month for a portfolio with `sides` sides of one dollar each.

    One-sided turnover T means a fraction T of each side is sold and replaced
    each month, so each side trades 2T dollars (T sold, T bought) and pays the
    one-way cost on each. A long-short portfolio (sides=2) pays 4 T c per
    dollar long.
    """
    return 2 * sides * turnover * one_way_cost


def break_even_cost(gross_mean: float, turnover: float, sides: int = 2, other_costs: float = 0.0) -> float:
    """One-way cost at which the average net return is zero."""
    if turnover <= 0:
        return float("inf")
    return max(gross_mean - other_costs, 0.0) / (2 * sides * turnover)


def net_returns(gross: np.ndarray, turnover, one_way_cost: float, borrow_fee_annual: float = 0.0,
                sides: int = 2) -> np.ndarray:
    """Monthly net returns: gross minus trading costs minus the borrow fee on the short side."""
    g = np.asarray(gross, dtype=float)
    tc = 2 * sides * np.asarray(turnover, dtype=float) * one_way_cost
    return g - tc - borrow_fee_annual / 12 * (sides == 2)


# ---------------------------------------------------------------------------
# A simulated stock universe
# ---------------------------------------------------------------------------


def simulate_universe(n_stocks: int = 1000, n_months: int = 480, seed: int = 0,
                      premia: dict | None = None, idio_vol: float = 0.10, mkt_vol: float = 0.045,
                      book_growth_vol: float = 0.15) -> dict:
    """Monthly returns, market caps and book values for a simulated cross-section. FOR TEACHING ONLY.

    Expected returns depend on three standardized signals, so the truth is
    known: book-to-market (value), the return from t-12 to t-2 (momentum) and
    the return in month t-1 (short-term reversal). `premia` gives each
    signal's monthly expected-return slope per standard deviation, e.g.
    {"value": 0.002, "momentum": 0.004, "reversal": -0.003}.
    """
    rng = np.random.default_rng(seed)
    premia = premia or {"value": 0.0015, "momentum": 0.004, "reversal": -0.003}
    N, T = n_stocks, n_months
    logme = rng.normal(6.0, 1.6, N)  # log market cap ($ millions), lognormal cross-section
    logbm = rng.normal(-0.5, 0.8, N)
    logbe = logme + logbm
    R = np.zeros((T, N))
    ME = np.zeros((T, N))
    BE = np.zeros((T, N))
    MU = np.zeros((T, N))
    for t in range(T):
        z_val = _z(logbe - logme)
        if t >= 13:
            cum = np.log1p(R[t - 12:t - 1]).sum(axis=0)
            z_mom = _z(cum)
        else:
            z_mom = np.zeros(N)
        z_rev = _z(R[t - 1]) if t >= 1 else np.zeros(N)
        mu = 0.005 + premia["value"] * z_val + premia["momentum"] * z_mom + premia["reversal"] * z_rev
        MU[t] = mu
        r = mu + mkt_vol * rng.standard_normal() + idio_vol * rng.standard_t(5, N) / math.sqrt(5 / 3)
        r = np.maximum(r, -0.95)
        R[t] = r
        logme = logme + np.log1p(r)
        # Book equity grows slowly and noisily, independent of price, updated monthly.
        logbe = logbe + 0.005 + book_growth_vol / math.sqrt(12) * rng.standard_normal(N)
        ME[t] = np.exp(logme)
        BE[t] = np.exp(logbe)
    return {"returns": R, "me": ME, "be": BE, "mu": MU, "premia": premia}


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def signals(u: dict) -> dict:
    """Signals known at the end of each month t, for portfolios held in month t+1.

    value     book-to-market, updated once a year at the end of June using book
              equity from six months earlier and market equity from December
              (the Fama-French convention), held for twelve months
    momentum  cumulative return from t-11 to t-1 (skipping month t), i.e. the
              return over months -12 to -2 relative to the holding month
    reversal  minus the return in month t (the month before the holding month),
              so that, as for the other signals, high values are bought
    """
    R, ME, BE = u["returns"], u["me"], u["be"]
    T, N = R.shape
    val = np.full((T, N), np.nan)
    current = None
    for t in range(T):
        if t >= 12 and t % 12 == 5:  # month index 5 = June if the simulation starts in January
            current = np.log(BE[t - 6] / ME[t - 6])
        if current is not None:
            val[t] = current
    mom = np.full((T, N), np.nan)
    lr = np.log1p(R)
    for t in range(11, T):
        mom[t] = lr[t - 11:t].sum(axis=0)  # months t-11..t-1, skipping t
    rev = -R.copy()
    return {"value": val, "momentum": mom, "reversal": rev}


def sort_weights(signal: np.ndarray, me: np.ndarray, frac: float = 0.3, hold_frac: float | None = None,
                 prev_long: np.ndarray | None = None, prev_short: np.ndarray | None = None):
    """Value-weighted long (top `frac`) and short (bottom `frac`) membership.

    With hold_frac > frac, a buy/hold spread: stocks already held stay in the
    long side while still in the top `hold_frac`, and new stocks enter only
    when in the top `frac` (and likewise for the short side).
    """
    ok = np.isfinite(signal)
    rank = np.full(signal.shape, np.nan)
    order = np.argsort(signal[ok])
    pct = np.empty(ok.sum())
    pct[order] = (np.arange(ok.sum()) + 0.5) / ok.sum()
    rank[ok] = pct
    top = np.nan_to_num(rank) >= 1 - frac
    bot = (np.nan_to_num(rank, nan=1.0) <= frac) & ok
    if hold_frac is not None and prev_long is not None:
        top = top | (prev_long & (np.nan_to_num(rank) >= 1 - hold_frac))
        bot = bot | (prev_short & (np.nan_to_num(rank, nan=1.0) <= hold_frac) & ok)
    return top, bot


def _vw(mask: np.ndarray, me: np.ndarray) -> np.ndarray:
    w = np.where(mask, me, 0.0)
    s = w.sum()
    return w / s if s > 0 else w


def backtest_sort(u: dict, name: str, frac: float = 0.3, hold_frac: float | None = None, start: int = 24) -> dict:
    """Monthly long-short returns and one-sided turnover of a value-weighted 30/70 sort.

    "expected" is the true expected long-short return each month, computed
    from the simulation's expected returns: free of the sampling noise in
    realized returns, so cost-mitigation rules can be compared exactly.

    Turnover for each side is half the sum of absolute differences between
    the new target weights and the drifted weights of the previous month (so
    price drift in a value-weighted portfolio needs no trading). The reported
    turnover is the average of the long and short sides, as in Novy-Marx and
    Velikov (2016).
    """
    R, ME = u["returns"], u["me"]
    sig = signals(u)[name]
    T, N = R.shape
    gross, turn, n_long, expected = [], [], [], []
    MU = u.get("mu")
    wl_prev = ws_prev = None
    long_prev = short_prev = None
    for t in range(start, T - 1):
        if not np.isfinite(sig[t]).any():
            continue
        top, bot = sort_weights(sig[t], ME[t], frac, hold_frac, long_prev, short_prev)
        wl, ws = _vw(top, ME[t]), _vw(bot, ME[t])
        if wl_prev is not None:
            tl = 0.5 * np.abs(wl - wl_prev).sum()
            ts = 0.5 * np.abs(ws - ws_prev).sum()
            turn.append(0.5 * (tl + ts))
        else:
            turn.append(np.nan)
        r = R[t + 1]
        gross.append(wl @ r - ws @ r)
        if MU is not None:
            expected.append(wl @ MU[t + 1] - ws @ MU[t + 1])  # the true expected spread (known in simulation)
        n_long.append(int(top.sum()))
        # Drift the weights through month t+1 for next month's turnover.
        wl_prev = wl * (1 + r) / (wl @ (1 + r))
        ws_prev = ws * (1 + r) / (ws @ (1 + r))
        long_prev, short_prev = top, bot
    return {"gross": np.array(gross), "turnover": np.array(turn), "n_long": np.array(n_long),
            "expected": np.array(expected)}


# ---------------------------------------------------------------------------
# Market impact and capacity
# ---------------------------------------------------------------------------


def sqrt_impact_cost(trade_value: np.ndarray, adv: np.ndarray, daily_vol: np.ndarray, y: float = 1.0,
                     days: int = 1) -> np.ndarray:
    """Square-root impact per dollar traded: y * sigma_daily * sqrt(trade / (ADV * days)).

    trade_value  dollars traded in each stock (spread over `days` trading days)
    adv          average daily dollar volume of each stock
    daily_vol    daily return volatility of each stock
    """
    q = np.asarray(trade_value, dtype=float) / (np.asarray(adv, dtype=float) * days)
    return y * np.asarray(daily_vol, dtype=float) * np.sqrt(np.maximum(q, 0.0))


def capacity_curve(aum_grid, gross_mean: float, turnover: float, weights: np.ndarray, adv: np.ndarray,
                   daily_vol: np.ndarray, half_spread: np.ndarray, y: float = 1.0, days: int = 5) -> dict:
    """Net monthly return of a long-short strategy as assets under management grow.

    Each month the strategy trades 2 * turnover of each side. Trades are
    allocated across stocks in proportion to `weights` (each side sums to 1),
    spread over `days` trading days, and pay the half-spread plus square-root
    impact. Returns the cost and net return at each AUM, and the AUM at which
    the net return halves and reaches zero (NaN if not reached on the grid).
    """
    aum = np.asarray(aum_grid, dtype=float)
    w = np.asarray(weights, dtype=float)
    cost = np.empty(len(aum))
    for i, A in enumerate(aum):
        trade = 2 * turnover * A * w  # dollars traded per stock per month, per side
        per_dollar = np.asarray(half_spread) + sqrt_impact_cost(trade, adv, daily_vol, y, days)
        side_cost = (trade * per_dollar).sum() / A  # cost as a fraction of one side's capital
        cost[i] = 2 * side_cost  # long and short sides
    net = gross_mean - cost

    def crossing(level):
        below = np.flatnonzero(net <= level)
        if not len(below):
            return float("nan")
        k = below[0]
        if k == 0:
            return float(aum[0])
        x0, x1, y0, y1 = math.log(aum[k - 1]), math.log(aum[k]), net[k - 1], net[k]
        return float(math.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))

    return {"aum": aum, "cost": cost, "net": net, "half_net_aum": crossing((gross_mean - cost[0]) / 2),
            "zero_net_aum": crossing(0.0)}
