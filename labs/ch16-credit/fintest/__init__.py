"""fintest: the analysis module for *Financial Engineering and the Evidence:
Active Learning by Verifying Famous Claims*.

This copy contains the parts this lab uses:

* fintest.data   - download and parse data from the Kenneth R. French Data Library
* fintest.stats  - returns, Sharpe ratios, drawdowns, Newey-West regressions, multiple-testing helpers
* fintest.inference - stylized facts, robust and bootstrap errors, GRS, Fama-MacBeth, Hodrick
* fintest.multiple - multiple-testing corrections, deflated Sharpe ratio, PBO by CSCV
* fintest.predictors - the Goyal-Welch equity-premium predictors
* fintest.forecast - walk-forward forecasts, out-of-sample R2, Clark-West, scoring rules
* fintest.costs - trading costs, turnover, a simulated stock universe, impact and capacity
* fintest.shiller - Robert Shiller's long-run market data (CAPE)
* fintest.anomalies - Open Source Asset Pricing readers, decay regressions, shrinkage
* fintest.vol    - EWMA, GARCH and GJR models, HAR, volatility-forecast losses, a FRED reader
* fintest.risk   - VaR and expected shortfall, Kupiec and Christoffersen tests, Basel traffic light
* fintest.options - Black-Scholes, option-writing returns, variance-swap payoffs, MPPM, Cboe reader, delta hedging
* fintest.rates  - Fed zero-coupon curve, forwards, Fama-Bliss, Campbell-Shiller, Cochrane-Piazzesi
* fintest.credit - expected-loss spreads, break-even default rates, FRED and excess-bond-premium readers
* fintest.plots  - the book's chart style

Later chapters add backtesting with costs.
"""

__version__ = "0.17.0"
