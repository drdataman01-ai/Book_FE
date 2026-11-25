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
* fintest.fx     - FRED exchange and interest rates, currency excess returns, Fama regressions, carry portfolios
* fintest.portfolio - allocation rules, Ledoit-Wolf shrinkage, HRP, walk-forward backtests, Sharpe-difference tests
* fintest.corr   - bond returns from yields, exceedance correlations, Forbes-Rigobon correction, tail dependence
* fintest.ml     - characteristic panels, ML model horse race, walk-forward training, long-short tests, random features
* fintest.tsfm   - classical one-step forecasters, foundation-model adapters, rolling-origin evaluation, contamination checks
* fintest.design - power analysis, Bayesianized p-values, specification curves, forking paths, pre-registration
* fintest.replicate - data fingerprints, append-only run log, comparison with original results
* fintest.plots  - the book's chart style

"""

__version__ = "1.0.0"
