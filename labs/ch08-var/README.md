# Lab 8: Value at Risk and Its Backtests

Companion lab for **Chapter 8** of *Financial Engineering in Practice*.

**What you will explore:** how often losses exceed a 99% VaR, in calm markets and in crises, for six common models.

## Quick start

```bash
pip install -r requirements.txt
python lab08.py                  # downloads daily factors from the French library
```

If the download is blocked, save `F-F_Research_Data_Factors_daily_CSV.zip` (Fama/French 3 Factors [Daily]) from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) in `data/` and run:

```bash
python lab08.py --ff3-daily-file data/F-F_Research_Data_Factors_daily_CSV.zip
```

Other options: `--start 1930-01-01` (first backtest day), `--fit-years 10` (GARCH estimation window), `--hs-window 250` (historical-simulation window). A full run takes about half a minute.

Or open `lab08_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab08.py --source sim
```

The simulation draws GJR-GARCH returns with Student-t shocks, so the GJR-t model is the true model and should pass its backtests. Results on simulated data test the code only.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Coverage | Is the 99% VaR breached on 1% of days? | Six models, each forecasting the next day's VaR from data up to today: normal with 250-day volatility; historical simulation (250 days); EWMA, λ = 0.94 (RiskMetrics); GARCH(1,1) with normal shocks; GJR-GARCH with Student-t shocks; filtered historical simulation (GJR volatility × empirical quantile of the last 1,000 standardized returns). GARCH models re-estimated yearly on the previous 10 years. Kupiec test; Basel zone for each calendar year | Lab Figures 8.1–8.2 |
| B. Clustering | Are breaches independent? | Christoffersen independence and conditional-coverage tests; probability of a breach after a breach; breaches per year | Lab Figure 8.3 |
| C. Stress | Where do the models fail? | Breaches in 1930–32, October 1987, 2008–09 and 2020; actual/expected breaches at 95%, 97.5%, 99%, 99.5% and 99.9%; historical-simulation windows of 250, 500 and 1,000 days | Lab Figure 8.4 |
| D. Expected shortfall | Are the bad days as bad as predicted? | ES at 97.5% (the Basel trading-book measure): ratio of average loss on breach days to average predicted ES; t-test on exceedance residuals scaled by volatility (McNeil and Frey 2000) | Lab Figure 8.5 |

### Design notes

- **Timing.** The forecast made at the end of day t is compared with the loss on day t+1. Nothing from day t+1 enters the forecast.
- **Portfolio.** The market return is Mkt-RF + RF, a long position in U.S. stocks. VaR is the loss on a one-dollar position, with mean daily return set to zero.
- **Basel zones** apply to 99% VaR over 250 days: green 0–4 breaches, yellow 5–9, red 10 or more. The lab classifies each calendar year, scaled to 250 days.
- **Tests have low power** at deep quantiles. At 99.9% only about 24 breaches are expected over the whole sample; one bad week moves the ratio a lot.

## Code testing status

| Part | Status |
|---|---|
| Normal and Student-t VaR and ES | Tested against known values and Monte Carlo |
| Historical simulation | Tested against direct calculation |
| Breach timing | Tested: forecast for day t+1 compared with day t+1's loss |
| Kupiec test | Tested against hand calculation; false-rejection rate checked by simulation |
| Christoffersen test | Tested: detects clustered breaches, accepts independent ones |
| Basel zones, ES backtest | Tested (ES backtest: unbiased model accepted, thin-tailed model rejected) |
| Full lab and notebook | Run end to end on simulated data (the true model passes; the normal model fails) |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab08.py`.

## Exercises

1. Replace the market with a 60/40 stock–bond portfolio (use FRED's 10-year yield to approximate bond returns). Which models fail now?
2. Add a 10-day VaR by the square-root-of-time rule and by simulating the GJR model ten days ahead. How different are they in crises?
3. Use a "stressed" historical simulation that always includes 2008–09 in its window, as Basel 2.5 required. How conservative is it in calm years?
4. Implement the Acerbi–Székely Z2 test of ES and compare it with the McNeil–Frey test.
5. Run the backtest separately for 1930–1979 and 1980 onward. Has any model's performance changed?

## Data citation

Kenneth R. French Data Library, Fama/French 3 Factors [Daily]. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
