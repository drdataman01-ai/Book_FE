# Lab 7: Forecasting Volatility

Companion lab for **Chapter 7** of *Financial Engineering in Practice*.

**What you will explore:** how predictable volatility is, which models forecast it best, and whether the forecasts help an investor.

## Quick start

```bash
pip install -r requirements.txt
python lab07.py                  # downloads daily factors (French) and the VIX (FRED)
```

If the downloads are blocked, fetch the files by hand and pass them in:

- `F-F_Research_Data_Factors_daily_CSV.zip` from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (Fama/French 3 Factors [Daily]);
- `VIXCLS.csv` from [FRED](https://fred.stlouisfed.org/series/VIXCLS) (Download → CSV).

```bash
python lab07.py --ff3-daily-file data/F-F_Research_Data_Factors_daily_CSV.zip --vix-file data/VIXCLS.csv
```

Other options: `--no-vix`, `--start 1990-01-01` (first forecast date), `--window-years 20` (estimation window), `--weight-cap 1.5` (Part D leverage cap), `--cost 0.001` (Part D cost per unit change in weight).

Or open `lab07_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab07.py --source sim
```

This simulates GJR-GARCH returns with fat-tailed shocks and a VIX-like series. Results on simulated data test the code only.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Stylized facts | Do returns and volatility have memory? | Autocorrelations of daily returns and absolute returns, lags 1–50; correlation of today's return with tomorrow's absolute return (leverage effect) | Lab Figure 7.1 |
| B. Forecasts | Which forecasts beat a random walk in volatility? | Market variance over the next 1, 5 and 22 days; random walk (last 22 days), long-run average, EWMA (λ = 0.94), GARCH(1,1), GJR-GARCH, HAR and the VIX; walk-forward from 1990, re-estimated yearly on the previous 20 years; QLIKE and MSE losses, Diebold–Mariano tests, Mincer–Zarnowitz regressions | Lab Figures 7.2–7.3 |
| C. Crises | Do the forecasts hold up when volatility jumps? | The same losses in 2008–09, 2020 and 2022 | Lab Figure 7.4 |
| D. Use | Can an investor profit from the forecasts? | Market exposure ∝ 1 / last month's realized variance, scaled to the market's volatility (Moreira and Muir 2017); Sharpe ratio, drawdown, alpha, costs and an optional leverage cap | Lab Figure 7.5 |

### Design notes

- **Returns** are the market's total daily return (Mkt-RF + RF), demeaned over the full sample before squaring; the mean is tiny relative to daily volatility.
- **The proxy for realized variance** is the sum of squared daily returns. It is noisy; intraday (five-minute) realized variance would be better but is not freely available back to 1926. QLIKE ranks forecasts correctly even with a noisy proxy (Patton 2011); MSE is dominated by a few extreme days.
- **The random walk** uses the last 22 days' variance, scaled to the horizon. A single day's squared return is too noisy a benchmark.
- **HAR** is estimated by OLS on squared daily returns, with forecasts floored at 10% of the window's average variance because a linear model can predict near-zero variance after calm days.
- **The VIX** is the market's 30-day implied volatility for the S&P 500 under the risk-neutral measure, so it includes a variance risk premium; expect it to overpredict on average (Mincer–Zarnowitz slope below 1).
- **Part D** scales weights using the full-sample volatility of the market, as Moreira and Muir do; this uses information not available in real time (Cederburg et al. 2020). Exercise 4 removes it.

## Code testing status

| Part | Status |
|---|---|
| EWMA and GARCH recursions | Tested against explicit loops |
| GARCH and GJR estimation | Tested: recovers known parameters from simulated data |
| Multi-step forecasts | Tested against iterating the recursion |
| HAR features, future sums, QLIKE, Mincer–Zarnowitz | Tested |
| FRED reader | Tested on a file in FRED's layout (including "." for missing values) |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab07.py`.

## Exercises

1. Replace normal innovations with Student-t in GARCH (`fit_garch(..., dist="t")`). Do forecasts improve?
2. Estimate HAR in logs (log variance on log daily, weekly and monthly averages). Does it beat the linear HAR under QLIKE?
3. Combine GARCH and the VIX in one Mincer–Zarnowitz regression. Does each add information?
4. In Part D, scale the weights using only past data (an expanding-window estimate of c). How much of the alpha survives?
5. Run Part B from 1930 to 1990 without the VIX. Are the rankings the same?

## Data citation

Kenneth R. French Data Library, Fama/French 3 Factors [Daily]. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/VIXCLS
