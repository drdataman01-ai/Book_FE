# Lab 10: Does Diversification Fail When You Need It Most?

Companion lab for **Chapter 10** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** correlations rise in crises, so diversification fails exactly when investors need it most.

## Quick start

```bash
pip install -r requirements.txt
python lab10.py                  # downloads three French library files and three FRED series
```

The lab takes under a minute. If downloads are blocked, save the files in `data/` and pass them in:

| Data | Source | Option |
|---|---|---|
| US market excess return and bill rate (from 1926) | French library: `F-F_Research_Data_Factors_CSV.zip` | `--ff3-file` |
| International (developed ex US) market excess return, in dollars (from 1990) | French library: `Developed_ex_US_3_Factors_CSV.zip` | `--intl-file` |
| 10 Industry Portfolios (from 1926) | French library: `10_Industry_Portfolios_CSV.zip` | `--ind10-file` |
| 10-Year Treasury Constant Maturity Rate (from 1953) | FRED [GS10](https://fred.stlouisfed.org/series/GS10) | `--gs10-file` |
| Consumer Price Index, all urban consumers (from 1947) | FRED [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | `--cpi-file` |
| Spot crude oil price, WTI (from 1946) | FRED [WTISPLC](https://fred.stlouisfed.org/series/WTISPLC) | `--oil-file` |

Any file except the first can be missing; the lab reports what it could not load and skips those parts.

Or open `lab10_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab10.py --source sim
```

The simulation switches between calm and stress regimes. In stress, volatility nearly triples and the true correlation between US and international stocks rises from 0.60 to 0.75. The stock–bond correlation is tied to inflation. Results on simulated data test the code only.

## What the lab does

All returns are monthly and in excess of the one-month Treasury bill rate. Bond returns are built from GS10: each month the lab buys a 10-year par bond at the previous month-end yield and sells it a month later at the new yield, keeping the coupon.

| Part | Question | Method | Output |
|---|---|---|---|
| A. Conditional correlations | Do correlations rise in turbulent markets? | Correlation of US stocks with international stocks, bonds and oil, and the average pairwise correlation of 10 industries: calm months, high-volatility months (top 20% by trailing 12-month US volatility), Forbes–Rigobon adjusted, and the worst 10% of US months; exceedance correlations against a bivariate normal with the same overall correlation | Lab Figures 10.1–10.2 |
| B. Tail dependence | Do assets crash together more than a normal distribution implies? | P(other asset in worst p-tail given US stocks in worst p-tail), p = 5%, 10%, 20% | Table |
| C. Stocks and bonds | Is the negative stock–bond correlation reliable? | Rolling 36-month correlation; by decade; by inflation regime; 2022 | Lab Figure 10.3 |
| D. Investor's view | What did diversification deliver in bad times? | 100% stocks, 60/40 and an equal-weight mix of US stocks, international stocks, bonds and oil: returns in the worst and best 5% of stock months, drawdowns, crisis periods, diversification ratio | Lab Figures 10.4–10.5 |

### Design notes

- **The volatility bias.** A correlation measured in high-volatility months is biased upward even if the underlying relationship has not changed (Boyer, Gibson and Loretan 1999; Forbes and Rigobon 2002). The correction assumes the extra volatility comes from the conditioning variable alone. If the true correlation does rise in crises, the correction removes part of a real increase, so treat the raw and adjusted numbers as bounds.
- **Conditioning on one tail** (the worst 10% of US months) lowers measured correlation for the same reason: it truncates the range of US returns. Exceedance correlations, which condition on both assets, are compared with their normal benchmark for this reason.
- **Bond returns** from constant-maturity yields ignore the difference between the yield on a par bond and on bonds actually traded, and use month-end rather than total-return data. They are close to, but not the same as, index returns.
- **Oil** is a single commodity and a volatile one; it stands in for commodities because long free commodity index series are hard to find.

## Verification status

| Part | Status |
|---|---|
| Bond returns from yields | Tested against hand calculation (unchanged yield, 1-point rise) |
| Exceedance correlations and normal benchmark | Tested: symmetric, below the overall correlation, falling into the tails |
| Forbes–Rigobon correction | Tested: recovers a known constant correlation from a high-volatility subsample |
| Tail dependence | Tested on independent and identical series |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab10.py`.

## Exercises

1. Change the high-volatility definition to the top 10% of months, or to NBER recessions. Does the conclusion change?
2. Use daily data from the French library (`F-F_Research_Data_Factors_daily_CSV.zip` and the daily industry file) to compute the average industry correlation in each month. How high did it go in October 1987, October 2008 and March 2020?
3. Replace the 10-year bond with a 2-year bond (FRED GS2, from 1976). Is the shorter bond a better hedge in 2022?
4. Find the months since 1953 in which both stocks and bonds fell by more than 2%. When did they cluster?
5. Add gold, if you can find a free monthly series. How did it behave in the worst stock months?

## Data citation

Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (Fama/French 3 Factors; Developed ex US 3 Factors; 10 Industry Portfolios).
Board of Governors of the Federal Reserve System, 10-Year Treasury Constant Maturity Rate [GS10]; U.S. Bureau of Labor Statistics, Consumer Price Index for All Urban Consumers [CPIAUCSL]; Federal Reserve Bank of St. Louis, Spot Crude Oil Price: West Texas Intermediate [WTISPLC]; all retrieved from FRED.
