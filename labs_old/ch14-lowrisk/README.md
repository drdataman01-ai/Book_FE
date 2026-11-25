# Lab 14: The Low-Risk Anomaly

Companion lab for **Chapter 14** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** low-beta and low-volatility stocks earn higher risk-adjusted returns than the CAPM predicts: the security market line is too flat.

## Quick start

```bash
pip install -r requirements.txt
python lab14.py                  # downloads from the French library and FRED
```

If the downloads are blocked, save these files in `data/` and pass them in:

| File | Source | Option |
|---|---|---|
| `F-F_Research_Data_Factors_CSV.zip` | [French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): Fama/French 3 Factors | `--ff3-file` |
| `F-F_Momentum_Factor_CSV.zip` | French library: Momentum Factor (Mom) | `--mom-file` |
| `Portfolios_Formed_on_BETA_CSV.zip` | French library: Portfolios Formed on Market Beta | `--beta-file` |
| `Portfolios_Formed_on_VAR_CSV.zip` | French library: Portfolios Formed on Variance | `--var-file` |
| `Portfolios_Formed_on_RESVAR_CSV.zip` | French library: Portfolios Formed on Residual Variance | `--resvar-file` |
| `GS10.csv` | [FRED GS10](https://fred.stlouisfed.org/series/GS10), 10-year Treasury yield, monthly | `--gs10-file` (or `--no-gs10`) |
| Betting Against Beta: Equity Factors, Monthly (optional) | [AQR data sets](https://www.aqr.com/Insights/Datasets) | `--aqr-file` |

The AQR file can be the original xlsx (sheet "BAB Factors"; needs `openpyxl`) or a CSV saved from that sheet. The lab finds the header row that starts with `DATE` and uses the `USA` column.

Or open `lab14_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab14.py --source sim
```

The simulation builds in a flat security market line and low-risk portfolios that fall when yields rise. Results on simulated data test the code only.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Security market line | Do high-beta stocks earn proportionally more? | Ten portfolios sorted on beta, total volatility and residual volatility, value- and equal-weighted: average excess return, realized beta, CAPM and four-factor alphas (market, size, value, momentum); slope of average return on beta | Lab Figure 14.1 |
| B. Long-short | Is there a tradable premium? | Low minus high (three lowest minus three highest deciles); BAB-style: each leg scaled by 1 / its ex-ante beta (rolling 60 months), so the portfolio is beta-neutral (Frazzini and Pedersen 2014); optional comparison with AQR's stock-level BAB factor | Lab Figure 14.2 |
| C. Stress tests | Is it robust? | Equal vs value weights; 1963–1990, 1991–2009, 2010 on; regression on the market and changes in the 10-year yield; calendar-year returns in 2008, 2020 and 2022; rolling ten-year CAPM alpha | Lab Figures 14.3–14.4 |
| D. Investor's view | Would a benchmarked investor hold it? | Lowest-volatility and lowest-beta quintiles against the market: Sharpe ratio, beta, alpha, drawdown, tracking error, information ratio, share of years behind the market | Lab Figure 14.5 |

### Design notes

- **Portfolios, not stocks.** The French portfolios are formed on beta estimated from past returns (and on variance and residual variance of past daily returns). The lab's BAB-style portfolio is built from these portfolios, not from individual stocks with rank weights as in Frazzini and Pedersen, so it is a simpler and more transparent version. Expect smaller and less significant results than for AQR's factor, which also tilts toward small stocks (Novy-Marx and Velikov 2022).
- **Ex-ante betas** use the previous 60 months (at least 36) of each leg's returns, without shrinkage: portfolio betas are estimated precisely, and shrinking them toward 1 leaves the portfolio with a negative realized beta.
- **Returns** in the French files are total returns; the lab subtracts the risk-free rate. Sorted portfolios start in July 1963.
- **The interest-rate test** uses the monthly change in the 10-year yield, in decimals. A coefficient of −5 means a 1-percentage-point rise in yields costs 5% that month, like a bond with duration 5.
- **Column layout.** The lab looks for the columns `Lo 10` … `Hi 10` (deciles) and `Lo 20` … `Hi 20` (quintiles) and for tables titled with "Value" and "Equal". **Confirm at first run** that the live files use these names; the parser is tested on a file in the documented layout.

## Verification status

| Part | Status |
|---|---|
| French-layout parsing (deciles, quintiles, value and equal weights) | Tested on a file in the library's layout; **confirm on the live files** |
| AQR reader | Tested on a CSV in AQR's layout; the xlsx path is untested |
| Ex-ante betas | Tested against direct calculation |
| BAB-style construction | Tested: beta-neutral, recovers a known alpha |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab14.py`.

## Exercises

1. Replace the four-factor model with the Fama–French five-factor model (download the 5-factor file). How much of the low-risk alpha is explained by profitability and investment?
2. Use only the lowest and highest deciles instead of three on each side. Are the results noisier?
3. Shrink the ex-ante betas toward 1 with weight 0.6, as Frazzini and Pedersen do for single stocks. What happens to the realized beta and alpha?
4. Add the Treasury return (or the yield change) as a factor and recompute the alphas in each subperiod.
5. Compare the lab's BAB-style returns with AQR's BAB factor. How correlated are they, and which months differ most?

## Data citation

Kenneth R. French Data Library. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Board of Governors of the Federal Reserve System, Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity [GS10], retrieved from FRED, Federal Reserve Bank of St. Louis.
AQR Capital Management, Betting Against Beta: Equity Factors, Monthly (optional). Frazzini, A., and Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1–25.
