# Lab 4: Momentum and Its Crashes

Companion lab for **Chapter 4** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** past winners keep outperforming past losers.

## Quick start

```bash
pip install -r requirements.txt
python lab04.py
```

The first run downloads five files from the Kenneth R. French Data Library and caches them in `data/`. Open `report/lab04_report.md`, or work through `lab04_walkthrough.ipynb`.

### If a download fails

Download the CSV versions from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and pass them with the options below. All five names were checked against the library's archived listing.

| Library page title | File | Option |
|---|---|---|
| Fama/French 3 Factors | `F-F_Research_Data_Factors_CSV.zip` | `--ff3-file` |
| Momentum Factor (Mom) | `F-F_Momentum_Factor_CSV.zip` | `--mom-file` |
| Momentum Factor (Mom) [Daily] | `F-F_Momentum_Factor_daily_CSV.zip` | `--mom-daily-file` |
| 10 Portfolios Formed on Prior (12-2) Return | `10_Portfolios_Prior_12_2_CSV.zip` | `--deciles-file` |
| 10 Industry Portfolios | `10_Industry_Portfolios_CSV.zip` | `--industries-file` |

**Optional, many countries:** a Jensen–Kelly–Pedersen CSV (https://jkpfactors.com) with columns `location`, `name`, `date`, `ret`, passed with `--jkp-file`; the lab keeps the `ret_12_1` momentum factor.

### Testing the code without real data

```bash
python lab04.py --source sim
```

Simulated results are not evidence about markets; the report and figures are labeled.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Replication | Do winners keep winning? | Mom from 1927 by decade, before, during and after the Jegadeesh–Titman (1965–1989) sample; average returns of the ten momentum deciles | Lab Figures 4.1–4.2 |
| B. Crashes | When does momentum lose? | Worst months; regression of Mom on the market with bear-market and bear-market-rebound interactions (Daniel and Moskowitz 2016), Newey–West t | Lab Figure 4.3 |
| C. Volatility scaling | Can the crashes be managed? | Weight = 12% / annualized volatility of daily Mom returns over the prior 126 trading days (Barroso and Santa-Clara 2015), compared with raw Mom at the same average volatility | Lab Figure 4.4 |
| D. Time-series momentum | Does trend following work? | Long or short each of 10 industry portfolios and the market by the sign of its past 12-month excess return, each scaled to 40% volatility (Moskowitz, Ooi and Pedersen 2012, equity-only) | Lab Figure 4.5 |
| E. Costs | Does it survive trading? | Turnover 34.5% a month (Novy-Marx and Velikov 2016), cost = 4 × turnover × one-way cost; the scaled strategy also pays for weight changes | Table |

### Design notes

- **Bear market** = cumulative market return over the previous 24 months below zero, known at the start of the month (Daniel and Moskowitz 2016).
- **Volatility forecast** uses daily data up to the end of the previous month only; a test checks that changing later data leaves earlier weights unchanged. Weights are uncapped by default (`--weight-cap` to limit leverage).
- **Time-series momentum** here uses only U.S. equity portfolios, which are highly correlated; Moskowitz, Ooi and Pedersen's 58 futures span four asset classes. Treat Part D as an illustration of the method, not a replication.

## Verification status

| Part | Status |
|---|---|
| Daily and decile file parsers | Tested on files in the library's layouts; **confirm on the live files at first run** |
| Volatility scaling | Tested for look-ahead and for weight = target / forecast volatility |
| Crash regression | Tested: recovers a known negative beta in bear-market rebounds in simulated data |
| Time-series momentum | Tested: positions use only earlier data |
| Full lab and notebook | Run end to end on simulated data |
| Live downloads | Not tested from the build environment (network blocked) |

Run the tests with `python tests/test_lab04.py`.

## Exercises

1. Re-run Part C with `--weight-cap 1.5`. How much of the improvement survives a leverage limit?
2. Replace the 24-month bear indicator with a 12-month one. Does the crash pattern hold?
3. Compute momentum's net Sharpe ratio after 2001 with a cost of 10 basis points. Is momentum more implementable today?
4. Using the JKP data, compare momentum's t-statistics across countries. Where does it fail, and is Japan an exception?
5. Implement Daniel and Moskowitz's dynamic strategy, which scales by forecast mean as well as variance, and compare it with Part C out of sample.

## Data citation

Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.
