# Lab 20: Does Machine Learning Find Hidden Alpha?

Companion lab for **Chapter 20** of *Financial Engineering in Practice*.

**What you will explore:** whether machine learning finds return predictability that linear models miss, and whether it survives costs.

## Quick start

```bash
pip install -r requirements.txt          # needs scikit-learn
python lab20.py                          # downloads three French library files; about 10 minutes
python lab20.py --fast                   # one seed, refit every 10 years; about 3 minutes
python lab20.py --gw-file data/PredictorData2024.xlsx    # adds Part D (complexity)
```

If downloads are blocked, save the zip files from the [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) in `data/` and pass them in:

| File | Library name | Option |
|---|---|---|
| `F-F_Research_Data_Factors_CSV.zip` | Fama/French 3 Factors | `--ff3-file` |
| `100_Portfolios_10x10_CSV.zip` | 100 Portfolios Formed on Size and Book-to-Market (10 x 10) | `--sbm100-file` |
| `49_Industry_Portfolios_CSV.zip` | 49 Industry Portfolios | `--ind49-file` |
| Goyal–Welch predictor spreadsheet (optional, Part D) | [Amit Goyal's website](https://sites.google.com/view/agoyal145), "Updated data" for Goyal and Welch (2008) | `--gw-file` |

Or open `lab20_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab20.py --source sim --fast
```

The simulation builds 150 portfolios whose expected returns depend linearly on value and momentum, and nonlinearly on an interaction of size with momentum and a volatility threshold. Results on simulated data test the code only.

## What the lab does

**Why portfolios, not stocks.** Gu, Kelly and Xiu (2020) used individual stocks from CRSP, which is not free. This lab uses 149 value-weighted portfolios from the French library: 100 sorted on size and book-to-market and 49 industries. Portfolios average away much of the noise in individual stocks and leave out microcaps, so the test is closer to what a large investor could trade, but it is a different test. Expect higher R² than for stocks and smaller long–short spreads.

**Characteristics** (all known at the end of month t, ranked each month across portfolios and scaled to [−1, 1]): log average market cap, log number of firms, log book-to-market, last month's return, momentum (t−11 to t−1), long-term reversal (t−35 to t−12), 12-month volatility, 60-month beta, 60-month idiosyncratic volatility, maximum monthly return over 12 months; plus two market variables (the market's past 12-month return and volatility). If a file lacks the size, firm-count or book-to-market table, the lab says so and sets that characteristic to the median.

**Target.** Next month's excess return minus that month's average across portfolios, so the models compete on ranking portfolios rather than timing the market. R² is measured against a forecast of zero, which here means "every portfolio earns the average".

**Training.** Expanding window. At each refit date (every 5 years from 1975; every 10 with `--fast`), each model's tuning parameter is chosen on the previous 10 years after training on everything earlier; the model is then refit on all data to the refit date. A row dated t has its target in t+1, so only rows dated before the refit month minus one enter training.

| Model | Tuned on validation data |
|---|---|
| OLS-3: size, book-to-market, momentum | — |
| OLS: all 12 inputs | — |
| Elastic net (inputs standardized) | penalty |
| Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`) | depth 1 or 2, 50 or 150 trees |
| Neural network, 32-16-8 (scikit-learn `MLPRegressor`), averaged over 3 seeds | weight penalty (0.1, 1 or 10, on standardized inputs and target) |

| Part | Question | Method | Output |
|---|---|---|---|
| A. Horse race | Do flexible models forecast better? | Out-of-sample R², Diebold–Mariano t against OLS-3, long–short top-minus-bottom decile | Lab Figures 20.1–19.2 |
| B. Robustness | Does it survive where it matters? | Large portfolios only (size deciles 6–10 and industries); 1975–1999 versus 2000 on; costs of 10, 25, 50 bp per unit of turnover | Lab Figure 20.3 |
| C. Stability and interpretation | Is it the seed? What does it use? | Neural network with one seed at a time; permutation importance (fitted before 2000, measured from 2000) | Lab Figure 20.4 |
| D. Complexity | Does complexity help market timing? | Ridge on 2 to 1,200 random Fourier features of the Goyal–Welch predictors, 12-month rolling window (Kelly, Malamud and Zhou 2024) | Lab Figure 20.5 |

### Design notes

- **No SHAP, no deep-learning framework.** The lab uses only scikit-learn so it installs anywhere. Permutation importance stands in for SHAP values; the networks are small.
- **Part D** is a compact version of the random-feature design, not a replication: one bandwidth, three ridge penalties and a few feature counts. Nagel (2025) argues that in short windows these models reduce to volatility-timed momentum; Exercise 5 tests that.
- **Run time** is dominated by the neural network. Use `--fast` while exploring and `--skip-c` to drop Part C.

## Code testing status

| Part | Status |
|---|---|
| Features | Tested: use only data up to t; target is t+1; demeaned target averages to zero |
| Walk-forward split | Tested: no training target realized after the refit date |
| Long–short and turnover | Tested on a known ranking |
| Models | Tested: find a planted signal (R² > 0.5%) and find nothing when there is none (R² < 0.1%) |
| Random-feature timing | Tested: runs and returns finite results |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab20.py` (about 20 seconds).

## Exercises

1. Drop the two market variables. Do the trees and the network lose more than OLS?
2. Restrict training and testing to the 49 industries only. Is there anything left to find?
3. Change the long–short book to the top and bottom 20%. How do Sharpe ratio and turnover change?
4. Add the Fama–French profitability and investment portfolios (`100_Portfolios_ME_OP_10x10`, `100_Portfolios_ME_INV_10x10`) to the panel. Does more data help the flexible models?
5. In Part D, compare the random-feature timing strategy with a simple rule: hold the market in proportion to its average return over the past 12 months, divided by its variance. How correlated are the two?

## Data citation

Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (100 Portfolios Formed on Size and Book-to-Market; 49 Industry Portfolios; Fama/French 3 Factors).
Goyal, A., and Welch, I. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455–1508 (updated data from Amit Goyal's website; Part D).
