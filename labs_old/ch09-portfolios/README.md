# Lab 9: Do Optimized Portfolios Beat Naive Ones?

Companion lab for **Chapter 9** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** portfolios built by optimization deliver better out-of-sample risk-adjusted returns than simple rules such as equal weighting (1/N).

## Quick start

```bash
pip install -r requirements.txt
python lab09.py                  # downloads four files from the Kenneth R. French Data Library
python lab09.py --fast           # six rules instead of nine (skips the slower constrained optimizers)
python lab09.py --window 240 --cost 0.005
```

A full run takes about 5 minutes on a laptop; `--fast` takes about 1 minute.

If downloads are blocked, save the zip files from the [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) in `data/` and pass them in:

| File | Library name | Option |
|---|---|---|
| `F-F_Research_Data_Factors_CSV.zip` | Fama/French 3 Factors (for the risk-free rate) | `--ff3-file` |
| `10_Industry_Portfolios_CSV.zip` | 10 Industry Portfolios | `--ind10-file` |
| `30_Industry_Portfolios_CSV.zip` | 30 Industry Portfolios | `--ind30-file` |
| `25_Portfolios_5x5_CSV.zip` | 25 Portfolios Formed on Size and Book-to-Market (5 x 5) | `--sbm25-file` |

The lab uses the value-weighted monthly returns in each file, minus the one-month Treasury bill rate.

Or open `lab09_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab09.py --source sim
```

The simulation draws assets from a one-factor model in which all assets have similar true Sharpe ratios, so 1/N is close to optimal by construction and the gain from estimating means is negative. Results on simulated data test the code only.

## What the lab does

Every rule is estimated on the previous `--window` months (default 120) and held for the next month; nothing from the holding month is used. Turnover is measured against the weights after they drift with the month's returns.

| Rule | Inputs estimated | Source |
|---|---|---|
| 1/N | None | DeMiguel, Garlappi and Uppal (2009) |
| Mean-variance (sample) | Means and covariance; tangency weights scaled to sum to 1, gross exposure capped at 5 | Markowitz (1952) |
| Mean-variance, long-only | Means and covariance; risk aversion 5, weights 0 to 1 | Frost and Savarino (1988) |
| Minimum variance (sample) | Covariance | |
| Minimum variance, long-only | Covariance | Jagannathan and Ma (2003) |
| Minimum variance, Ledoit–Wolf | Shrunk covariance | Ledoit and Wolf (2004) |
| Risk parity (inverse volatility) | Volatilities | |
| Equal risk contribution | Covariance | Maillard, Roncalli and Teiletche (2010) |
| Hierarchical risk parity | Correlation clusters and variances | López de Prado (2016) |

| Part | Question | Method | Output |
|---|---|---|---|
| A. The rules | Which rule wins out of sample? | Nine rules on 10 industries; Sharpe ratio, certainty equivalent (γ = 1), turnover; Jobson–Korkie–Memmel test against 1/N | Lab Figures 9.1–9.2 |
| B. Asset sets | Does the answer depend on the assets? | 10 industries, 30 industries, 25 size/book-to-market; net of 25 bp per unit of turnover | Lab Figure 9.3 |
| C. Stress | How robust is the ranking? | Windows of 60, 120 and 240 months; costs 0, 25, 50 bp; rebalancing every 1, 3 or 12 months; returns in 2008, 2020 and 2022 | Lab Figure 9.4 |
| D. Why | Where does optimization go wrong? | Gross exposure, largest weight, share of short positions, month-to-month weight changes | Lab Figure 9.5 |

### Design notes

- **Unconstrained mean-variance** can produce extreme weights when the estimated tangency portfolio's net exposure is near zero. The cap on gross exposure (5) keeps the backtest finite; it is a choice, and Exercise 2 asks you to change it.
- **The Sharpe-difference test** assumes returns are serially uncorrelated and roughly normal; with fat tails it overstates precision somewhat. Ledoit and Wolf (2008) give a robust alternative.
- **Industry and size/book-to-market portfolios** are already diversified. Results with individual stocks, where estimation error is worse, are typically less favourable to optimization.

## Verification status

| Part | Status |
|---|---|
| Minimum-variance and tangency weights | Tested against closed-form solutions |
| Risk parity, equal risk contribution, HRP | Tested: weights sum to 1, risk contributions equal, inverse-volatility ordering |
| Ledoit–Wolf | Tested: shrinkage intensity between 0 and 1, moves eigenvalues toward their mean |
| Walk-forward | Tested: no look-ahead, turnover computed against drifted weights |
| Sharpe-difference test | Tested: correct size when true Sharpe ratios are equal |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab09.py`.

## Exercises

1. Use a 360-month window. How much closer does sample mean-variance get to 1/N?
2. Change the gross-exposure cap on unconstrained mean-variance from 5 to 2 and to 20. What happens to the Sharpe ratio and turnover?
3. Add a rule that mixes 1/N and minimum variance, Ledoit–Wolf, half and half. Does it beat both?
4. Replace the sample mean with the cross-sectional average mean (a crude Bayes–Stein estimator). Does mean-variance improve?
5. Run the lab on 49 industries. Does adding assets help or hurt the optimized rules?

## Data citation

Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (10 and 30 Industry Portfolios; 25 Portfolios Formed on Size and Book-to-Market; Fama/French 3 Factors).
