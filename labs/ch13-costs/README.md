# Lab 13: What Survives the Trip from Paper to Portfolio?

Companion lab for **Chapter 13** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** a strategy that is profitable in a backtest is profitable to trade.

## Quick start

```bash
pip install -r requirements.txt
python lab13.py
```

The first run downloads three files from the Kenneth R. French Data Library and caches them in `data/`. The lab runs in a few seconds. Open `report/lab13_report.md` for the results. For a guided version, open `lab13_walkthrough.ipynb`; it asks you to predict each result before you see it.

### If the download fails

Download these CSV files from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and save them in `data/`, or pass them directly:

| Library page title | File |
|---|---|
| Fama/French 3 Factors | `F-F_Research_Data_Factors_CSV.zip` |
| Momentum Factor (Mom) | `F-F_Momentum_Factor_CSV.zip` |
| Short-Term Reversal Factor (ST Rev) | `F-F_ST_Reversal_Factor_CSV.zip` |

```bash
python lab13.py --ff3-file F-F_Research_Data_Factors_CSV.zip \
                --mom-file F-F_Momentum_Factor_CSV.zip \
                --rev-file F-F_ST_Reversal_Factor_CSV.zip
```

### Testing the code without real data

```bash
python lab13.py --source sim
```

**Results from simulated factor data are not evidence about markets;** the report and figures are labeled. The lab never switches to simulated data on its own.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Gross performance | How good are the strategies on paper? | Mean, t-statistic and Sharpe ratio of HML, Mom and ST Rev, July 1963 onward | Table |
| B. Turnover | How much does each strategy trade, and why? | Published turnover (Novy-Marx and Velikov 2016) beside turnover measured in a simulated stock universe with Fama–French portfolio construction | Lab Figure 13.3 |
| C. Costs | Which strategies survive trading costs? | Net returns for one-way costs from 0 to 100 bp plus a borrow fee; break-even costs; rankings before and after costs | Lab Figures 13.1–13.2 |
| D. Capacity | How large can each strategy grow? | Half-spreads plus square-root market impact in a simulated universe scaled to a $50 trillion market | Lab Figure 13.4 |
| E. Mitigation | Can design reduce costs? | Buy/hold spreads in the simulated universe, measured with the true expected returns | Lab Figure 13.5 |

### Design notes

- **Cost accounting.** One-sided turnover T is the share of each side sold (and replaced) per month. A long-short portfolio with $1 on each side trades 2T on each side, so its monthly cost is 4 × T × c, where c is the one-way cost per dollar traded (half the effective spread plus commissions and impact). The borrow fee is charged on the short side. Novy-Marx and Velikov's Table 3 costs imply about the same c (about 45 bp) for all three strategies under this formula; a test checks it.
- **Published turnover** is the value-weighted, full-sample average of the long and short sides from Novy-Marx and Velikov (2016), Table 3 (NBER working paper w20721): value 2.91%, momentum 34.52%, short-term reversal 90.87% a month. The paper's own cost estimates use stock-level effective spreads that vary over time; the lab applies a single cost to make the trade-off transparent. `--turnover simulated` uses the simulated universe's turnover instead.
- **The simulated universe** has 1,000 stocks over 40 years, lognormal market caps, fat-tailed returns, and expected returns that depend on three signals with known slopes. Portfolios are value-weighted 30/70 sorts. Value uses book-to-market each June from book equity and market equity six months earlier, held for a year; momentum uses returns from month −12 to −2; reversal uses the previous month's return. It has no delistings, new listings or changes in book equity accounting, so its turnover is a lower bound on real turnover.
- **Capacity assumptions** (all adjustable): average daily volume of 0.4% of market cap; half-spreads from 2 bp for the largest stocks to 50 bp for the smallest; square-root impact y × σ_daily × √(trade / volume) with y = 1; each month's trades spread over 5 days; trades allocated in proportion to holdings. Capacity numbers are illustrations of the mechanics, not estimates.
- **Part E** compares rules using the simulation's true expected returns, so differences are not sampling noise.

## Options

```text
--source {french,sim}     real data (default) or simulated factor data
--ff3-file / --mom-file / --rev-file PATH
--start / --end YYYY-MM   sample (default 1963-07 to the latest month)
--cost C                  headline one-way cost, decimal (default 0.002 = 20 bp)
--borrow-fee F            annual short-borrowing fee (default 0.0025)
--turnover {published,simulated}
--impact-y Y              square-root impact coefficient (default 1.0)
--trade-days D            days over which each month's trades are spread (default 5)
--total-cap X             total market cap of the simulated universe in $ (default 5e13)
--n-stocks N              stocks in the simulated universe (default 1000)
--seed N                  seed for the simulated universe (default 2026)
--sim-seed N              seed for simulated factor data (--source sim)
--out DIR                 output folder (default report/)
```

## Outputs (`report/`)

- `lab13_report.md`: all results, with a "Your verdict" prompt
- `fig13_1_net_vs_cost.png` to `fig13_5_banding.png`
- `table_gross.csv`, `table_turnover.csv`, `table_costs.csv`, `table_cost_sensitivity.csv`, `table_capacity.csv`, `table_banding.csv`
- `summary.json`: headline numbers for grading or automated checks

## Files

```text
ch13-costs/
├── README.md
├── requirements.txt
├── lab13.py                 # the lab: parts A–E, figures, report
├── lab13_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (the parts this lab uses)
│   ├── costs.py             # cost accounting, simulated universe, banding, impact, capacity
│   ├── data.py, stats.py, inference.py, multiple.py, predictors.py, forecast.py, plots.py
├── data/                    # downloaded files are cached here
└── tests/test_lab13.py      # runs offline
```

Run the tests with `python tests/test_lab13.py` (or `pytest tests/`).

## Verification status

| Part | Status |
|---|---|
| Cost formulas, break-even, borrow fee | Tested against hand calculations |
| Consistency with Novy-Marx and Velikov | Tested: their turnover and cost figures imply a common one-way cost under 4 × T × c |
| Turnover | Tested: zero for an unchanging value-weighted portfolio; matches a direct calculation |
| Signals | Tested: changing future data leaves past signals unchanged; value changes only in June |
| Simulated universe | Tested: turnover ordered value < momentum < reversal; true premia recovered; banding cuts turnover and keeps every stock a standard sort buys |
| Impact and capacity | Tested against the formula; net return falls with AUM |
| Factor file parsers | Tested on files in the library's layouts (including a padded `Mom` header); **confirm on the live files at first run** |
| Full lab and notebook | Run end to end on simulated factor data |
| Live download | Not tested from the build environment (network blocked) |

## Exercises

1. Re-run with `--start 2001-05`, after U.S. stocks moved to decimal prices and spreads narrowed, and with `--cost 0.001`. Does reversal survive in the low-cost era?
2. Re-run Part C with `--borrow-fee 0.02`, a fee typical of hard-to-borrow stocks. Which strategy is most exposed to shorting costs, and why?
3. Part D: halve and double the impact coefficient (`--impact-y`) and the trading window (`--trade-days`). Which assumption moves capacity most?
4. Part E: add a rule that rebalances momentum only every three months. Compare its turnover and expected return with banding.
5. Compute the cost of the market portfolio itself (turnover close to zero). Why is it the right benchmark for a cost-aware comparison?

## Data citation

Kenneth R. French Data Library: Fama/French 3 Factors; Momentum Factor (Mom); Short-Term Reversal Factor (ST Rev), monthly. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
