# Lab 5: Do Simple Trading Rules Beat Buy-and-Hold?

Companion lab for **Chapter 5** of *Financial Engineering in Practice*.

**What you will explore:** how trend rules perform against buy-and-hold, and how much of their apparent success remains after correcting for the number of rules tried.

## Quick start

```bash
pip install -r requirements.txt
python lab05.py
```

The first run downloads the data listed below and caches it in `data/`. Open `report/lab05_report.md`, or work through `lab05_walkthrough.ipynb`. Parts A and B need only the French library; if the AQR file or a FRED series cannot be downloaded, the lab says which part it skipped and carries on.

### Data (all free)

| Source | Series | Used in | Option |
|---|---|---|---|
| [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Fama/French 3 Factors [Daily] (`F-F_Research_Data_Factors_daily_CSV.zip`) | A, B | `--ff3-daily-file` |
| Kenneth R. French Data Library | Fama/French 3 Factors, monthly (`F-F_Research_Data_Factors_CSV.zip`) | C, D, E | `--ff3-file` |
| [AQR data sets](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Factors-Monthly) | Time Series Momentum: Factors, Monthly (xlsx), from January 1985 | C, D | `--aqr-file` |
| [FRED](https://fred.stlouisfed.org) | GS10 (10-year Treasury yield), MCOILWTICO (WTI crude), H.10 exchange rates and three-month interbank rates for nine currencies (as in Lab 11) | E | `--fred-dir` (CSVs named `<SERIES>.csv`) |
| [Stooq](https://stooq.com) (optional) | Daily Dow Jones index, `^dji` (Stooq asks for a free key and limits daily downloads) | A | `--dow-file` |

The French market series is a total-return index; Brock, Lakonishok and LeBaron used the Dow Jones price index, without dividends. Pass `--dow-file` to use it in Part A.

### Testing the code without real data

```bash
python lab05.py --source sim          # about 20 seconds
python lab05.py --source sim --fast   # a smaller rule universe, for a quick check
```

Simulated results are not evidence about markets; the report and figures are labeled.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. The 26 rules | Did Brock, Lakonishok and LeBaron's rules predict returns, and do they still? | Variable-length moving averages (VMA), fixed-length moving averages (FMA) and trading-range breakouts (TRB), with 0% and 1% bands; buy and sell counts, mean returns, buy–sell difference and t; 1926–1986 and 1987 on | Lab Figure 5.1 |
| B. The search correction | Is the best rule better than luck once the search is counted? | 1,764 long-or-cash rules (moving-average crossovers and breakouts, seven bands, four holding periods) against buy-and-hold; White's Reality Check and Hansen's SPA test, stationary bootstrap (mean block 10 days, 1,000 samples); both periods and rolling ten-year windows | Lab Figure 5.2 |
| C. Across asset classes | Does trend following earn a premium? | AQR's TSMOM factors: mean, Sharpe ratio, alpha against the U.S. market, before and after 2009 | Table |
| D. The smile | When does trend following pay? | Quarterly TSMOM on the market and its square; the ten worst quarters for U.S. stocks | Lab Figure 5.3 |
| E. Asset by asset | Is it time-series momentum, or each asset's average return? | Huang, Li, Wang and Zhou (2020): separate regressions of each asset's volatility-scaled return on its past 12-month return, the pooled slope with a block-bootstrap p-value, and TSM against the time-series history strategy (TSH) | Lab Figures 5.4–5.5 |

### Design notes

- **No look-ahead.** Every signal at the close of day *t* uses prices up to *t* and is applied to day *t*+1; a test checks that changing future prices never changes past positions.
- **Fixed-length rules** report 10-day returns after each signal and ignore new signals during the 10-day holding period, as Brock and coauthors did.
- **Part B performance** is the daily return of each rule (market when long, Treasury bills otherwise) minus buy-and-hold. `--cost` charges a one-way cost each time a rule switches.
- **Part E assets** are U.S. stocks, a 10-year Treasury bond priced from GS10, nine currencies (excess returns from interest differentials and spot changes) and crude oil. Oil uses the spot price change less the bill rate, a proxy for a futures return. Volatility is a 36-month rolling standard deviation known before each month.

## Options

```text
--source {download,sim}   real data (default) or simulated data
--ff3-daily-file, --ff3-file, --aqr-file, --dow-file, --fred-dir   local copies of the data
--n-boot N                Part B bootstrap samples (default 1,000)
--block L                 Part B mean block length in days (default 10)
--cost C                  Part B one-way cost per switch, decimal (default 0)
--fast                    smaller rule universe and fewer bootstrap samples
```

## Outputs (`report/`)

- `lab05_report.md`: all results, with questions to think about
- `fig5_1_rules.png` to `fig5_5_tsm_tsh.png`
- `table_*.csv` and `summary.json`

## Files

```text
ch05-trading-rules/
├── README.md
├── requirements.txt
├── lab05.py                 # the lab: data, parts A–E, figures, report
├── lab05_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (the parts this lab uses)
│   └── search.py            # rule universe, Reality Check, SPA
├── data/                    # downloaded files are cached here
└── tests/test_lab05.py      # runs offline
```

## Code testing status

| Part | Status |
|---|---|
| Signals | Tested: no look-ahead; moving-average, breakout, holding and thinning rules checked on hand-made prices |
| Reality Check and SPA | Tested: size near 5% under the null; a rule with real skill is found with p < 0.01 |
| Stationary bootstrap | Tested: new blocks start at the rate 1 / mean block |
| AQR and Stooq readers | Tested on files in the published layouts; **confirm on the live files at first run** |
| Full lab and notebook | Run end to end on simulated data |
| Live downloads | Not tested from the build environment (network blocked); each error message explains the manual route |

## Exercises

1. Re-run Part B with `--cost 0.0005`. How many rules beat buy-and-hold now?
2. Restrict the universe to the 26 original rules (edit `rule_universe`). Does the Reality Check p-value fall?
3. In Part E, replace the 12-month lookback with 3 and 24 months. Is 12 special?

## Data citation

Kenneth R. French Data Library; AQR Capital Management, Time Series Momentum: Factors, Monthly; Federal Reserve Bank of St. Louis, FRED.
