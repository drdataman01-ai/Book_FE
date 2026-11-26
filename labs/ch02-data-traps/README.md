# Lab 2: Four Data Traps

Companion lab for **Chapter 2** of *Financial Engineering in Practice*.

**What you will explore:** how the way returns are averaged, missing failures, look-ahead and tiny stocks change the return a backtest appears to earn.

The lab measures four ways data can flatter a result. Three use nearly a century of real U.S. market data. One, survivorship, uses a simulated market where the right answer is known, because free data sources do not include delisted stocks.

## Quick start

```bash
pip install -r requirements.txt
python lab02.py
```

The first run downloads three files from the Kenneth R. French Data Library and caches them in `data/`. The lab then runs in about 10 seconds. Open `report/lab02_report.md` for the results.

For a guided version, open `lab02_walkthrough.ipynb` in Jupyter. It asks you to predict each result before you see it.

### If the download fails

Download these three CSV files from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and either save them in `data/` or pass them directly:

| Library page title | File |
|---|---|
| Fama/French 3 Factors | `F-F_Research_Data_Factors_CSV.zip` |
| Fama/French 3 Factors [Daily] | `F-F_Research_Data_Factors_daily_CSV.zip` |
| Portfolios Formed on Size | `Portfolios_Formed_on_ME_CSV.zip` |

```bash
python lab02.py --monthly-file F-F_Research_Data_Factors_CSV.zip \
                --daily-file F-F_Research_Data_Factors_daily_CSV.zip \
                --size-file Portfolios_Formed_on_ME_CSV.zip
```

### Testing the code without real data

```bash
python lab02.py --source sim
```

This uses simulated returns for Parts A, C and D. **Results from simulated data are not evidence about markets;** the report and figures are labeled. The lab never switches to simulated data on its own.

## The four traps

| Part | Trap | What the lab does | Data |
|---|---|---|---|
| A | Averages | Compares arithmetic and geometric average returns for Treasury bills, the market, and the largest and smallest 10% of stocks | Real |
| B | Survivorship | Simulates about 6,000 stocks over 30 years, all with the same expected return. Compares a full dataset with one missing delisting returns and one with survivors only | Simulated by design |
| C | Look-ahead | (C1) A 200-day moving-average timing rule, traded correctly versus with one day of look-ahead. (C2) The rule's window chosen with hindsight versus walk-forward | Real, daily |
| D | Micro-caps and weighting | The size premium (smallest minus largest decile) in value- and equal-weighted portfolios, with and without the smallest decile, before and after Banz (1981) | Real |

### Design notes

- **Part A** annualizes both averages the same way, by compounding the monthly figure, so the comparison is like for like.
- **Part B** delists a stock for poor performance when its price falls below 20% of its listing price, with a final return of −30% (Shumway 1997). It also removes about 0.3% of stocks a month through acquisitions at a 20% premium, and replaces every exit with a new listing. The volatility strategy sorts stocks each month on their trailing 12-month volatility, using only past data.
- **Part C** holds Treasury bills when out of the market. The look-ahead version uses the same day's closing price to decide the position it is credited with that day. The walk-forward version chooses, each January, the window with the best Sharpe ratio up to the previous December; both versions are evaluated over the same period, after a 10-year warm-up.
- **Part D** uses the last ten columns of the French size file (the deciles), so it does not depend on exact column names. Statistics use Newey–West standard errors.

## Options

```text
--source {french,sim}     real data (default) or simulated data for testing
--monthly-file PATH       Fama/French 3 Factors (monthly)
--daily-file PATH         Fama/French 3 Factors [Daily]
--size-file PATH          Portfolios Formed on Size
--ma-window N             Part C rule window in trading days (default 200)
--publication-split M     last month before the size effect was published (default 1980-12)
--sim-firms N             Part B: stocks in the simulated market (default 2000)
--sim-years N             Part B: years simulated (default 30)
--delist-return R         Part B: delisting return (default -0.30)
--seed N                  Part B seed (default 2026)
--out DIR                 output folder (default report/)
```

## Outputs (`report/`)

- `lab02_report.md`: all results, with questions to think about
- `fig2_1_averages.png` to `fig2_5_size_deciles.png`
- `table_*.csv`: every table in the report, plus the windows chosen each year in Part C2
- `summary.json`: headline numbers for grading or automated checks

## Files

```text
ch02-data-traps/
├── README.md
├── requirements.txt
├── lab02.py                 # the lab: data, parts A–D, figures, report
├── lab02_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (the parts this lab uses)
│   ├── data.py              # French library download; single- and multi-table parsing
│   ├── stats.py             # returns, Sharpe ratios, drawdowns, Newey–West
│   └── plots.py             # chart style
├── data/                    # downloaded files are cached here
└── tests/test_lab02.py      # runs offline
```

Run the tests with `python tests/test_lab02.py` (or `pytest tests/`). In the book's repository, `fintest/` is shared by all labs; this folder carries its own copy so it runs on its own.

## Code testing status

| Part | Status |
|---|---|
| Statistics and the four analyses | Tested end to end on simulated data |
| No-look-ahead checks | Tested: the volatility sort ignores future data; the look-ahead rule beats the correct rule on known data |
| Survivorship simulation | Tested: survivors-only data overstates returns relative to the full data |
| Notebook | All cells executed on simulated data |
| French library parsers | Tested on files in the library's published layouts, including the multi-table size file; **confirm on the live files at first run** |
| Live download from the French library | Not tested from the build environment (network blocked); the error message explains the manual route |

## Exercises

1. Part B: re-run with `--delist-return -0.55`, closer to estimates for Nasdaq stocks. How do the three datasets change?
2. Part C: try `--ma-window 50` and `--ma-window 300`. Is the look-ahead gain larger for fast or slow rules? Explain why.
3. Part C: add a one-day trading cost of 0.1% each time the rule switches. Does the correctly traded rule still beat buy-and-hold?
4. Part D: change `--publication-split` to 1990-12 and 2000-12. When does the size premium appear to fade?
5. **Challenge:** using a free price source such as Yahoo Finance, download today's S&P 500 members and backtest an equal-weighted portfolio of them over the past 20 years. Compare it with the equal-weighted S&P 500 index over the same period. Which trap from this lab explains the difference?

## Data citation

Kenneth R. French Data Library: Fama/French 3 Factors (monthly and daily); Portfolios Formed on Size. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
