# Lab 1: Brilliant by Chance

Companion lab for **Chapter 1** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** a strategy with an impressive backtest has skill.

You build 1,000 market-timing strategies that switch between U.S. stocks and Treasury bills **at random**, so every one has zero skill by construction. You backtest them on the first half of nearly a century of real market data, pick the winners, and follow them into the second half.

## Quick start

```bash
pip install -r requirements.txt
python lab01.py
```

The first run downloads the monthly Fama–French factor file from the Kenneth R. French Data Library and caches it in `data/`. The lab takes a few seconds. Open `report/lab01_report.md` for the results.

For a guided version, open `lab01_walkthrough.ipynb` in Jupyter. It runs the same analysis step by step and asks you to predict results before you see them.

### If the download fails

Some networks block the data site. Download **Fama/French 3 Factors** (monthly, CSV) from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), then run:

```bash
python lab01.py --data-file F-F_Research_Data_Factors_CSV.zip
```

### Testing the code without real data

```bash
python lab01.py --source sim
```

This uses simulated returns with realistic averages and volatility. **Results from simulated data are not evidence about markets;** every figure and the report are labeled when it is used. The lab never switches to simulated data on its own.

## What the lab does

| Part | Question | Output |
|---|---|---|
| A. Signal and noise | How good is the market itself, and how much does a 10-year record vary? | Lab Figure 1.1, years-to-detect table |
| B. Brilliant by chance | How many zero-skill strategies look significant in a backtest, and what happens to them afterward? | Lab Figures 1.2–1.4, top-10 table |
| C. The cost of searching | How does the best backtest improve as more strategies are tried? | Lab Figure 1.5 |

**The random strategies.** Each strategy is in the market or in Treasury bills each month. It switches at random, with an average holding period between 3 and 24 months drawn for each strategy, so the strategies look like real timing rules but use no information. Positions are set before each month's return is known.

**Measuring skill.** A strategy that is in the market part of the time naturally earns part of the market's return. Skill is therefore measured as **alpha against buy-and-hold**: the intercept from regressing the strategy's excess return on the market's excess return, with Newey–West standard errors. For a strategy with no skill, the true alpha is zero.

**Backtest and after selection.** By default the sample is split in the middle (about 1926–1975 and 1976–today). Use `--split 1990-12` to change it.

## Options

```text
--source {french,sim}   real data (default) or simulated data for testing
--data-file PATH        a French library zip or CSV you downloaded yourself
--n-strategies N        number of random strategies (default 1000)
--split YYYY-MM         last month of the backtest (default: middle of the sample)
--window-years N        window for rolling Sharpe ratios (default 10)
--nw-lags N             Newey–West lags (default: rule of thumb)
--seed N                seed for the random strategies (default 2026)
--out DIR               output folder (default report/)
```

## Outputs (`report/`)

- `lab01_report.md`: all results, with a "Your verdict" prompt
- `fig1_1_rolling_sharpe.png` to `fig1_5_search_cost.png`
- `table_years_to_detect.csv`, `table_top10.csv`, `table_search_cost.csv`, `table_all_strategies.csv`
- `summary.json`: every headline number, for grading or automated checks

## What to expect

About 2.5% of zero-skill strategies should pass t > 1.96 by luck, which is about 25 of 1,000. Your count will vary, often between about 15 and 50, for two reasons that are themselves worth understanding. First, all 1,000 strategies are tested on the same single market history, so their results are not fully independent. Second, heavy-tailed returns make standard errors slightly too small in finite samples. Re-run with different `--seed` values and compare.

## Files

```text
ch01-brilliant-by-chance/
├── README.md
├── requirements.txt
├── lab01.py                 # the lab: data, parts A–C, figures, report
├── lab01_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (Lab 1 subset)
│   ├── data.py              # Kenneth French Data Library download and parsing
│   ├── stats.py             # Sharpe ratios, Newey–West, multiple-testing helpers
│   └── plots.py             # chart style
├── data/                    # downloaded files are cached here
└── tests/test_lab01.py      # runs offline: parser, statistics, full lab on simulated data
```

Run the tests with `python tests/test_lab01.py` (or `pytest tests/`).

## Verification status

| Part | Status |
|---|---|
| Statistics (Sharpe ratio, Newey–West, Bonferroni, normal quantiles) | Tested against manual calculations |
| Random strategies and the full lab pipeline | Tested end to end on simulated data |
| Notebook | All cells executed on simulated data |
| French library parser | Tested on a file in the library's published layout; **confirm on the live file at first run** |
| Live download from the French library | Not tested from the build environment (network blocked); the error message explains the manual route |

## Exercises

1. Re-run with five different seeds. Record the number of strategies with t > 1.96 and the best t each time.
2. Run `--n-strategies 10000`. How large is the best backtest t-statistic now? Compare it with Lab Figure 1.5.
3. Move the split to 1990-12. Does anything change in the conclusions?
4. **Challenge:** add a "needle" strategy with genuine but modest skill. For example, it could be in the market with probability 0.55 in months when the market rises and 0.45 when it falls. This uses future information, so it is skill by construction. Does it rank near the top of the 1,000 in the backtest? What does that tell you about finding real skill among many tries?

## Data citation

Kenneth R. French Data Library, Fama/French 3 Factors (monthly). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
