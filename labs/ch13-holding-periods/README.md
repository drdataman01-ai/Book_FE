# Lab 13: When Do Investors Sell, and What Does It Cost Them?

Companion lab for **Chapter 13** of *Financial Engineering in Practice*.

**What you will explore:** how the rule an investor uses to decide when to sell changes what the investor experiences and keeps.

## Quick start

```bash
pip install -r requirements.txt
python lab13.py
```

The first run downloads one file from the Kenneth R. French Data Library and caches it in `data/`. The lab runs in under a minute. Open `report/lab13_report.md` for the results. For a guided version, open `lab13_walkthrough.ipynb`; it asks you to predict each result before you see it.

### If the download fails

Download this CSV file from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and save it in `data/`, or pass it directly:

| Library page title | File |
|---|---|
| 49 Industry Portfolios | `49_Industry_Portfolios_CSV.zip` |

```bash
python lab13.py --industry-file 49_Industry_Portfolios_CSV.zip
```

### Testing the code without real data

```bash
python lab13.py --source sim
```

**Results from simulated industry data are not evidence about markets;** the report and figures are labeled. The lab never switches to simulated data on its own.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Optional stopping | Can a selling rule change the expected return of a fair game? | Six rules on the same 50,000 simulated martingale price paths: fixed 12 months, hold 120 months, profit target, stop-loss, bracket, and "get even" | Table, Lab Figure 13.1 |
| B. Random horizons | How should returns over different holding periods be compared? | With a positive drift, each sale annualized on its own against the log return per year invested (Wald's identity) | Table, Lab Figure 13.2 |
| C. Real prices | Do the rules earn the assets' average return per year invested on real data? | The same rules on the 49 value-weighted industry portfolios, a position bought every month in every industry, longest hold 60 months; moving-block bootstrap over entry months | Table, Lab Figure 13.3 |
| D. The disposition test | Do the winners a disposition investor sells beat the losers the investor keeps? | Sell at +20%, never sell below cost before the cap; next-12-month return of winners sold against losers held, Newey–West t-statistic | Table, Lab Figure 13.4 |
| E. Taxes and costs | Which selling rule leaves the most after tax? | 20,000 simulated investors over 30 years: buy and hold, sell every 12 months, sell winners at +20%, sell losers at −20% | Table, Lab Figure 13.5 |

### Design notes

- **Rules are checked at month ends** against the purchase price, using only prices up to that month. A test changes every price after each sale and confirms that no sale moves.
- **The cap.** Every rule sells after its longest holding period (120 months in Parts A and B, 60 in Parts C and D). A bounded horizon is what the optional stopping theorem needs, and the share of positions still below cost at the cap is the lab's measure of "losers held".
- **Comparing horizons.** "Log return per year invested" is the total log return of all positions divided by the total years they were held. Under geometric Brownian motion it equals μ − σ²/2 for every rule (Wald's identity), so a gap on real data points to a property that the simulation lacks, such as momentum or reversal.
- **Part C benchmark.** Each position's log return is compared with its industry's average monthly log return over the whole sample, times the months held. Positions bought in nearby months share returns, so standard errors come from a moving-block bootstrap over entry months with blocks as long as the cap.
- **Part D** follows Odean (1998) in comparing realized gains with paper losses, but uses industries rather than individual stocks, because individual-stock returns are not free. Industry portfolios are less volatile than single stocks, so the +20% target is reached less often; try `--up 0.1`.
- **Part E taxes.** Each sale pays tax at the stated rate on the realized gain; a realized loss earns a credit at the same rate, used at once. Gains are not split into short and long term. All rules see the same price shocks, so differences between them are not simulation noise.

## Options

```text
--source {french,sim}     real data (default) or simulated industry data
--industry-file PATH      49 Industry Portfolios (monthly) zip or CSV
--start / --end YYYY-MM   sample for Parts C and D (default: the whole file)
--up U / --down D         profit target and stop-loss, decimals (default 0.2 and 0.2)
--cap N                   longest hold in Parts C and D, months (default 60)
--sim-cap N               longest hold in Parts A and B, months (default 120)
--paths N                 simulated paths in Parts A and B (default 50000)
--sigma S / --mu M        volatility and expected return in the simulations (default 0.30 and 0.08)
--tax T / --cost C        tax rate and one-way cost in Part E (default 0.20 and 0.001)
--years N                 horizon in Part E (default 30)
--tax-paths N             simulated investors in Part E (default 20000)
--boot N                  bootstrap replications in Part C (default 400)
--seed N                  seed for the simulations (default 2026)
--sim-seed N              seed for simulated industry data (--source sim)
--out DIR                 output folder (default report/)
```

## Outputs (`report/`)

- `lab13_report.md`: all results, with questions to think about
- `fig13_1_distributions.png` to `fig13_5_taxes.png`
- `table_optional_stopping.csv`, `table_per_year.csv`, `table_real_prices.csv`, `table_disposition_by_decade.csv`, `table_taxes.csv`
- `summary.json`: headline numbers for grading or automated checks

## Files

```text
ch13-holding-periods/
├── README.md
├── requirements.txt
├── lab13.py                 # the lab: parts A–E, figures, report
├── lab13_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (the parts this lab uses)
│   ├── holding.py           # selling rules, stopping times, disposition test, taxes
│   ├── data.py, stats.py, plots.py
├── data/                    # downloaded files are cached here
└── tests/test_lab13.py      # runs offline
```

Run the tests with `python tests/test_lab13.py` (or `pytest tests/`).

## Code testing status

| Part | Status |
|---|---|
| Selling rules | Tested on hand-made price paths for every rule and exit reason |
| Look-ahead | Tested: changing prices after a sale never changes the sale |
| Optional stopping | Tested: every rule's mean return is zero within four standard errors on 200,000 martingale paths |
| Wald's identity | Tested: log return per year invested equals μ − σ²/2 for every rule |
| Positions from a panel | Tested: holding-period returns match direct compounding; assets that start late enter only when they have data |
| Disposition test | Tested: detects a gap when expected returns persist, and none when they do not |
| Tax accounting | Tested against a hand calculation; deferral ranks as theory predicts |
| Industry file parser | Tested on the library's layout (value- and equal-weighted tables, −99.99 as missing); **confirm on the live file at first run** |
| Full lab and notebook | Run end to end on simulated industry data |
| Live download | Not tested from the build environment (network blocked) |

## Exercises

1. Part A: raise the profit target to +50% and lower the stop to −50%. Which rule now has the highest win rate, and does any mean move away from zero?
2. Part C: re-run with `--start 1963-07` and with `--cap 24`. Do the rules' gaps change sign? Which property of industry returns, momentum or long-run reversal, would explain each sign?
3. Part D: re-run with `--up 0.1`. Compare your gap with Odean's (1998) figure of about 3.4 percentage points over the following year for individual stocks. Why might industries show a smaller gap?
4. Part E: set `--tax 0` and confirm that the rules' wealth nearly coincides. Then set `--cost 0.005`. Which rule is hurt most by costs, and which by taxes?
5. Dollar-weighted returns (Dichev 2007): add a column to Part C that invests $1 more in each industry after a 12-month gain of more than 20%. Compute the internal rate of return of the resulting cash flows and compare it with the time-weighted return.

## Data citation

Kenneth R. French Data Library: 49 Industry Portfolios, monthly, value-weighted. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
