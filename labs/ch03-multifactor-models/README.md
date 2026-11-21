# Lab 3: Do Size and Value Explain Returns That Beta Cannot?

Companion lab for **Chapter 3** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** additional factors improve our ability to explain differences in portfolio returns, but testing many candidate factors can produce false discoveries.

## Quick start

```bash
pip install -r requirements.txt
python lab03.py
```

The first run downloads two files from the Kenneth R. French Data Library and caches them in `data/`. Open `report/lab03_report.md`, or work through `lab03_walkthrough.ipynb`.

### If a download fails

Download the CSV versions from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and pass them with the options below.

| Library page title | File | Option |
|---|---|---|
| Fama/French 3 Factors | `F-F_Research_Data_Factors_CSV.zip` | `--factors-file` |
| 25 Portfolios Formed on Size and Book-to-Market (5 x 5) | `25_Portfolios_5x5_CSV.zip` | `--portfolios-file` |

### Testing the code without real data

```bash
python lab03.py --source sim
```

Simulated results are not evidence about markets; the report and figures are labeled.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Fama and French (1993) | Do SMB and HML explain what beta cannot? | CAPM and three-factor regressions on the 25 value-weighted size/book-to-market portfolios, July 1963 on: alphas with Newey–West t, loadings, R², GRS test | Lab Figures 3.1–3.3 |
| B. Multiple testing | How many useless factors look significant? | 1,000 simulated factors with a true premium of zero, tested at \|t\| > 1.96 and \|t\| > 3.0, 200 runs; independent and correlated (ρ = 0.5) | Lab Figure 3.4 |
| C. Economic significance | Are the premiums large enough to matter? | Annualized premium, Sharpe ratio and t of Mkt-RF, SMB and HML, full sample and three subperiods; break-even one-way cost at an assumed turnover | Lab Figure 3.5 |

### Design notes

- **Sample.** July 1963 onward by default, as in Fama and French (1993); use `--start 1926-07` for the full history.
- **Portfolios** are taken by position from the value-weighted table, so the code does not depend on exact column names.
- **Part B** uses the length of the real sample for each simulated factor and a monthly volatility of 3%. The correlated design shares one common shock across all factors; it does not change the expected count, but it makes the count far more variable.
- **Break-even cost** = annual premium / (4 × 12 × monthly turnover per leg). The default turnover of 3% a month is an assumption, not a measurement; change it with `--turnover`.

## Options

```text
--source {french,sim}   real data (default) or simulated data
--factors-file PATH     Fama/French 3 Factors (monthly) zip or CSV
--portfolios-file PATH  25 Portfolios Formed on Size and Book-to-Market zip or CSV
--start, --end          sample for Parts A and C (default 1963-07 to latest)
--nw-lags N             Newey-West lags (default: rule of thumb)
--n-factors, --n-reps   Part B design (default 1,000 factors, 200 runs)
--corr RHO              Part B correlated design (default 0.5)
--turnover X            Part C assumed monthly turnover per leg (default 0.03)
```

## Outputs (`report/`)

- `lab03_report.md`: all results, with a "Your verdict" prompt
- `fig3_1_alpha_grid.png` to `fig3_5_premiums.png`
- `table_*.csv`: regressions for each model, false discoveries, premiums, break-even costs
- `summary.json`: headline numbers for grading or automated checks

## Files

```text
ch03-multifactor-models/
├── README.md
├── requirements.txt
├── lab03.py                 # the lab: data, parts A–C, figures, report
├── lab03_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (the parts this lab uses)
├── data/                    # downloaded files are cached here
└── tests/test_lab03.py      # runs offline
```

Run the tests with `python tests/test_lab03.py` (or `pytest tests/`).

## Verification status

| Part | Status |
|---|---|
| Regressions | Tested against a direct least-squares fit; nested R² checked |
| False-discovery experiment | Tested: about 5% of useless factors clear \|t\| > 1.96 |
| Break-even cost | Tested against the formula |
| Full lab and notebook | Run end to end on simulated data |
| French library parser | Tested on a file in the library's published layout; **confirm on the live files at first run** |
| Live download | Not tested from the build environment (network blocked); the error message explains the manual route |

## Exercises

1. Re-run Part A with `--start 1926-07`. Do the conclusions change?
2. Which portfolio keeps a significant three-factor alpha? Compare it with Fama and French (1993).
3. Re-run Part B with `--corr 0.8`. What happens to the worst run?
4. Part C: with what monthly turnover would HML's premium be erased by a one-way cost of 0.5%?

## Data citation

Kenneth R. French Data Library: Fama/French 3 Factors (monthly); 25 Portfolios Formed on Size and Book-to-Market (5 x 5). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
