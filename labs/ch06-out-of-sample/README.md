# Lab 6: Do Valuations Predict the Equity Premium Out of Sample?

Companion lab for **Chapter 6** of *Financial Engineering in Practice*.

**What you will explore:** whether predictors that are significant in sample beat the historical mean out of sample.

## Quick start

1. Download the Goyal–Welch predictor data from [Amit Goyal's website](https://sites.google.com/view/agoyal145) (under Goyal and Welch 2008, **Updated data**; free). Save it in `data/` with "predictor" or "goyal" in the file name. The links on that page point to Google Drive, so the lab does not download the file itself.
2. Run:

```bash
pip install -r requirements.txt
python lab06.py
```

The lab runs in under a minute. Open `report/lab06_report.md`, or work through `lab06_walkthrough.ipynb`.

### Testing the code without real data

```bash
python lab06.py --source sim
```

Simulated results are not evidence about markets; the report and figures are labeled. In the simulation only the dividend–price ratio truly predicts the premium.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. In sample | Which predictors look significant on the whole history? | Next month's equity premium on each of the 14 predictors: slope, White t, R² | Lab Figure 6.1 |
| B. Walk-forward | Do they forecast? | Expanding-window forecasts from January 1965 (at least 20 years of training data); out-of-sample R² against the historical mean; Clark–West and Diebold–Mariano tests; cumulative squared-error differences; the kitchen-sink regression | Lab Figure 6.3 |
| C. Remedies | Can restrictions or combinations rescue them? | Campbell–Thompson sign and non-negativity restrictions; the equal-weighted combination of the 14 forecasts; certainty-equivalent gain of a mean-variance investor (risk aversion 3, weights 0 to 150%) | Lab Figures 6.2 and 6.4 |
| D. After 2005 | Did anything keep working after Goyal and Welch's original sample? | The same forecasts, evaluated before and after January 2005 | Table |
| E. Reality Check | How good does the best of many useless rules look? | 840 long-or-cash moving-average and breakout rules on a simulated market with no equity premium and no predictability; nominal p-value of the best rule against White's Reality Check and Hansen's SPA test | Lab Figure 6.5 |

### The 14 predictors

Dividend–price ratio, dividend yield, earnings–price ratio and dividend payout ratio (logs); book-to-market; Treasury-bill rate, long-term yield, long-term bond return and term spread; default yield spread (Baa − Aaa) and default return spread; stock variance; net equity expansion; inflation, lagged one month for its publication delay.

### Design notes

- **Target.** The simple equity premium of month *t*+1 (CRSP S&P 500 total return minus the Treasury bill rate), paired with predictors known at the end of month *t*.
- **No look-ahead.** Every forecast uses only data available at its date; a test checks that changing future premiums never changes past forecasts.
- **Clark–West** is the right test for a model that nests the historical mean; Diebold–Mariano is reported for comparison and is undersized here.
- **Part E** needs no market data. Each rule's performance is its excess return over Treasury bills while in the market; with no equity premium, no rule has skill.

## Options

```text
--source {goyal,sim}    real data (default) or simulated data
--data-file PATH        the Goyal-Welch spreadsheet, or a CSV export of its Monthly sheet
--start, --end          sample (default 1927-01 to the latest month)
--oos-start YYYY-MM     first forecast (default 1965-01)
--post-split YYYY-MM    Part D split (default 2005-01)
--gamma G               Part C risk aversion (default 3)
--n-boot N              Part E bootstrap samples (default 500)
```

## Outputs (`report/`)

- `lab06_report.md`: all results, with questions to think about
- `fig6_1_in_sample.png` to `fig6_5_reality_check.png`
- `table_*.csv` and `summary.json`

## Files

```text
ch06-out-of-sample/
├── README.md
├── requirements.txt
├── lab06.py                 # the lab: data, parts A–E, figures, report
├── lab06_walkthrough.ipynb  # guided, step-by-step version
├── fintest/                 # the book's analysis module (the parts this lab uses)
├── data/                    # save the Goyal-Welch file here
└── tests/test_lab06.py      # runs offline
```

## Code testing status

| Part | Status |
|---|---|
| Goyal–Welch reader and predictor definitions | Tested on a file in the published layout; **confirm on the live file at first run** |
| Walk-forward forecasts | Tested: no look-ahead; Campbell–Thompson forecasts never negative |
| Known truth | Tested: in simulated data the true predictor is found in and out of sample |
| Reality Check | Tested: does not reject when no rule has skill |
| Full lab and notebook | Run end to end on simulated data |

## Exercises

1. Start the forecasts in 1976 instead of 1965 (`--oos-start 1976-01`). Which conclusions change?
2. Which predictor has the largest in-sample t-statistic, and what is its out-of-sample R²?
3. In Part C, change the risk aversion to 5. Does any certainty-equivalent gain change sign?

## Data citation

Goyal, A., and I. Welch (2008), "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction," *Review of Financial Studies*; updated data from Amit Goyal's website.
