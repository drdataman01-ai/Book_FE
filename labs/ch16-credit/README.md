# Lab 16: Do Credit Spreads Compensate Mainly for Expected Default Losses?

Companion lab for **Chapter 16** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** the yield spread of corporate bonds over Treasuries mostly reflects the losses investors expect from default.

## Quick start

```bash
pip install -r requirements.txt
python lab16.py                  # downloads five FRED series
python lab16.py --pd-baa 0.005 --recovery 0.30    # your own default and recovery assumptions
```

If downloads are blocked, save each series from FRED as CSV in `data/` and pass it in:

| Series | FRED | Option |
|---|---|---|
| Moody's Seasoned Aaa Corporate Bond Yield (monthly, from 1919) | [AAA](https://fred.stlouisfed.org/series/AAA) | `--aaa-file` |
| Moody's Seasoned Baa Corporate Bond Yield (monthly, from 1919) | [BAA](https://fred.stlouisfed.org/series/BAA) | `--baa-file` |
| 10-Year Treasury Constant Maturity Rate (monthly, from 1953) | [GS10](https://fred.stlouisfed.org/series/GS10) | `--gs10-file` |
| NBER recession indicator | [USREC](https://fred.stlouisfed.org/series/USREC) | `--usrec-file` |
| Industrial Production Index | [INDPRO](https://fred.stlouisfed.org/series/INDPRO) | `--indpro-file` |
| Gilchrist–Zakrajšek excess bond premium (optional) | Federal Reserve Board (`ebp_csv.csv`, linked from its FEDS Notes on the excess bond premium) | `--ebp-file` |

Or open `lab16_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab16.py --source sim
```

The simulation builds a Baa–Aaa spread from a fixed expected loss plus a risk premium that jumps in recessions and forecasts weaker output. Results on simulated data test the code only.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Spreads | How large are spreads, and when? | Baa–Aaa (from 1919), Baa–Treasury and Aaa–Treasury (from 1953): averages, decades, recessions versus expansions | Lab Figure 16.1 |
| B. Decomposition | What share is expected loss? | Expected-loss spread −ln(1 − PD·(1 − R)) from adjustable annual default rates and recovery; residual; the default rate that would be needed to justify the whole spread | Lab Figures 16.2–16.3 |
| C. Risk premium | Does the residual behave like a premium? | Industrial production growth over the next 12 months on each spread (and on the excess bond premium if provided), Newey–West t; growth by spread quintile; spreads around recession starts | Lab Figure 16.4 |
| D. Stress | How sensitive is the answer? | Share explained over a grid of default rates (0.1% to 2%) and recoveries (20% to 60%); crisis peaks; subperiods | Lab Figure 16.5 |

### Default and recovery assumptions

The defaults come from Moody's *Corporate Default and Recovery Rates, 1920–2006* (2007): an average annual issuer-weighted default rate of 0.272% for Baa and 0.000% for Aaa over 1920–2006, and an average recovery of 38% for senior unsecured bonds over 1982–2006 (prices 30 days after default). Change them with `--pd-baa`, `--pd-aaa` and `--recovery`; Part D shows a whole grid.

### Design notes

- **Maturity mismatch.** Moody's seasoned yields are on long-maturity bonds (typically 20 years or more), while GS10 is a 10-year yield. When the Treasury curve slopes up, this alone widens the measured Baa–Treasury spread; the Baa–Aaa spread avoids the problem but understates the full credit spread.
- **Taxes.** Interest on corporate bonds is subject to state income tax; Treasury interest is not. Elton, Gruber, Agrawal and Mann (2001) found that this explains a large part of corporate–Treasury spreads.
- **One-period approximation.** The expected-loss spread treats the annual default rate as constant over the bond's life. Cumulative default rates rise faster than linearly as ratings migrate, so long-horizon Baa losses can be somewhat higher per year than the annual average suggests; Part D's grid covers that range.

## Verification status

| Part | Status |
|---|---|
| Expected-loss and break-even formulas | Tested, including risk-neutral pricing identity |
| Excess bond premium reader | Tested on a file in the documented layout; **confirm on the live file** |
| Decomposition | Tested on data with a known split |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab16.py`.

## Exercises

1. Use the 1970–2006 Baa cumulative 10-year default rate (4.6%, Moody's) instead of the annual average. What annual rate does it imply, and how much of the spread does it explain?
2. Find the default and recovery assumptions under which expected loss explains 75% of the Baa–Treasury spread. Are they plausible?
3. Add the excess bond premium. Does it forecast industrial production better than the raw Baa–Aaa spread?
4. Compare the spread's forecasting power before and after 1985.
5. Compute the Baa–Aaa spread at each NBER recession peak. Is the spread already high before recessions start, or does it rise only afterwards?

## Data citation

Moody's, Moody's Seasoned Aaa and Baa Corporate Bond Yields [AAA, BAA]; Board of Governors of the Federal Reserve System, 10-Year Treasury Constant Maturity Rate [GS10] and Industrial Production [INDPRO]; NBER-based recession indicator [USREC]; all retrieved from FRED, Federal Reserve Bank of St. Louis.
Gilchrist, S., and Zakrajšek, E. (2012). Credit spreads and business cycle fluctuations. *American Economic Review*, 102(4), 1692–1720 (excess bond premium, optional).
Moody's Investors Service (2007). *Corporate Default and Recovery Rates, 1920–2006*.
