# Lab 19: Does Selling Options Earn a Reliable Premium?

Companion lab for **Chapter 19** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** implied volatility exceeds the volatility that follows, on average, so option sellers, for example covered-call and put-writing strategies, earn a reliable premium.

## Quick start

```bash
pip install -r requirements.txt
python lab19.py                  # downloads daily factors (French) and the VIX (FRED)
```

If the downloads are blocked, save the files in `data/` and pass them in:

| File | Source | Option |
|---|---|---|
| `F-F_Research_Data_Factors_daily_CSV.zip` | [French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): Fama/French 3 Factors [Daily] | `--ff3-daily-file` |
| `VIXCLS.csv` | [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | `--vix-file` |
| PUT index history (optional) | [Cboe indices](https://www.cboe.com/us/indices/): Cboe S&P 500 PutWrite Index | `--put-file` |
| BXM index history (optional) | Cboe indices: Cboe S&P 500 BuyWrite Index | `--bxm-file` |

The Cboe reader skips any preamble, finds the first dated row and uses the last numeric column; levels are turned into month-end returns.

Other options: `--call-strike 1.02` (strike of the rebuilt covered call, as a multiple of the index), `--iv-shift -2` (price the rebuilt options this many vol points below the VIX).

Or open `lab19_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab19.py --source sim
```

The simulation prices a VIX 12% above the volatility the model expects, so a premium is built in. Results on simulated data test the code only.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Variance risk premium | Does implied volatility exceed what follows? | Month-end VIX against realized volatility of the market over the next 21 trading days, from 1990; average gap with Newey–West t, share of months positive, worst months | Lab Figures 19.1–18.2 |
| B. Strategies | Do option sellers beat the market? | Rebuilt one-month strategies: at-the-money put-write (identical to an at-the-money covered call by put–call parity) and a 2% out-of-the-money covered call, priced by Black–Scholes at the VIX; optional Cboe PUT and BXM. Excess return, volatility, Sharpe, skewness, drawdown, beta, CAPM alpha | Lab Figure 19.3 |
| C. Stress tests | What happens in crashes, and does the Sharpe ratio flatter them? | Crisis periods; the five worst market months; up and down betas; the manipulation-proof performance measure (MPPM, relative risk aversion 3) | Lab Figure 19.4 |
| D. Equity versus volatility | How much is just beta? | Split each strategy's excess return into beta × market premium and alpha; compare with the market scaled to the same beta; a pure short variance swap struck at VIX², sized to 10% volatility | Lab Figure 19.5 |

### Design notes

- **The rebuilt strategies are an approximation.** They roll at calendar month ends (Cboe rolls on the third Friday), price options at the VIX rather than at the at-the-money implied volatility, ignore bid–ask spreads, and use the total-return CRSP market rather than the S&P 500. The VIX is computed from a strip of options that includes out-of-the-money puts, so it usually exceeds at-the-money implied volatility; pricing at the VIX therefore *overstates* the premium a seller collects. Try `--iv-shift -2`, and compare with Cboe's PUT and BXM, which use actual traded prices.
- **Realized volatility** is measured on the CRSP value-weighted market, a close proxy for the S&P 500, over the next 21 trading days (the VIX covers 30 calendar days).
- **The variance swap** payoff is (VIX/100)² × 21/252 minus the sum of squared daily returns over the next 21 trading days; VIX² approximates the variance swap rate.

## Verification status

| Part | Status |
|---|---|
| Black–Scholes prices, delta, vega, implied volatility | Tested against textbook values and numerical derivatives |
| Monthly option-writing returns | Tested against hand calculation; put–call parity checked |
| Variance swap payoff | Tested against direct calculation |
| MPPM, capture ratios | Tested (MPPM penalizes negative skew at equal mean and volatility) |
| Cboe reader | Tested on a file with a preamble and thousands separators; **confirm on the live files** |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab19.py`.

## Exercises

1. Rerun Part B with `--iv-shift -2` and `-4`. How much of the rebuilt strategies' alpha survives a realistic at-the-money volatility?
2. Download Cboe's PUT and BXM. How closely do the rebuilt strategies track them? In which months do they differ most, and why?
3. Compute the MPPM for risk aversion 1, 3 and 6. At what level of risk aversion does put-writing stop beating the market?
4. Split the sample at 2008. Is the variance risk premium smaller after the financial crisis?
5. Add a strategy that sells 5% out-of-the-money puts. Compare its skewness and alpha with the at-the-money version.

## Data citation

Kenneth R. French Data Library, Fama/French 3 Factors [Daily]. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis.
Cboe Global Markets, Cboe S&P 500 PutWrite Index (PUT) and BuyWrite Index (BXM) (optional).
