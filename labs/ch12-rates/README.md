# Lab 12: Do Forward Rates Predict Future Interest Rates?

Companion lab for **Chapter 12** of *Financial Engineering in Practice*.

**What you will explore:** whether forward rates are unbiased forecasts of future short rates, as the expectations hypothesis says they should be.

## Quick start

```bash
pip install -r requirements.txt
python lab12.py                  # downloads the Federal Reserve's zero-coupon yield curve
```

If the download is blocked, save `feds200628.csv` from the Federal Reserve's page [The U.S. Treasury Yield Curve: 1961 to the Present](https://www.federalreserve.gov/data/nominal-yield-curve.htm) in `data/` and run:

```bash
python lab12.py --gsw-file data/feds200628.csv
```

The reader skips the file's preamble, finds the header row that starts with `Date`, uses the zero-coupon yield columns `SVENY01` to `SVENY10` (continuously compounded, percent) and takes month-end values.

Option: `--oos-start 1985-01` (first out-of-sample forecast).

Or open `lab12_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab12.py --source sim
```

The simulation builds yields from an AR(1) short rate plus a time-varying term premium, so the expectations hypothesis is false by construction. Results on simulated data test the code only.

## What the lab does

Notation follows Cochrane and Piazzesi (2005): y(n) is the n-year zero yield, p(n) = −n·y(n) the log price, f(n) = p(n−1) − p(n) the forward rate, and rx(n) = p(n−1) one year later − p(n) − y(1) the one-year excess log return on an n-year bond. Monthly observations of annual returns overlap, so all standard errors are Newey–West with 18 lags.

| Part | Question | Method | Output |
|---|---|---|---|
| A. Classic tests | Are forward rates unbiased forecasts? | Fama–Bliss: rx(n) on f(n) − y(1) (hypothesis: slope 0); change in the one-year rate over n−1 years on f(n) − y(1) (slope 1); Campbell–Shiller: y(n−1) a year later − y(n) on (y(n) − y(1))/(n−1) (slope 1), n = 2 to 10 | Lab Figures 12.2–12.3 |
| B. Cochrane–Piazzesi | How much of bond excess returns is forecastable? | Average rx(2..5) on y(1), f(2)..f(5); loadings of each maturity on the factor; out-of-sample R² from 1985 with an 11-month gap, Clark–West test | Lab Figure 12.4 |
| C. Subperiods | Does it hold over time? | Start–1984, 1985–2007, 2008–2015 (zero lower bound), 2016 on | Table |
| D. Investor's view | What could a bond investor use? | Average excess return, volatility and Sharpe ratio by maturity; excess returns after a steep versus flat curve (5-year forward spread above or below its past median) | Lab Figure 12.5 |

### Design notes

- **Zero-coupon yields** are fitted curves, not traded bonds; fitting error adds noise, especially at short maturities and in the early years. Cochrane and Piazzesi used the Fama–Bliss discount bond data from CRSP, which are not free; results with the Fed curve are similar in spirit but not identical.
- **Ten-year yields** start in 1971 in the Fed file, so regressions at maturities above 7 years use a shorter sample.
- **Out of sample**, the 11-month gap ensures no training target overlaps the forecast date (Chapter 6).

## Code testing status

| Part | Status |
|---|---|
| Fed file reader | Tested on a file in the documented layout (preamble, SVENY columns, NA rows); **confirm on the live file** |
| Forwards and excess returns | Tested against hand calculation |
| Regressions | Tested: recover slopes of 0 and 1 where the expectations hypothesis holds exactly, and reject it when a time-varying premium is added |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab12.py`.

## Exercises

1. Add the three principal components (level, slope, curvature) of yields as predictors. Does the Cochrane–Piazzesi factor add to them? (See Bauer and Hamilton 2018 on small-sample tests.)
2. Use maturities 1 to 10 in the Cochrane–Piazzesi regression. Is the tent shape stable?
3. Run Part B with a rolling 20-year window instead of an expanding one.
4. Build a strategy that holds the 5-year bond when the forecast excess return is positive and the 1-year bond otherwise. Compare its Sharpe ratio with always holding the 5-year bond.
5. Compare the 2008–2015 results with the rest of the sample. Why might regressions behave differently when the short rate is pinned near zero?

## Data citation

Gürkaynak, R. S., Sack, B., and Wright, J. H. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics*, 54(8), 2291–2304. Data: Board of Governors of the Federal Reserve System, https://www.federalreserve.gov/data/nominal-yield-curve.htm
