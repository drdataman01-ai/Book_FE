# Lab 11: Do High-Interest Currencies Depreciate Enough to Offset Their Yield?

Companion lab for **Chapter 11** of *Financial Engineering in Practice*.

**What you will explore:** whether high-interest currencies fall enough to cancel their extra yield, as uncovered interest parity says they should.

## Quick start

```bash
pip install -r requirements.txt
python lab11.py                  # downloads about 20 FRED series
python lab11.py --cost 0.001     # carry net of 10 bp per unit of weight traded
```

If downloads are blocked, save the series below from FRED as CSV files named `<SERIES>.csv` in one folder and run `python lab11.py --data-dir data`. Missing currencies are skipped and listed in the report.

| Currency | Exchange rate (FRED) | Quote | 3-month rate (FRED, OECD) |
|---|---|---|---|
| Japanese yen | EXJPUS | yen per $ | IR3TIB01JPM156N |
| British pound | EXUSUK | $ per £ | IR3TIB01GBM156N |
| Swiss franc | EXSZUS | francs per $ | IR3TIB01CHM156N |
| Canadian dollar | EXCAUS | C$ per $ | IR3TIB01CAM156N |
| Australian dollar | EXUSAL | $ per A$ | IR3TIB01AUM156N |
| New Zealand dollar | EXUSNZ | $ per NZ$ | IR3TIB01NZM156N |
| Swedish krona | EXSDUS | kronor per $ | IR3TIB01SEM156N |
| Norwegian krone | EXNOUS | kroner per $ | IR3TIB01NOM156N |
| Euro (from 1999) | EXUSEU | $ per € | IR3TIB01EZM156N |
| U.S. | | | IR3TIB01USM156N |

Optional: `VIXCLS` (FRED) for Lab Figure 11.5, and the French library's Fama/French 3 Factors (`--ff3-file`) for stock-market exposure. **Confirm the OECD series identifiers on FRED at first run**; the lab reports any it could not load.

Or open `lab11_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab11.py --source sim
```

The simulation lets high-rate currencies depreciate by only 30% of their interest advantage and adds rare risk-off months in which they crash. Results on simulated data test the code only.

## What the lab does

All exchange rates are converted to dollars per unit of foreign currency. A U.S. investor who holds a foreign currency for a month earns the excess log return rx = (i* − i_US)/12 + Δs, the interest differential plus the change in the exchange rate. By covered interest parity the differential equals the forward discount, so these are the returns on a monthly forward contract.

| Part | Question | Method | Output |
|---|---|---|---|
| A. Fama regressions | Do high-rate currencies depreciate by the differential? | Δs on last month's (i_US − i*)/12 for each currency; UIP slope 1; Newey–West errors; pooled slope | Lab Figure 11.1 |
| B. Carry | Does borrowing low and lending high pay? | Long the three highest-rate currencies, short the three lowest, monthly; split into interest and exchange-rate parts; high, low and dollar portfolios | Lab Figures 11.2–11.3 |
| C. Stress tests | When does it lose? | Subperiods (to 1998, 1999–2007, 2008 on); costs of 0–20 bp per unit traded; beta to stocks and to VIX changes, including bad months only; the ten worst months | Table |
| D. Investor's view | What would holding it feel like? | Drawdowns, recovery time, share of losses in the worst 5% of months; carry against VIX changes | Lab Figures 11.4–11.5 |

### Design notes

- **Monthly averages.** FRED's H.10 and OECD series are monthly averages, not month-end values. This smooths returns slightly and understates the sharpness of crashes; daily data would show them more clearly.
- **Forward discount.** The lab uses interbank rate differentials instead of forward rates, which are not free. Covered interest parity held closely before 2008 and has shown small deviations since (Du, Tepper and Verdelhan 2018).
- **Costs.** Bid–ask spreads on G10 currencies are a few basis points; the lab's cost table shows the effect.

## Code testing status

| Part | Status |
|---|---|
| Excess-return arithmetic | Tested by hand |
| Fama regressions | Tested: recover a slope of 1 under UIP and −1 when it fails |
| Carry portfolio | Tested: weights, one-month signal lag, costs |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab11.py`.

## Exercises

1. Use two currencies on each side instead of three. Does the carry return rise or fall? Does the skewness change?
2. Exclude the euro and run the sample from 1980 to 1998. Compare the Fama slopes with those after 1999.
3. Add a momentum filter: hold the carry portfolio only when its past 3-month return is positive. Does it avoid the crashes?
4. Regress carry returns on the dollar portfolio and on stock-market returns together. How much of carry is explained?
5. Find the months of the 2008 unwind in your data. How many years of average carry returns did they erase?

## Data citation

Board of Governors of the Federal Reserve System (H.10 exchange rates) and OECD, Main Economic Indicators (three-month interbank rates), retrieved from FRED, Federal Reserve Bank of St. Louis.
