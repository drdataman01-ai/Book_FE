# Lab 18: Does Delta Hedging Remove Option Risk?

Companion lab for **Chapter 18** of *Financial Engineering in Practice*.

**What you will explore:** how much risk is left for an option seller who delta-hedges as Black–Scholes prescribes.

## Quick start

```bash
pip install -r requirements.txt
python lab18.py                  # downloads daily factors (French) and the VIX (FRED); about a minute
```

If the downloads are blocked, save `F-F_Research_Data_Factors_daily_CSV.zip` from the [French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and `VIXCLS.csv` from [FRED](https://fred.stlouisfed.org/series/VIXCLS) in `data/`:

```bash
python lab18.py --ff3-daily-file data/F-F_Research_Data_Factors_daily_CSV.zip --vix-file data/VIXCLS.csv
```

Other options: `--iv-shift -2` (price and hedge at the VIX minus 2 points, closer to at-the-money implied volatility), `--n-sims 20` (simulated Black–Scholes paths per month).

Or open `lab18_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab18.py --source sim
```

The simulation draws GJR-GARCH returns with fat-tailed shocks and prices a VIX 12% above expected volatility. Results on simulated data test the code only.

## What the lab does

At each month end since 1990 the lab sells a one-month at-the-money call on the market, priced by Black–Scholes with the VIX as the implied volatility. It buys delta units of the index, finances them at the risk-free rate, rebalances on a schedule, and pays the option's payoff at expiry. The final value of this position is the **hedging P&L**, shown as a share of the premium. The same exercise is repeated on simulated geometric Brownian motion paths with the month's implied volatility, where Black–Scholes holds exactly.

| Part | Question | Method | Output |
|---|---|---|---|
| A. Residual risk | How much risk is left after hedging? | Mean, standard deviation, skewness, 1% quantile and worst month of hedged P&L, daily to never hedged, on real and Black–Scholes paths; share of the unhedged risk removed | Lab Figure 18.1 |
| B. Source | Where does the error come from? | Regression of the P&L on the gamma–theta approximation ½ΓS²(σ²Δt − r²) summed over the month, and on the gap between implied and realized variance | Lab Figure 18.3 |
| C. Frictions | Do hedging frequency and costs change it? | Every 1, 2, 5 and 10 days; costs of 0, 2, 5 and 10 basis points per unit of index traded | Lab Figures 18.4–17.5 |
| D. Tail | When are the losses? | The worst months with the market move, implied and realized volatility and the largest daily move; share of all losses from the worst 5% of months | Lab Figure 18.2 |

### Design notes

- **One volatility for pricing and hedging.** The option is priced and hedged at the VIX. Hedging at a different volatility (for example, realized) changes the shape of the P&L; Exercise 3 explores this.
- **Calls only.** A delta-hedged call and a delta-hedged put with the same strike are the same position (put–call parity), so their hedged P&Ls are identical; the tests check this.
- **The underlying** is the CRSP value-weighted total-return index, a proxy for the S&P 500, with no dividend yield in the model (consistent with a total-return index). The VIX includes a skew premium, so pricing at the VIX overstates at-the-money implied volatility; this shifts the *mean* P&L up but has little effect on its spread.
- **Black–Scholes benchmark.** With daily hedging, the standard deviation of the hedging error is about √(π/4)·σ·vega/√N (Kamal and Derman 1999), about a fifth of the premium for a one-month at-the-money option with 21 rebalances.

## Code testing status

| Part | Status |
|---|---|
| Hedging engine | Tested on Black–Scholes paths: mean near zero, error in line with the Kamal–Derman formula, about halving with four times as many rebalances |
| Unhedged payoff, put–call equivalence, costs | Tested |
| Gamma–theta approximation | Tested: tracks the actual P&L, negative when realized volatility exceeds implied |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab18.py`.

## Exercises

1. Rerun with `--iv-shift -2`. Which statistics change, and which do not?
2. Hedge every 10 minutes? Not possible with daily data, but you can halve the time step on simulated paths. How much does the Black–Scholes error fall, and why can the real-path error not fall the same way?
3. Price the option at the VIX but hedge with the trailing 21-day realized volatility. Compare the distribution of P&L.
4. Repeat Part A for options 5% out of the money. Is the residual risk larger or smaller relative to the premium?
5. Remove the ten months with the largest daily moves. How much of the residual risk remains?

## Data citation

Kenneth R. French Data Library, Fama/French 3 Factors [Daily]. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS], retrieved from FRED, Federal Reserve Bank of St. Louis.
