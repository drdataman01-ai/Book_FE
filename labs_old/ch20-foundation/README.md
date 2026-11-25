# Lab 20: Do Foundation Models Beat Classical Forecasts?

Companion lab for **Chapter 20** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** pretrained time-series foundation models, used zero-shot, forecast financial and economic series better than classical models.

## Quick start

```bash
pip install -r requirements.txt
pip install chronos-forecasting timesfm            # optional, large downloads (PyTorch and model weights)
python lab20.py --models chronos-bolt-small chronos-bolt-base timesfm
```

Without the optional packages the lab still runs the classical models and the contamination demonstration, and it lists which foundation models could not be loaded. With them, a CPU run takes a few minutes per model; the first run downloads the weights from Hugging Face.

If downloads of the data are blocked, save the files in `data/` and pass them in:

| Data | Source | Option |
|---|---|---|
| Monthly market excess return (from 1926) | French library: `F-F_Research_Data_Factors_CSV.zip` | `--ff3-file` |
| Daily market return, for monthly realized variance (from 1926) | French library: `F-F_Research_Data_Factors_daily_CSV.zip` | `--ff3d-file` |
| Consumer Price Index | FRED [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) | `--cpi-file` |
| Industrial Production Index | FRED [INDPRO](https://fred.stlouisfed.org/series/INDPRO) | `--indpro-file` |
| Unemployment rate | FRED [UNRATE](https://fred.stlouisfed.org/series/UNRATE) | `--unrate-file` |
| 10-year Treasury yield | FRED [GS10](https://fred.stlouisfed.org/series/GS10) | `--gs10-file` |

Options: `--eval-start 2000-01` (first forecast), `--cutoff 2024-01` (see Part C), `--context 512` (months of history given to each foundation model).

Or open `lab20_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Testing the code without real data

```bash
python lab20.py --source sim --models standin
```

The simulation builds six series with known dynamics (independent returns, long-memory log variance, AR inflation and output growth, near-white-noise changes in rates). `standin` is a **test double, not a foundation model**: it averages the naive, AR and HAR-type forecasts so the pipeline has something to run. Results on simulated data test the code only.

## What the lab does

Every month from `--eval-start`, each model forecasts the next month's value of six series using only data up to the current month.

| Series | Built from |
|---|---|
| Market excess return | French Mkt-RF |
| Log realized variance of the market | log of the monthly sum of squared daily market returns |
| Inflation | monthly percentage change in CPIAUCSL |
| Industrial production growth | monthly percentage change in INDPRO |
| Unemployment rate change | monthly change in UNRATE |
| 10-year yield change | monthly change in GS10 |

| Model | What it is |
|---|---|
| Historical mean | Average of all past values |
| Naive (last value) | This month's value |
| AR (BIC) | Autoregression on the last 30 years, 1 to 12 lags chosen by BIC: **the benchmark** |
| HAR-type (1, 3, 12) | Regression on the last value and its 3- and 12-month averages |
| GARCH(1,1) | Realized-variance series only: GARCH on monthly returns, mapped to log variance |
| Chronos-Bolt (small, base) | Amazon's pretrained model, zero-shot, median as point forecast, 10% and 90% quantiles |
| TimesFM | Google's pretrained model, zero-shot |

| Part | Question | Method | Output |
|---|---|---|---|
| A. Accuracy | Who forecasts best? | MSE relative to AR, R² against the historical mean | Lab Figure 20.1, Lab Figure 20.5 |
| B. Significance | Is any gain more than luck? | Diebold–Mariano t against AR; share of months with the smaller error; cumulative squared-error difference | Lab Figure 20.2 |
| C. Contamination | Could the model have seen the test data? | Accuracy before and after `--cutoff`; a deliberately contaminated "memorizing" model shows what leakage looks like | Lab Figure 20.3 |
| D. Uncertainty | Are intervals honest? | Share of outcomes inside each model's 80% interval | Lab Figure 20.4 |

### Design notes

- **Contamination.** Foundation models are pretrained on large public collections of time series. Public economic and financial series, or close relatives, may be among them, so a model can look good on history simply because it has seen it. The only clean test is on data published after the model's training cutoff. Set `--cutoff` to the model's documented training-data cutoff (or its release date, if that is all that is known); the default, January 2024, is a placeholder. The post-cutoff window is short, so its results are noisy.
- **The memorizing model** is given 60% of the true value before the cutoff and nothing after it. It is there to show the signature of contamination, a large gap between the two windows. It is not a model anyone should use.
- **GARCH** sees one monthly return per month, while the realized-variance models see a variance built from daily data, so it is handicapped on this task; it is included because it is the classical volatility model.
- **Adapters.** The Chronos adapter is written for the `BaseChronosPipeline.predict_quantiles` interface of `chronos-forecasting`; the TimesFM adapter tries the 2.5 interface and then the 1.x/2.0 interface. Both packages change between releases. If an adapter fails, the lab says so and continues; fixing it usually means following the package's current quick-start.

## Verification status

| Part | Status |
|---|---|
| Rolling-origin evaluation | Tested: the forecast for month t+1 does not change when data from t+1 on change |
| AR and HAR-type forecasters | Tested: recover a known AR(1) forecast and error size |
| Scores and cutoff split | Tested: perfect forecast, ranking against the naive model, interval coverage, memorizer signature |
| Missing packages | Tested: reported, not fatal |
| Foundation-model adapters | **Not tested**: PyTorch, `chronos-forecasting` and `timesfm` could not be installed in the build environment |
| Full lab and notebook | Run end to end on simulated data with the stand-in |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab20.py`.

## Exercises

1. Run Chronos-Bolt small and base. Does the larger model forecast better on any series?
2. Change `--context` from 512 to 64 months. Which models suffer most from a shorter history?
3. Set `--cutoff` to each model's release date. Do the foundation models' MSE ratios rise after the cutoff?
4. Add a 12-month horizon (forecast the sum of the next 12 months). Do the foundation models do relatively better at longer horizons?
5. Combine the best foundation model with AR by averaging their forecasts. Does the combination beat both?

## Data citation

Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (Fama/French 3 Factors, monthly and daily).
U.S. Bureau of Labor Statistics, Consumer Price Index for All Urban Consumers [CPIAUCSL] and Unemployment Rate [UNRATE]; Board of Governors of the Federal Reserve System, Industrial Production Index [INDPRO] and 10-Year Treasury Constant Maturity Rate [GS10]; all retrieved from FRED, Federal Reserve Bank of St. Louis.
