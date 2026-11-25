# Lab 16: Put the Factor Zoo on Trial

Companion lab for **Chapter 16** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

**Claim on trial:** most published return predictors are products of data mining.

## Quick start

1. Download from [Open Source Asset Pricing](https://www.openassetpricing.com) (Chen and Zimmermann 2022):
   - long-short returns for the original papers' portfolios, either the wide file (`PredictorLSretWide.csv`: a date column, then one column per signal) or the long format (`signalname, port, date, ret`, of which the lab keeps `port == "LS"`);
   - the signal documentation, `SignalDoc.csv` (columns `Acronym`, `Year`, `SampleStartYear`, `SampleEndYear`).

   The Python package `openassetpricing` downloads both (`dl_port("op", "pandas")` and `dl_signal_doc("pandas")`); save them as CSV in `data/`.
2. Run:

```bash
pip install -r requirements.txt
python lab16.py --returns data/PredictorLSretWide.csv --signal-doc data/SignalDoc.csv
```

Optional, Part C: a second construction of the same signals, for example value-weighted deciles or the version excluding stocks below the NYSE 20th size percentile:

```bash
python lab16.py --returns-alt data/ex_nyse_p20_me_LS.csv --label-alt "Micro-caps excluded"
```

Returns in percent are converted to decimals automatically.

### Testing the code without real data

```bash
python lab16.py --source sim
```

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Replication | How strong was the original evidence? | t-statistic of each long-short return over the paper's own sample; shares clearing t > 1.96, t > 3 and Benjamini–Hochberg at 2.5% | Lab Figure 16.1 |
| B. Decay | What happened afterward? | Returns after the sample ended (before publication) and after publication; pooled regression of returns scaled by each signal's in-sample mean on post-sample and post-publication indicators, month-clustered errors (McLean and Pontiff 2016) | Lab Figures 16.2–15.3 |
| C. Methodology | Does the construction change the answer? | The same statistics for a second construction (optional) | Table |
| D. Shrinkage | How much to believe each estimate? | Empirical Bayes: each in-sample mean shrunk toward the cross-sectional average by its noise; posterior probability of a positive true mean; accuracy for post-publication returns | Lab Figure 16.4 |
| E. Known truth | How much decay does selection alone create? | A simulated literature in which only signals with in-sample t > 2 are published, with and without true post-publication decay | Lab Figure 16.3 |

### Design notes

- **Periods.** In-sample: the paper's `SampleStartYear` to `SampleEndYear`. Post-sample: after the sample ended, up to and including the publication year. Post-publication: after the publication year. Years are calendar years, a simplification of McLean and Pontiff's month-level dates.
- **Decay regression.** Only signals with positive in-sample means enter, and each signal's returns are divided by its in-sample mean, so coefficients read as proportional changes.
- **Shrinkage** uses a normal prior estimated from the signals themselves. Because published signals were selected for strong in-sample results, this reduces noise but does not undo selection; Jensen, Kelly and Pedersen (2023) use a richer hierarchical model with factor themes and many countries.

## Verification status

| Part | Status |
|---|---|
| Readers (wide, long, SignalDoc; percent conversion) | Tested on files in the documented layouts; **confirm on the live files at first run** |
| Period statistics | Tested against direct calculation |
| Decay regression | Tested: recovers a known 50% post-publication decline |
| Shrinkage | Tested: noisier estimates are shrunk more |
| Simulation | Tested: selection alone produces apparent decay |
| Full lab and notebook | Run end to end on simulated data |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab16.py`.

## Exercises

1. Run Part C with the micro-caps-excluded construction. How many signals drop below t = 1.96?
2. Restrict the sample to signals published after 2000. Is post-publication decay larger or smaller?
3. Replace the normal prior in Part D with a mixture: a share of signals with zero mean and the rest normal. Estimate the share of zeros.
4. Group signals by category (for example valuation, momentum, profitability). Does decay differ by category?
5. In Part E, vary the publication threshold from t > 2 to t > 3. How does the apparent decay change?

## Data citation

Chen, A. Y., and Zimmermann, T. (2022). Open source cross-sectional asset pricing. *Critical Finance Review*, 11(2), 207–264. Data: https://www.openassetpricing.com.
