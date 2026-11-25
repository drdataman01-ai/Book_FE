# Lab 23: From Claim to Study — Pre-registration and Power

Companion lab for **Chapter 23** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

This chapter is about designing a study, not testing a claim. The lab gives you the tools to plan one: how much data a claim needs, what testing many ideas costs, how much freedom in research choices can manufacture a result, and a pre-registration document that fixes the plan before you see the data.

## Quick start

```bash
pip install -r requirements.txt
python lab23.py                                   # downloads Portfolios Formed on Size for Part C
python lab23.py --claim-file my_claim.json --expected-sharpe 0.4
```

The run takes under a minute. It writes `report/lab23_report.md`, five figures and `report/preregistration.md`.

If the download is blocked, save `Portfolios_Formed_on_ME_CSV.zip` from the [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) in `data/` and pass `--size-file`. The lab uses the value- and equal-weighted monthly returns of the tercile (`Lo 30`, `Hi 30`), quintile (`Lo 20`, `Hi 20`) and decile (`Lo 10`, `Hi 10`) portfolios; confirm these column names on the live file.

Or open `lab23_walkthrough.ipynb`, which runs the same steps with a prediction prompt before each one.

### Your own pre-registration

Write a JSON file with any of these fields and pass it with `--claim-file`; fields you leave out keep the size-effect example:

```json
{
  "title": "...", "authors": "...", "claim": "...", "hypotheses": "...", "data": "...", "sample": "...",
  "primary": "...", "secondary": "...", "power": "{power}", "decision": "...", "reporting": "..."
}
```

Leave `"power": "{power}"` to have the lab fill in the power analysis for `--expected-sharpe`. The document ends with a SHA-256 hash of its text; `fintest.design.verify_preregistration` checks that nothing was changed. Commit the file to a public repository, or post it on the Open Science Framework, **before** running the primary test, so the time stamp is independent of you.

### Testing the code without real data

```bash
python lab23.py --source sim
```

Only Part C uses data; with `--source sim` it uses simulated size portfolios with a January effect built in.

## What the lab does

| Part | Question | Method | Output |
|---|---|---|---|
| A. Power for returns | How many years are needed to detect a Sharpe ratio? | t ≈ SR × √years; 80% power at t = 1.96 and 3.0; simulation check with Student-t(4) shocks and AR(1) = 0.15 | Lab Figure 23.1 |
| B. Predictability and multiple testing | How many observations for an R²? What t is needed after many tests? | T ≈ (z_α + z_β)²(1 − R²)/R²; Bonferroni thresholds | Lab Figure 23.2 |
| C. Forking paths | How much do research choices matter? | Size effect under 36 combinations of cut, weighting, start date and January treatment (specification curve); a pure-noise signal tested in the same 36 ways, 300 times | Lab Figures 23.3–23.4 |
| D. Bayesianized p-values | How likely is the null after a given t? | Minimum Bayes factor exp(−t²/2) with prior odds from 1:1 to 99:1 against (Harvey 2017) | Lab Figure 23.5 |
| E. Pre-registration | What exactly will be tested? | A 10-section template filled in and hashed | `preregistration.md` |

## Verification status

| Part | Status |
|---|---|
| Power formulas | Tested against closed form and simulation |
| R² sample size, multiple-testing thresholds | Tested |
| Bayesianized p-values | Tested against Harvey's (2017) example (p = 0.05, odds 4:1 → about 37%) |
| Specification grid, forking-paths experiment | Tested: noise is significant about 5% of the time in one pre-chosen specification and far more often somewhere in the grid |
| Pre-registration hash | Tested: detects a one-word change |
| Full lab and notebook | Run end to end with simulated size portfolios |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab23.py`.

## Exercises

1. A colleague's strategy has a backtested Sharpe ratio of 0.6 over 12 years. What is the power of that test if the true Sharpe ratio is 0.3?
2. Add a fifth choice to the specification grid in Part C (for example, dropping the 1930s). How does the share of significant specifications change?
3. Increase the noise experiment's grid to 72 specifications. What share of studies now finds at least one significant result?
4. Write a pre-registration for the claim you plan to replicate in Chapter 23, including the power analysis.
5. For the size effect after 1982, what is the lowest probability that the null is true under prior odds of 4:1 against?

## Data citation

Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (Portfolios Formed on Size).
