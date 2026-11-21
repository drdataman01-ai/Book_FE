# Lab 23: The Capstone Template

Companion lab for **Chapter 23** of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

The capstone is a replication of a published finance claim. This lab is the scaffolding: one command creates a project folder with everything a credible replication needs, and a worked example shows what a finished one looks like.

## Quick start

```bash
pip install -r requirements.txt

# a new, empty project
python lab23_capstone.py new --name size-replication --title "Banz (1981) revisited" \
    --claim "Small stocks earn higher returns than large stocks." --authors "Your Name"
cd size-replication
#   1. edit preregistration_fields.json, original.json, config.json, src/load.py, src/analysis.py
#   2. python run.py freeze        -> preregistration.md with a SHA-256 hash; commit it now
#   3. python run.py all           -> data, replication, extensions, report

# the worked example (momentum, French 10 portfolios formed on prior 12-2 returns)
python lab23_capstone.py demo                # downloads the data
python lab23_capstone.py demo --source sim   # simulated data, tests the code only
```

The demo writes `demo/momentum-replication/`. Its `original.json` holds **example values**, not the numbers of any paper; replace them with the statistics from the paper you replicate, citing table and page.

## What a project contains

| Path | Purpose |
|---|---|
| `preregistration_fields.json` → `preregistration.md` | The plan, frozen with a hash before the replication (`run.py freeze` refuses to overwrite it without `--force`) |
| `original.json` | The original paper's statistics to reproduce |
| `config.json` | Original sample, post-publication start, costs, tolerances, data source |
| `src/load.py`, `src/analysis.py` | Data loading and analysis; edit for your claim |
| `run.py` | The pipeline: `freeze`, `data`, `replicate`, `extend`, `report`, `all` |
| `results/` | `replication.json`, `comparison.csv` (status per statistic), `status.txt`, `extensions.csv`, `cumulative.png` |
| `logs/run_log.jsonl` | Append-only log: time, SHA-256 of every input, parameters, outputs, package versions |
| `report/report_template.md` → `report/report.md` | The report, with comparison and extension tables filled in and TODO sections for your writing |
| `review/` | Referee report and response-to-referees templates (Chapter 23, Section 23.6) |
| `tests/test_replication.py` | Checks that the pipeline runs and gives the same numbers twice |
| `LICENSE`, `CITATION.cff`, `.gitignore` | MIT licence for code, citation metadata, raw data excluded from version control by default |
| `fintest/` | A copy of the book's analysis module (version 1.0.0) |

**Replication status.** Each statistic in `original.json` is compared with the replication on the original sample: *match* (within the tolerance in `config.json`, default 15% relative), *same sign, different size*, *different sign* or *not reproduced*. The project is *reproduced* if all match, *not reproduced* if any sign differs or most are missing, and *partly reproduced* otherwise.

## Verification status

| Part | Status |
|---|---|
| Fingerprints, run log, comparison and status | Tested |
| Project creation | Tested: every file in the layout is created |
| Worked example | Run end to end on simulated data: pre-registration frozen and verified, five log entries, report generated, re-freezing refused, the project's own test passes |
| Real data | Not run from the build environment (network blocked) |

Run the tests with `python tests/test_lab23_capstone.py`.

## Milestones (see Chapter 23)

1. Proposal: the claim, the paper, the data and a power analysis (Lab 23).
2. Pre-registration frozen and committed.
3. Replication of the original sample, with differences explained.
4. Extensions: post-publication sample, costs, robustness grid.
5. Report and code release; peer review; response to referees; final verdict.

## Data citation

Kenneth R. French Data Library (10 Portfolios Formed on Prior (12-2) Return) for the worked example; your project's own sources otherwise.
