"""Lab 23: The Capstone Template - a ready-made replication project.

Chapter 23 of *Financial Engineering and the Evidence: Active Learning by Verifying Famous Claims*.

WHAT THE LAB DOES
  `new`   creates a replication project folder with a fixed structure:
          pre-registration, the original paper's statistics to reproduce,
          a one-command pipeline (freeze, data, replicate, extend, report),
          an append-only run log with data fingerprints, report and referee
          templates, tests, licence and citation files.
  `demo`  creates a project for a worked example (momentum: winners minus
          losers from the Kenneth R. French Data Library) and runs every
          step, so you can see what a finished capstone looks like.

USAGE
  python lab23_capstone.py new --name size-effect-replication --title "Banz (1981) revisited"
  python lab23_capstone.py demo                     # downloads French momentum portfolios
  python lab23_capstone.py demo --source sim        # simulated data, tests the code only
  cd <project> && python run.py all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Templates written into a new project
# ---------------------------------------------------------------------------

README = """# {title}

A replication of a published finance claim, built from the capstone template in Chapter 23 of
*Financial Engineering and the Evidence*.

**Claim:** {claim}

**Status:** see `results/status.txt` after running the pipeline. Verdict and date: see `report/report.md`.

## How to reproduce everything

```bash
pip install -r requirements.txt
python run.py all            # freeze -> data -> replicate -> extend -> report
python -m pytest tests       # or: python tests/test_replication.py
```

Steps can be run one at a time: `python run.py freeze`, `data`, `replicate`, `extend`, `report`.
Add `--source sim` to any step to test the code on simulated data.

## Layout

| Path | Contents |
|---|---|
| `preregistration.md` | The plan, written and hashed before the replication (`run.py freeze`) |
| `preregistration_fields.json` | The fields the plan is built from; edit before freezing |
| `original.json` | Statistics reported in the original paper, with table and page references |
| `config.json` | Samples, data sources and tolerances |
| `src/` | Data loading and analysis code |
| `data/raw/`, `data/processed/` | Downloaded and cleaned data (raw data are not committed if licensing forbids it) |
| `results/` | Tables, figures, comparison with the original, status |
| `logs/run_log.jsonl` | One record per run: time, input fingerprints, parameters, outputs, environment |
| `report/` | Report template and the generated report |
| `review/` | Referee report and response templates |
| `tests/` | Checks that the pipeline runs and reproduces its own saved results |

## Data

{data}

## Licence and citation

Code: MIT licence (`LICENSE`). Cite this replication with `CITATION.cff`. Data remain under their providers' terms.
"""

PREREG_FIELDS = {
    "title": "{title}",
    "authors": "{authors}",
    "claim": "{claim}",
    "hypotheses": "H0: ... H1: ... (state the direction).",
    "data": "Sources, versions and download dates.",
    "sample": "Original sample period; post-publication period; exclusions.",
    "primary": "The single test that decides the replication, with standard errors and threshold.",
    "secondary": "Extensions and the robustness grid.",
    "power": "Expected effect size, sample length and power (use fintest.design.years_for_sharpe).",
    "decision": "Reproduced if every statistic in original.json matches within the tolerances in config.json; "
                "partly reproduced if signs agree but sizes differ; not reproduced otherwise. The claim survives if "
                "the post-publication primary test is significant with the original sign.",
    "reporting": "All statistics in original.json, all extensions, the run log and any deviations.",
}

ORIGINAL_EXAMPLE = {
    "_note": "Replace these EXAMPLE values with the numbers in the original paper and cite table and page for each.",
    "_source": "Paper, table, page",
    "mean_monthly_return": 0.0100,
    "t_statistic": 3.0,
    "annualized_sharpe": 0.60,
}

CONFIG = {
    "original_sample": ["1965-01", "1989-12"],
    "post_publication_start": "1993-01",
    "cost_per_turnover_bp": [0, 25, 50],
    "assumed_monthly_turnover": 0.8,
    "tolerance": {"rel": 0.15, "abs": 0.0},
    "data_source": "french",
}

LOAD_PY = '''"""Load the data for the replication. Edit this file for your own claim."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fintest.data import find_table, load_french_tables


def load(source: str = "french", path=None) -> pd.Series:
    """Monthly winners-minus-losers return: top minus bottom decile of the French portfolios formed on prior
    (12-2) returns, value-weighted. With source='sim', a simulated series (testing only)."""
    if source == "sim":
        rng = np.random.default_rng(0)
        idx = pd.period_range("1927-01", periods=1188, freq="M")
        x = 0.009 + 0.05 * rng.standard_t(4, len(idx)) / np.sqrt(2)
        return pd.Series(x, index=idx, name="wml")
    t = load_french_tables("mom_portfolios_10", path=path)
    vw = find_table(t, "value", "weight")
    cols = [c.strip() for c in vw.columns]
    vw.columns = cols
    hi = next(c for c in cols if c.lower().startswith("hi"))
    lo = next(c for c in cols if c.lower().startswith("lo"))
    return (vw[hi] - vw[lo]).dropna().rename("wml")
'''

ANALYSIS_PY = '''"""Analysis for the replication. Edit for your own claim; keep functions pure (data in, numbers out)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def nw_t(x: pd.Series, lags: int = 6) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n))


def summary(x: pd.Series) -> dict:
    return {"mean_monthly_return": float(x.mean()), "t_statistic": nw_t(x),
            "annualized_sharpe": float(x.mean() / x.std(ddof=1) * math.sqrt(12)), "months": int(x.notna().sum()),
            "start": str(x.dropna().index[0]), "end": str(x.dropna().index[-1])}


def extensions(x: pd.Series, cfg: dict) -> pd.DataFrame:
    rows = []
    a, b = cfg["original_sample"]
    windows = {"original sample": (a, b), "post-publication": (cfg["post_publication_start"], None),
               "full sample": (None, None)}
    for label, (s, e) in windows.items():
        y = x.loc[pd.Period(s, "M") if s else None: pd.Period(e, "M") if e else None]
        for c in cfg["cost_per_turnover_bp"]:
            net = y - c / 1e4 * cfg["assumed_monthly_turnover"]
            rows.append({"window": label, "cost_bp": c, **summary(net)})
    return pd.DataFrame(rows)
'''

RUN_PY = '''"""One-command pipeline: python run.py all   (steps: freeze, data, replicate, extend, report)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fintest.design import preregistration, verify_preregistration  # noqa: E402
from fintest.replicate import RunLog, compare_to_original, replication_status  # noqa: E402
from src.analysis import extensions, summary  # noqa: E402
from src.load import load  # noqa: E402

LOG = RunLog(HERE / "logs" / "run_log.jsonl")
CFG = json.loads((HERE / "config.json").read_text())


def freeze(args):
    p = HERE / "preregistration.md"
    if p.exists() and not args.force:
        ok = verify_preregistration(p.read_text())
        print(f"preregistration.md already frozen (hash {'valid' if ok else 'INVALID: the file was edited'}); "
              "use --force to rewrite, and record the deviation in the report.")
        return
    fields = json.loads((HERE / "preregistration_fields.json").read_text())
    p.write_text(preregistration(fields))
    LOG.record("freeze", inputs=[HERE / "preregistration_fields.json", p], outputs={"file": "preregistration.md"})
    print("Wrote preregistration.md. Commit it (or post its hash) before running the replication.")


def data(args):
    x = load(args.source)
    out = HERE / "data" / "processed" / "series.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    x.to_frame().to_csv(out)
    raw = sorted((HERE / "data" / "raw").glob("*.*")) + sorted((HERE / "data").glob("*.zip"))   # downloads land in data/
    LOG.record("data", inputs=[out, *raw], params={"source": args.source}, outputs={"rows": len(x)})
    print(f"Saved {len(x)} observations to {out.relative_to(HERE)}")


def _series():
    df = pd.read_csv(HERE / "data" / "processed" / "series.csv", index_col=0)
    s = df.iloc[:, 0]
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def replicate(args):
    x = _series()
    a, b = CFG["original_sample"]
    rep = summary(x.loc[pd.Period(a, "M"):pd.Period(b, "M")])
    orig = {k: v for k, v in json.loads((HERE / "original.json").read_text()).items() if not k.startswith("_")}
    tab = compare_to_original(orig, rep, CFG["tolerance"]["rel"], CFG["tolerance"]["abs"])
    status = replication_status(tab)
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "replication.json").write_text(json.dumps(rep, indent=2))
    tab.to_csv(res / "comparison.csv", index=False)
    (res / "status.txt").write_text(status + "\\n")
    LOG.record("replicate", inputs=[HERE / "data" / "processed" / "series.csv", HERE / "original.json"],
               params={"sample": CFG["original_sample"], "tolerance": CFG["tolerance"]},
               outputs={"status": status, **rep})
    print(tab.to_string(index=False))
    print("Status:", status)


def extend(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = _series()
    ext = extensions(x, CFG)
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    ext.to_csv(res / "extensions.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x.index.to_timestamp(), np.cumsum(np.log1p(x)), color="#2b6cb0")
    for d in CFG["original_sample"] + [CFG["post_publication_start"]]:
        ax.axvline(pd.Period(d, "M").to_timestamp(), color="#888", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Cumulative log return")
    ax.set_title("The claim over time (dashed: original sample and publication)")
    fig.tight_layout()
    fig.savefig(res / "cumulative.png", dpi=150)
    plt.close(fig)
    LOG.record("extend", inputs=[HERE / "data" / "processed" / "series.csv"], params=CFG,
               outputs={"rows": len(ext)})
    print(ext[["window", "cost_bp", "mean_monthly_return", "t_statistic", "annualized_sharpe"]].round(4).to_string(index=False))


def _md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\\n".join(out)


def report(args):
    res = HERE / "results"
    tmpl = (HERE / "report" / "report_template.md").read_text()
    comp = pd.read_csv(res / "comparison.csv")
    ext = pd.read_csv(res / "extensions.csv")[["window", "cost_bp", "mean_monthly_return", "t_statistic",
                                                "annualized_sharpe", "months"]]
    prereg_ok = verify_preregistration((HERE / "preregistration.md").read_text()) \\
        if (HERE / "preregistration.md").exists() else False
    text = tmpl.replace("{{COMPARISON}}", _md(comp)).replace("{{EXTENSIONS}}", _md(ext)) \\
        .replace("{{STATUS}}", (res / "status.txt").read_text().strip()) \\
        .replace("{{DATE}}", date.today().isoformat()) \\
        .replace("{{PREREG}}", "valid" if prereg_ok else "missing or edited") \\
        .replace("{{RUNS}}", str(len(LOG.read())))
    (HERE / "report" / "report.md").write_text(text)
    LOG.record("report", inputs=[res / "comparison.csv", res / "extensions.csv"], outputs={"file": "report/report.md"})
    print("Wrote report/report.md; edit the sections marked TODO.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["freeze", "data", "replicate", "extend", "report", "all"])
    ap.add_argument("--source", choices=["french", "sim"], default=CFG.get("data_source", "french"))
    ap.add_argument("--force", action="store_true", help="rewrite a frozen pre-registration")
    args = ap.parse_args(argv)
    steps = {"freeze": freeze, "data": data, "replicate": replicate, "extend": extend, "report": report}
    for name in (["freeze", "data", "replicate", "extend", "report"] if args.step == "all" else [args.step]):
        steps[name](args)


if __name__ == "__main__":
    main()
'''

REPORT_TEMPLATE = """# Replication report: {title}

Date: {{{{DATE}}}} · Pre-registration hash: {{{{PREREG}}}} · Runs logged: {{{{RUNS}}}}

## 1. The claim
{claim}

TODO: one paragraph on the original paper: authors, year, journal, sample, method and headline result.

## 2. Method
TODO: data sources and versions; how the original variables were rebuilt; any unavoidable differences from the
original (for example, free data instead of CRSP); standard errors.

## 3. Replication
Statistics from the original paper compared with this replication on the original sample
(status: **{{{{STATUS}}}}**).

{{{{COMPARISON}}}}

TODO: explain any differences. Which choices drive them?

## 4. Extensions
Original sample, post-publication period and full sample, before and after trading costs.

{{{{EXTENSIONS}}}}

![Cumulative return](../results/cumulative.png)

TODO: robustness grid or specification curve (Chapter 23); other markets; other periods.

## 5. Deviations from the pre-registration
TODO: list every deviation and why. Write "None" if there were none.

## 6. Verdict (dated)
TODO: survives / weakens / fails / provisional, as of {{{{DATE}}}}, in one paragraph, with what new evidence would
change it.

## 7. Code and data
All code is in this repository; `python run.py all` reproduces every number. Input fingerprints are in
`logs/run_log.jsonl`.
"""

REFEREE_TEMPLATE = """# Referee report

Paper: ...
Date: ...

## Summary (3-5 sentences)
What the paper claims, how it tests the claim and what it finds, in neutral terms.

## Overall assessment
Is the claim important, is the design credible, and do the conclusions follow from the evidence?

## Major comments (numbered; each with the problem, why it matters and what would fix it)
1. ...

## Minor comments
1. ...

## Checklist (Chapter 23)
- [ ] Claim falsifiable and primary test stated
- [ ] Sample and benchmarks justified
- [ ] Robust standard errors; multiple testing addressed
- [ ] Power or confidence intervals reported
- [ ] Robustness grid shown
- [ ] Costs and look-ahead addressed
- [ ] Code and data available
"""

RESPONSE_TEMPLATE = """# Response to referees

We thank the referee(s) for their comments. Each comment is reproduced below in italics, followed by our response
and the location of any change in the revised report.

## Referee 1

*Comment 1.1: ...*

**Response.** ... (Change: Section X, Table Y.)
"""

TEST_PY = '''"""Checks that the pipeline runs end to end and reproduces its own saved results."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def test_pipeline_is_deterministic():
    subprocess.run([sys.executable, str(HERE / "run.py"), "data", "--source", "sim"], check=True, cwd=HERE)
    subprocess.run([sys.executable, str(HERE / "run.py"), "replicate"], check=True, cwd=HERE)
    first = json.loads((HERE / "results" / "replication.json").read_text())
    subprocess.run([sys.executable, str(HERE / "run.py"), "replicate"], check=True, cwd=HERE)
    second = json.loads((HERE / "results" / "replication.json").read_text())
    assert first == second


if __name__ == "__main__":
    test_pipeline_is_deterministic()
    print("ok")
'''

GITIGNORE = """__pycache__/
*.pyc
.ipynb_checkpoints/
# Raw data from licensed sources should not be committed; remove this line if the licence allows it
data/raw/*
!data/raw/.gitkeep
"""

LICENSE = """MIT License

Copyright (c) {year} {authors}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

CITATION = """cff-version: 1.2.0
message: "If you use this replication, please cite it as below."
title: "{title}"
authors:
  - name: "{authors}"
date-released: "{today}"
license: MIT
"""


def new_project(dest: Path, title: str, claim: str, authors: str, data_note: str, overwrite: bool = False) -> Path:
    from datetime import date

    dest = Path(dest).resolve()
    if dest.exists():
        if not overwrite:
            raise SystemExit(f"{dest} exists; choose another --name or pass --overwrite")
        shutil.rmtree(dest)
    for sub in ("src", "data/raw", "data/processed", "results", "logs", "report", "review", "tests"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    (dest / "data" / "raw" / ".gitkeep").write_text("")
    fmt = {"title": title, "claim": claim, "authors": authors, "data": data_note}
    (dest / "README.md").write_text(README.format(**fmt))
    fields = {k: v.format(**fmt) for k, v in PREREG_FIELDS.items()}
    (dest / "preregistration_fields.json").write_text(json.dumps(fields, indent=2))
    (dest / "original.json").write_text(json.dumps(ORIGINAL_EXAMPLE, indent=2))
    (dest / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (dest / "src" / "__init__.py").write_text("")
    (dest / "src" / "load.py").write_text(LOAD_PY)
    (dest / "src" / "analysis.py").write_text(ANALYSIS_PY)
    (dest / "run.py").write_text(RUN_PY)
    (dest / "report" / "report_template.md").write_text(REPORT_TEMPLATE.format(**fmt))
    (dest / "review" / "referee_report_template.md").write_text(REFEREE_TEMPLATE)
    (dest / "review" / "response_to_referees_template.md").write_text(RESPONSE_TEMPLATE)
    (dest / "tests" / "test_replication.py").write_text(TEST_PY)
    (dest / ".gitignore").write_text(GITIGNORE)
    (dest / "LICENSE").write_text(LICENSE.format(year=date.today().year, authors=authors or "The authors"))
    (dest / "CITATION.cff").write_text(CITATION.format(title=title, authors=authors or "The authors",
                                                       today=date.today().isoformat()))
    shutil.copytree(ROOT / "fintest", dest / "fintest", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "requirements.txt", dest / "requirements.txt")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="create an empty replication project")
    n.add_argument("--name", required=True)
    n.add_argument("--title", default="Replication project")
    n.add_argument("--claim", default="State the claim as one falsifiable sentence.")
    n.add_argument("--authors", default="")
    n.add_argument("--data", default="Describe each data source, its version and how to obtain it.")
    n.add_argument("--dir", default=".")
    n.add_argument("--overwrite", action="store_true")
    d = sub.add_parser("demo", help="create and run the worked example (momentum)")
    d.add_argument("--source", choices=["french", "sim"], default="french")
    d.add_argument("--dir", default=str(ROOT / "demo"))
    args = ap.parse_args(argv)
    if args.cmd == "new":
        p = new_project(Path(args.dir) / args.name, args.title, args.claim, args.authors, args.data, args.overwrite)
        print(f"Created {p}. Next: edit preregistration_fields.json, original.json and config.json, then "
              f"`cd {p} && python run.py freeze`.")
        return p
    p = new_project(Path(args.dir) / "momentum-replication", "Momentum: winners minus losers (worked example)",
                    "Stocks with the highest returns over the past 12 months, skipping the most recent month, "
                    "outperform those with the lowest returns over the following month.",
                    "Capstone template", "Kenneth R. French Data Library, 10 Portfolios Formed on Prior (12-2) Return, "
                    "value-weighted monthly returns.", overwrite=True)
    cmd = [sys.executable, str(p / "run.py"), "all", "--source", args.source]
    subprocess.run(cmd, check=True, cwd=p)
    print(f"Demo project in {p}; open report/report.md.")
    return p


if __name__ == "__main__":
    main()
