"""Tests for Lab 23. Run with:  python tests/test_lab23_capstone.py   (or: pytest tests/)

Runs offline: data fingerprints, the append-only run log, the comparison with original statistics and the
overall status, project creation, and the worked example run end to end on simulated data (including the
project's own test and the frozen pre-registration).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fintest.design import verify_preregistration  # noqa: E402
from fintest.replicate import RunLog, compare_to_original, fingerprint, replication_status  # noqa: E402


def test_fingerprint_and_log():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "a.csv"
        f.write_text("x\n1\n")
        fp = fingerprint([f, Path(tmp) / "missing.csv"], base=tmp)
        import hashlib
        assert fp["a.csv"]["sha256"] == hashlib.sha256(b"x\n1\n").hexdigest()
        assert len(fp["a.csv"]["sha256"]) == 64 and fp["missing.csv"]["note"] == "missing"
        log = RunLog(Path(tmp) / "logs" / "run.jsonl", base=tmp)
        log.record("one", inputs=[f])
        log.record("two")
        recs = log.read()
        assert [r["step"] for r in recs] == ["one", "two"] and "a.csv" in recs[0]["inputs"]


def test_comparison_and_status():
    t = compare_to_original({"a": 1.0, "b": 2.0, "c": -1.0}, {"a": 1.05, "b": 3.0, "c": 0.5}, rel_tol=0.1)
    assert list(t["status"]) == ["match", "same sign, different size", "different sign"]
    assert replication_status(t) == "not reproduced"
    assert replication_status(t.iloc[:2]) == "partly reproduced"
    assert replication_status(t.iloc[:1]) == "reproduced"
    assert compare_to_original({"a": 1.0}, {})["status"].iloc[0] == "not reproduced"


def test_new_project_layout():
    import lab23_capstone

    with tempfile.TemporaryDirectory() as tmp:
        p = lab23_capstone.main(["new", "--name", "proj", "--dir", tmp, "--title", "T", "--claim", "C"])
        for f in ("README.md", "run.py", "original.json", "config.json", "preregistration_fields.json",
                  "src/load.py", "src/analysis.py", "report/report_template.md", "review/referee_report_template.md",
                  "tests/test_replication.py", "LICENSE", "CITATION.cff", ".gitignore", "fintest/design.py"):
            assert (p / f).exists(), f


def test_demo_end_to_end():
    import lab23_capstone

    with tempfile.TemporaryDirectory() as tmp:
        p = lab23_capstone.main(["demo", "--source", "sim", "--dir", tmp])
        assert verify_preregistration((p / "preregistration.md").read_text())
        assert (p / "results" / "status.txt").read_text().strip() in ("reproduced", "partly reproduced", "not reproduced")
        rep = (p / "report" / "report.md").read_text()
        assert "{{" not in rep and "Pre-registration hash: valid" in rep
        steps = [json.loads(l)["step"] for l in (p / "logs" / "run_log.jsonl").read_text().splitlines()]
        assert steps == ["freeze", "data", "replicate", "extend", "report"]
        # freezing again must not overwrite the plan
        before = (p / "preregistration.md").read_text()
        subprocess.run([sys.executable, str(p / "run.py"), "freeze"], check=True, cwd=p, capture_output=True)
        assert (p / "preregistration.md").read_text() == before
        # the project's own test passes
        subprocess.run([sys.executable, str(p / "tests" / "test_replication.py")], check=True, cwd=p,
                       capture_output=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
