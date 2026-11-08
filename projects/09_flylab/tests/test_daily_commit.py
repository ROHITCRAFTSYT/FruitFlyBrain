"""
daily_commit.py must leave exactly one snapshot commit per day, include any
uncommitted training output, and never sweep in anything else (hand-staged
work or code edits). Runs against a throwaway git repo, never the real one.

    ..\\..\\..\\.venv-sim\\Scripts\\python -m pytest test_daily_commit.py
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import daily_commit as DC  # noqa: E402

DAY = date(2026, 11, 8)


def sh(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture(autouse=True)
def no_sim_import(monkeypatch):
    monkeypatch.setattr(DC, "_optimal_fn", lambda: (lambda st: st.get("level") == 2))   # skip train_brains/flygym


@pytest.fixture
def repo(tmp_path: Path):
    sh(tmp_path, "init", "-q", "-b", "main")
    sh(tmp_path, "config", "user.name", "Test")
    sh(tmp_path, "config", "user.email", "test@example.com")
    sh(tmp_path, "config", "commit.gpgsign", "false")
    flylab = tmp_path / "projects" / "09_flylab"
    (flylab / "training" / "turn").mkdir(parents=True)
    (flylab / "training" / "turn" / "status.json").write_text(json.dumps(
        {"task": "turn", "round": 7, "level": 2, "best": {"physics_val": 1.0, "n": 24},
         "test": {"physics_test": 0.94}}))
    (flylab / "brain.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("projects/09_flylab/training/logs/\n")
    sh(tmp_path, "add", "-A")
    sh(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path, flylab


def commits(repo: Path) -> list[str]:
    return sh(repo, "log", "--format=%s").splitlines()


def test_one_commit_per_day_with_output_but_nothing_else(repo):
    root, flylab = repo
    # pending training output (should be committed) ...
    (flylab / "training" / "turn" / "status.json").write_text(json.dumps({"task": "turn", "round": 8, "level": 2}))
    # ... a code edit and a hand-staged file (must NOT be committed)
    (flylab / "brain.py").write_text("x = 2\n")
    (root / "notes.txt").write_text("wip\n")
    sh(root, "add", "notes.txt")

    res = DC.run(root, flylab, today=DAY, retries=1, wait_s=0)
    assert res.startswith("committed"), res
    assert commits(root)[0] == f"{DC.SUBJECT} {DAY.isoformat()}"
    files = sh(root, "show", "--name-only", "--format=", "HEAD").split()
    assert "projects/09_flylab/training/daily/2026-11.md" in files
    assert "projects/09_flylab/training/turn/status.json" in files
    assert "projects/09_flylab/brain.py" not in files and "notes.txt" not in files
    assert "notes.txt" in sh(root, "diff", "--cached", "--name-only")     # still staged, untouched
    journal = (flylab / "training" / "daily" / "2026-11.md").read_text()
    assert "## 2026-11-08 (Sun)" in journal and "| turn |" in journal

    again = DC.run(root, flylab, today=DAY, retries=1, wait_s=0)
    assert again.startswith("already committed today")
    assert len(commits(root)) == 2
    assert journal == (flylab / "training" / "daily" / "2026-11.md").read_text()   # no duplicate entry


def test_next_day_appends_new_entry(repo):
    root, flylab = repo
    DC.run(root, flylab, today=DAY, retries=1, wait_s=0)
    nxt = date(2026, 11, 9)
    res = DC.run(root, flylab, today=nxt, retries=1, wait_s=0)
    assert res.startswith("committed"), res
    journal = (flylab / "training" / "daily" / "2026-11.md").read_text()
    assert journal.count("## 2026-11-0") == 2


def test_skips_off_main_and_mid_merge(repo):
    root, flylab = repo
    sh(root, "checkout", "-q", "-b", "feature")
    assert "skipped: on feature" in DC.run(root, flylab, today=DAY, retries=1, wait_s=0)
    sh(root, "checkout", "-q", "main")
    (root / ".git" / "MERGE_HEAD").write_text("0" * 40 + "\n")
    assert "middle of a merge" in DC.run(root, flylab, today=DAY, retries=1, wait_s=0)
    assert len(commits(root)) == 1
