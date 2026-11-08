"""
Make sure the FruitFlyBrain repo gets a local commit every day, with nobody
touching it (Windows Task Scheduler).

    python daily_commit.py install   # register the scheduled task (run once)
    python daily_commit.py run       # commit today's snapshot unless it is already done
    python daily_commit.py status    # last runs + task state
    python daily_commit.py remove    # unregister the task

supervise.py already commits training progress every hour, but only while it
is running and only when a brain changed, so days where the supervisor was
down (or nothing moved) left no commit. This runs on its own schedule
(sign-in, unlock, and 21:00 daily with wake-from-sleep; a missed run happens as
soon as the PC is back), and once per day it:

  1. appends a dated entry to training/daily/<YYYY-MM>.md: every brain's level,
     physics validation/test score and round, whether the supervisor is alive,
     and what was committed in the last 24 h;
  2. commits that entry together with any training output the supervisor has
     not committed yet (the same paths supervise.py publishes).

It only ever commits those paths, using `git commit --only`, so half-finished
code or anything you staged by hand is never swept into an automatic commit.
It never pushes unless asked (--push); the supervisor's hourly push publishes
these commits when it runs. Every run is idempotent: a second run on the same
day sees today's snapshot commit and exits. Results go to
training/logs/daily_commit.log.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PYW = REPO / ".venv-sim" / "Scripts" / "pythonw.exe"
TASK = "FlyLab daily commit"
SUBJECT = "FlyLab: daily snapshot"
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
TASKS = ["walk_forward", "turn", "goto", "odor_seek", "odor_avoid", "light_seek", "light_avoid"]
# What supervise.py publishes (relative to the FlyLab folder / the repo root).
FLYLAB_PATHS = ["training", "state/brains", "dashboard/videos"]
REPO_PATHS = ["README.md", "docs/gifs"]
LOCK_STALE_S = 15 * 60
OptimalFn = Callable[[dict], bool] | None


def log_path(flylab: Path) -> Path:
    return flylab / "training" / "logs" / "daily_commit.log"


def log(flylab: Path, msg: str) -> None:
    p = log_path(flylab)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    if sys.stdout:                       # pythonw has no console
        print(msg)


def git_exe() -> str:
    found = shutil.which("git")
    if found:
        return found
    for cand in (r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files (x86)\Git\cmd\git.exe"):
        if Path(cand).exists():
            return cand
    return "git"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([git_exe(), *args], cwd=repo, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# ---------------------------------------------------------------- snapshot


def _optimal_fn() -> OptimalFn:
    """train_brains.is_optimal when the sim venv is importable, else None."""
    try:
        sys.path.insert(0, str(HERE))
        from train_brains import is_optimal  # noqa: PLC0415
        return is_optimal
    except Exception:  # noqa: BLE001 - numpy/flygym missing or broken: fall back to raw numbers
        return None


def brain_rows(flylab: Path, is_optimal: OptimalFn = None) -> list[str]:
    rows = ["| brain | state | level | physics val | test | round |", "|---|---|---|---|---|---|"]
    for task in TASKS:
        p = flylab / "training" / task / "status.json"
        if not p.exists():
            rows.append(f"| {task} | no status yet | | | | |")
            continue
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            rows.append(f"| {task} | unreadable ({type(e).__name__}) | | | | |")
            continue
        best, test = st.get("best") or {}, st.get("test") or {}
        state = "training"
        if is_optimal is not None:
            try:
                state = "optimal" if is_optimal(st) else "training"
            except Exception:  # noqa: BLE001
                state = "?"
        val = f"{best.get('physics_val', 0):.0%} of {best.get('n', '?')}" if best else "-"
        tst = f"{test.get('physics_test', 0):.0%}" if test else "-"
        rows.append(f"| {task} | {state} | {st.get('level', 0) + 1} | {val} | {tst} | {st.get('round', 0)} |")
    return rows


def supervisor_running(flylab: Path) -> bool | None:
    if sys.platform != "win32":
        return None
    pat = str(flylab / "supervise.py")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
                        f"Where-Object {{ $_.CommandLine -like '*{pat}*' }} | Measure-Object | "
                        "ForEach-Object Count"], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = r.stdout.strip()
    return int(out) > 0 if out.isdigit() else None


def snapshot_entry(repo: Path, flylab: Path, today: date, is_optimal: OptimalFn = None) -> str:
    sup = supervisor_running(flylab)
    recent = git(repo, "log", "--since=24 hours ago", "--format=%h %ad %s", "--date=format:%H:%M").stdout
    recent = [l for l in recent.splitlines() if l.strip()]
    lines = [f"## {today.isoformat()} ({today.strftime('%a')})", "", *brain_rows(flylab, is_optimal), ""]
    lines.append("- training supervisor: " + {True: "running", False: "**not running**",
                                                None: "unknown"}[sup])
    if recent:
        lines.append(f"- commits in the last 24 h: {len(recent)} (latest `{recent[0]}`)")
    else:
        lines.append("- commits in the last 24 h: none")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- committing


def busy_reason(repo: Path) -> str | None:
    gd = Path(git(repo, "rev-parse", "--absolute-git-dir").stdout.strip() or repo / ".git")
    for marker, why in (("rebase-merge", "a rebase"), ("rebase-apply", "a rebase/am"),
                        ("MERGE_HEAD", "a merge"), ("CHERRY_PICK_HEAD", "a cherry-pick"),
                        ("BISECT_LOG", "a bisect")):
        if (gd / marker).exists():
            return f"repo is in the middle of {why}"
    lock = gd / "index.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime > LOCK_STALE_S and not git_running():
        lock.unlink(missing_ok=True)     # left behind by a killed git; nothing is using it
    return None


def git_running() -> bool:
    if sys.platform != "win32":
        return True                      # can't tell cheaply: assume busy, never delete the lock
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq git.exe", "/NH"], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return "git.exe" in r.stdout


def done_today(repo: Path, today: date) -> str | None:
    r = git(repo, "log", f"--since={today.isoformat()} 00:00", "--fixed-strings",
            f"--grep={SUBJECT} {today.isoformat()}", "--format=%h", "-1")
    return r.stdout.strip() or None


def run(repo: Path = REPO, flylab: Path = HERE, today: date | None = None, push: bool = False,
        branch: str = "main", is_optimal: OptimalFn = None, retries: int = 6, wait_s: float = 20) -> str:
    """Commit today's snapshot. Returns a one-line outcome (also logged)."""
    today = today or date.today()

    def out(msg: str) -> str:
        log(flylab, msg)
        return msg

    if (h := done_today(repo, today)):
        return out(f"already committed today ({h}); nothing to do")
    if (why := busy_reason(repo)):
        return out(f"skipped: {why}; will retry on the next trigger")
    cur = git(repo, "symbolic-ref", "--short", "-q", "HEAD").stdout.strip()
    if cur != branch:
        return out(f"skipped: on {cur or 'a detached HEAD'}, not {branch}; will retry on the next trigger")

    journal = flylab / "training" / "daily" / f"{today:%Y-%m}.md"
    journal.parent.mkdir(parents=True, exist_ok=True)
    text = journal.read_text(encoding="utf-8") if journal.exists() else (
        f"# FlyLab daily snapshots, {today:%B %Y}\n\nWritten by daily_commit.py once a day.\n")
    if f"## {today.isoformat()} " not in text:          # a failed earlier run may have written it already
        if is_optimal is None:
            is_optimal = _optimal_fn()
        text = text.rstrip("\n") + "\n\n" + snapshot_entry(repo, flylab, today, is_optimal)
        journal.write_text(text, encoding="utf-8", newline="\n")

    rel = flylab.relative_to(repo).as_posix()
    paths = [f"{rel}/{p}" for p in FLYLAB_PATHS] + REPO_PATHS
    paths = [p for p in paths if (repo / p).exists()]
    msg_lines = [f"{SUBJECT} {today.isoformat()}", ""]
    status = git(repo, "status", "--porcelain", "--", *paths).stdout.splitlines()
    extra = [l[3:] for l in status if not l[3:].endswith(journal.name)]
    msg_lines.append(f"Daily entry in {journal.relative_to(repo).as_posix()}"
                     + (f", plus {len(extra)} uncommitted training output file(s)." if extra else "."))
    msg = "\n".join(msg_lines) + f"\n\n{TRAILER}\n"

    last = ""
    for attempt in range(1, retries + 1):           # the supervisor may be committing right now
        a = git(repo, "add", "--", *paths)
        c = a if a.returncode else git(repo, "commit", "-q", "--only", "-m", msg, "--", *paths)
        if not c.returncode:
            h = git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
            res = f"committed {h}: {SUBJECT} {today.isoformat()} ({len(extra)} output file(s) included)"
            if push:
                git(repo, "pull", "-q", "--rebase", "--autostash", "origin", branch)
                p = git(repo, "push", "-q", "origin", "HEAD")
                res += "; pushed" if not p.returncode else f"; push failed: {p.stderr.strip()[:200]}"
            return out(res)
        last = (c.stderr or c.stdout).strip()
        if (h := done_today(repo, today)):             # someone else committed it meanwhile
            return out(f"already committed today ({h}); nothing to do")
        if "index.lock" not in last or attempt == retries:
            break
        time.sleep(wait_s)
    return out(f"FAILED to commit: {last[:400]}")


# ---------------------------------------------------------------- scheduling


def ps(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True)


def install() -> None:
    py = PYW if PYW.exists() else Path(sys.executable).with_name("pythonw.exe")
    action = (f"New-ScheduledTaskAction -Execute '{py}' -Argument '\"{Path(__file__)}\" run' "
              f"-WorkingDirectory '{HERE}'")
    script = f"""
$ErrorActionPreference = 'Stop'
$user = "$env:USERDOMAIN\\$env:USERNAME"
$a = {action}
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$logon.Delay = 'PT2M'
$daily = New-ScheduledTaskTrigger -Daily -At '21:00'
$cls = Get-CimClass -Namespace ROOT\\Microsoft\\Windows\\TaskScheduler -ClassName MSFT_TaskSessionStateChangeTrigger
$unlock = New-CimInstance -CimClass $cls -ClientOnly -Property @{{StateChange = 8; UserId = $user; Delay = 'PT1M'}}
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
     -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
     -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
$p = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger @($logon, $unlock, $daily) -Settings $s `
     -Principal $p -Description 'Daily FlyLab snapshot commit (daily_commit.py)' -Force | Out-Null
'ok'
"""
    r = ps(script)
    ok = r.stdout.strip().endswith("ok")
    log(HERE, f"scheduled task '{TASK}': {'installed' if ok else 'FAILED ' + r.stderr.strip()[:400]}")


def remove() -> None:
    r = ps(f"Unregister-ScheduledTask -TaskName '{TASK}' -Confirm:$false; 'ok'")
    log(HERE, f"scheduled task '{TASK}' removed: {r.stdout.strip() or r.stderr.strip()[:200]}")


def status() -> None:
    r = ps(f"$t = Get-ScheduledTask -TaskName '{TASK}' -ErrorAction SilentlyContinue; "
           "if ($t) { $i = $t | Get-ScheduledTaskInfo; "
           "\"task: $($t.State); last run $($i.LastRunTime) (result $($i.LastTaskResult)); next $($i.NextRunTime)\" }"
           " else { 'task: NOT installed (python daily_commit.py install)' }")
    print(r.stdout.strip())
    h = done_today(REPO, date.today())
    print(f"today: {'committed ' + h if h else 'not committed yet'}")
    p = log_path(HERE)
    if p.exists():
        print("recent log:")
        for line in p.read_text(encoding="utf-8").splitlines()[-8:]:
            print("  " + line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="status", choices=["run", "install", "status", "remove"])
    ap.add_argument("--push", action="store_true", help="also push after committing (run only)")
    args = ap.parse_args()
    if args.cmd == "run":
        try:
            run(push=args.push)
        except Exception as e:  # noqa: BLE001 - a scheduled run must leave a trace, never a silent crash
            log(HERE, f"FAILED with {type(e).__name__}: {e}")
            raise
    else:
        {"install": install, "status": status, "remove": remove}[args.cmd]()


if __name__ == "__main__":
    os.chdir(HERE)
    main()
