"""
Keep every FlyLab brain training until it is optimal -- unattended.

The supervisor runs one `train_brains.py` process per skill (each skill keeps
its own brain and its own training process), at most --workers at a time,
fewest turns first, then weakest skill. Each process trains for a time slice (--slice hours) and
then yields its CPU core, so a hard skill can't starve the others. Crashed
workers are logged and relaunched; results are committed and pushed to git
every --push-every hours, and brains that changed are re-filmed in MuJoCo
(render_brains.py) for the dashboard.

It never stops on its own. Skills below optimal get CPU first; spare slots
polish optimal brains (train_brains.py --polish: a polished brain is kept only
if it is still optimal afterwards). Once a day it runs daily.py (fresh-arena
checks, README status table, GIFs) and resets crash counters. autostart.py keeps
it running across reboots.

    ..\\..\\.venv-sim\\Scripts\\python supervise.py              # 2 workers, 1 h slices
    ..\\..\\.venv-sim\\Scripts\\python supervise.py --workers 1  # one core only
    ..\\..\\.venv-sim\\Scripts\\python supervise.py --keep-awake # don't let Windows sleep meanwhile

Everything it does is appended to training/logs/supervisor.log; each skill's
trainer output goes to training/logs/<skill>.log.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tasks as T
from train_brains import LEVELS, TRAIN_DIR, write_summary
from train_brains import is_optimal as optimal

LOGS = TRAIN_DIR / "logs"
REPO = HERE.parent.parent
MAX_CRASHES = 3              # consecutive crashes before a skill is set aside
# Workers get their own process group and no console, so a Ctrl+C (or a console
# closing) elsewhere can't take the whole training run down with it.
DETACH = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else 0
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
DAILY_STAMP = LOGS / "daily_last.txt"   # date of the last daily.py run
POLISH_MAX_ROUNDS = 1_000_000


def _now():
    return datetime.now().isoformat(timespec="seconds")


def say(msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "supervisor.log", "a", encoding="utf-8") as fh:
        fh.write(f"{_now()}  {msg}\n")


def status(task):
    p = TRAIN_DIR / task / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"round": 0, "level": 0, "best": None}




def foreign_jobs():
    """Skills (and "daily") whose train_brains.py / daily.py is already running but
    isn't ours -- e.g. workers orphaned when an earlier supervisor was killed.
    Starting a second trainer on such a skill would interleave two processes
    writing the same status/log files, so those skills wait until it exits."""
    if sys.platform != "win32":
        return set()
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                        "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True)
    jobs = set()
    for line in r.stdout.splitlines():
        if "daily.py" in line:
            jobs.add("daily")
        m = re.search(r"train_brains\.py\s.*--skills\s+(\S+)", line)
        if m:
            jobs.add(m.group(1))
    return jobs


def daily_due():
    return not DAILY_STAMP.exists() or DAILY_STAMP.read_text().strip() != date.today().isoformat()


def weakness(task):
    st = status(task)
    b = st.get("best") or {}
    return (st.get("level", 0), b.get("physics_val", -1), b.get("physics_reward", -9))


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def render_videos(proc):
    """Re-film brains that changed since their last video (real MuJoCo episodes,
    low priority, in the background); the next push publishes them."""
    if proc is not None and proc.poll() is None:
        return proc                                          # still rendering
    log = open(LOGS / "render.log", "a", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-u", "render_brains.py", "--changed-only"],
                            cwd=HERE, stdout=log, stderr=subprocess.STDOUT, creationflags=DETACH)


def push_progress():
    rel = HERE.relative_to(REPO).as_posix()
    git("add", f"{rel}/training", f"{rel}/state/brains", f"{rel}/dashboard/videos", "README.md", "docs/gifs")
    if not git("diff", "--cached", "--quiet").returncode:
        return                                              # nothing new
    lines = []
    for task in T.TASKS:
        st = status(task)
        b = st.get("best") or {}
        lines.append(f"- {task}: " + ("optimal" if optimal(st) else f"level {st.get('level', 0) + 1}/{len(LEVELS)}")
                     + f", physics val {b.get('physics_val', 0):.0%} of {b.get('n', 6)}, round {st['round']}")
    msg = "FlyLab: continuous training progress\n\n" + "\n".join(lines) + f"\n\n{TRAILER}\n"
    c = git("commit", "-q", "-m", msg)
    git("pull", "-q", "--rebase", "--autostash", "origin", "main")   # don't fail if GitHub is ahead
    p = git("push", "-q", "origin", "HEAD")
    say(f"git commit/push: {'ok' if not c.returncode and not p.returncode else (c.stderr + p.stderr).strip()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2, help="skills trained at the same time (CPU cores)")
    ap.add_argument("--slice", type=float, default=1.0, help="hours per training turn before yielding")
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--push-every", type=float, default=1.0, help="hours between git commits (0 = never)")
    ap.add_argument("--keep-awake", action="store_true", help="ask Windows not to sleep while training runs")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if args.keep_awake and sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED

    say(f"supervisor started: {args.workers} workers, {args.slice} h slices")
    running: dict[str, subprocess.Popen] = {}
    crashes = {t: 0 for t in T.TASKS}
    renderer = None
    turns = {t: 0 for t in T.TASKS}      # round-robin: fewest turns first, then weakest
    last_push = time.time()
    daily = None
    daily_tried = None
    reported = set()

    while True:
        for task, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[task]
            crashes[task] = 0 if code == 0 else crashes[task] + 1
            say(f"[{task}] trainer exited with code {code}" + (f" (crash {crashes[task]}/{MAX_CRASHES})" if code else ""))

        # once a day: fresh-arena checks, README table, GIFs (takes one worker slot)
        if daily is not None and daily.poll() is not None:
            say(f"daily.py finished with code {daily.returncode}")
            if daily.returncode == 0:   # stamp only on success: a run cut off by shutdown retries after reboot
                DAILY_STAMP.write_text(date.today().isoformat())
            daily = None
            write_summary()
            if args.push_every:
                push_progress()
        # anything still running from an earlier (killed) supervisor counts as busy
        busy = len(running) + (daily is not None)
        foreign = set()
        if busy < args.workers or (daily is None and daily_due()):   # only look when we'd start something
            foreign = foreign_jobs() - set(running) - ({"daily"} if daily is not None else set())
        if foreign != reported:
            if foreign:
                say(f"waiting for jobs left by an earlier supervisor: {sorted(foreign)}")
            reported = foreign
        if daily is None and "daily" not in foreign and daily_due() and daily_tried != date.today():
            daily_tried = date.today()                      # at most one attempt per day per supervisor run
            crashes = {t: 0 for t in T.TASKS}               # give set-aside skills a fresh chance
            log = open(LOGS / "daily.log", "a", encoding="utf-8")
            daily = subprocess.Popen([sys.executable, "-u", "daily.py"], cwd=HERE, stdout=log,
                                     stderr=subprocess.STDOUT, creationflags=DETACH)
            say(f"daily.py started (pid {daily.pid})")

        # skills below optimal first; spare slots polish optimal brains (never exits)
        ok = lambda t: crashes[t] < MAX_CRASHES and t not in running and t not in foreign
        pending = [(t, False) for t in T.TASKS if ok(t) and not optimal(status(t)) and status(t)["round"] < args.max_rounds]
        polish = [(t, True) for t in T.TASKS if ok(t) and optimal(status(t))]
        free = max(0, args.workers - len(running) - (daily is not None) - len(foreign))
        queue = sorted(pending, key=lambda x: (turns[x[0]], weakness(x[0]))) +             sorted(polish, key=lambda x: (turns[x[0]], weakness(x[0])))
        for task, pol in queue[:free]:
            log = open(LOGS / f"{task}.log", "a", encoding="utf-8")
            cmd = [sys.executable, "-u", "train_brains.py", "--skills", task, "--hours", str(args.slice),
                   "--val", str(args.val), "--test", str(args.test),
                   "--max-rounds", str(POLISH_MAX_ROUNDS if pol else args.max_rounds)] + (["--polish"] if pol else [])
            running[task] = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, creationflags=DETACH)
            turns[task] += 1
            say(f"[{task}] trainer started (pid {running[task].pid}), "
                + ("polishing an optimal brain" if pol else f"level {status(task).get('level', 0) + 1}/{len(LEVELS)}"))

        if args.push_every and time.time() - last_push > args.push_every * 3600:
            push_progress()
            renderer = render_videos(renderer)
            last_push = time.time()
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("supervisor crashed:\n" + traceback.format_exc())
        raise
