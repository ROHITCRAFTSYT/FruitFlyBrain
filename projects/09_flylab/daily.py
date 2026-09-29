"""
Once-a-day upkeep, run by supervise.py:

  1. recheck.py     -- test every optimal brain on fresh physics arenas
  2. README         -- rewrite the brain-status table in the top-level README
                       from training/*/status.json and the fresh checks
  3. make_gifs.py   -- regenerate the training-progress GIF and the brain films

It only reads training results and writes documentation; it never changes a brain.

    ..\\..\\.venv-sim\\Scripts\\python daily.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import recheck
import tasks as T
from train_brains import LEVELS, TRAIN_DIR, is_optimal, wilson


def status(task):
    p = TRAIN_DIR / task / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def last_check(task):
    p = TRAIN_DIR / task / "checks.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    return rows[-1] if rows else None


def brain_table():
    rows, n_opt = [], 0
    for task in T.TASKS:
        st = status(task)
        if not st:
            continue
        b, t = st.get("best") or {}, st.get("test") or {}
        opt = is_optimal(st)
        n_opt += opt
        nt = t.get("n", 0)
        lo, hi = wilson(t.get("physics_test", 0), nt)
        c = last_check(task)
        fresh = f"{c['success']:.0%} of {c['n']} ({c['date']})" if c else "–"
        state = "✅ optimal" if opt else f"🔄 training (level {st.get('level', 0) + 1}/{len(LEVELS)})"
        rows.append(f"| {T.PRETTY[task]} | {st['round']} | {b.get('physics_val', 0):.0%} of {b.get('n', 0)} | "
                    f"**{t.get('physics_test', 0):.0%} of {nt}** ({lo:.0%}–{hi:.0%}) | {fresh} | {state} |")
    head = [f"**Status on {date.today().isoformat()}** (rewritten daily by `daily.py`; the "
            "[dashboard](https://rohitcraftsyt.github.io/FruitFlyBrain/projects/09_flylab/dashboard/) is live):", "",
            "| Skill brain | Rounds | Physics validation | **Held-out physics test** (95% CI) | Fresh-arena check | Status |",
            "|---|---:|---:|---:|---:|---|"]
    return "\n".join(head + rows), n_opt


def update_readme():
    readme = REPO / "README.md"
    text = readme.read_text(encoding="utf-8")
    table, n_opt = brain_table()
    text, k1 = re.subn(r"<!-- BRAINS:START -->.*?<!-- BRAINS:END -->",
                       f"<!-- BRAINS:START -->\n{table}\n<!-- BRAINS:END -->", text, flags=re.S)
    text, k2 = re.subn(r"<!-- OPT -->.*?<!-- /OPT -->", f"<!-- OPT -->{n_opt} of {len(T.TASKS)}<!-- /OPT -->", text)
    if k1:
        readme.write_text(text, encoding="utf-8")
    print(f"README brain table {'updated' if k1 else 'markers not found'}; optimal count "
          f"{'updated' if k2 else 'marker not found'} ({n_opt}/{len(T.TASKS)})", flush=True)


def main():
    print("== fresh-arena checks ==", flush=True)
    sys.argv = [sys.argv[0]]
    recheck.main()
    print("== README ==", flush=True)
    update_readme()
    print("== GIFs ==", flush=True)
    for job in ("p09", "flylab_videos"):
        r = subprocess.run([sys.executable, "-u", str(REPO / "make_gifs.py"), job], cwd=REPO,
                           capture_output=True, text=True)
        print(f"make_gifs {job}: {'ok' if r.returncode == 0 else 'failed: ' + r.stderr[-400:]}", flush=True)


if __name__ == "__main__":
    main()
