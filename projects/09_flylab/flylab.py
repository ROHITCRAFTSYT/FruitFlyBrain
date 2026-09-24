"""
FlyLab -- tell a virtual fruit fly what to do; it learns, then does it.

    ..\\..\\.venv-sim\\Scripts\\python flylab.py                  # interactive
    ..\\..\\.venv-sim\\Scripts\\python flylab.py "find the food"  # one command
    ..\\..\\.venv-sim\\Scripts\\python flylab.py demo             # full curriculum

Each command is a LESSON:
  1. the brain trains on that task in the calibrated surrogate body
     (Evolution Strategies) -- every skill has its own brain, so nothing is overwritten;
  2. the trained brain is loaded into the full NeuroMechFly physics fly, which
     performs your exact request in MuJoCo;
  3. you get a video, a trajectory map and an updated learning chart.
The brain is saved after every lesson, so the fly keeps improving across
sessions. Type `help` for everything it understands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import brain as B
import render as R
import surrogate as SG
import tasks as T
import trainer as TR

STATE = HERE / "state"
OUT = HERE / "outputs"
LESSONS = OUT / "lessons"

NEW_SKILL_GENS = 150
NEW_SENSORY_SKILL_GENS = 250    # harder skills (noisy senses / precise turning) get more practice
SENSORY = {"odor_seek", "odor_avoid", "light_seek", "light_avoid", "turn"}
IMPROVE_GENS = 100
REFRESH_GENS = 30

HELP = """
What you can tell the fly (plain English works):
  find the food | follow the smell | go eat the banana     -> smell its way to food
  avoid the smell | escape the repellent                   -> run from a bad odor
  go to the light | walk toward the sun                    -> phototaxis
  hide from the light | avoid the light                    -> seek the dark
  go to 10, 5   (mm; fly starts at 0,0 facing +x)          -> navigate to a point
  walk forward 12 mm | walk straight                       -> walk and stop
  turn left 90 | turn right 45 | turn around               -> turn on the spot
  practice <any task above> [generations]                  -> extra training, no video
Other commands:
  skills   show what it has learned      show   replay the last video
  demo     teach every skill in order    reset  wipe all brains (start as a newborn)
Continuous training of every brain (physics-validated): run train_brains.py
  help     this text                     quit   leave FlyLab
""".rstrip()


def banner(brain):
    print("=" * 72)
    print("  FlyLab  |  a virtual Drosophila you can teach")
    print("  body: NeuroMechFly v2 (EPFL) in MuJoCo physics   brain: "
          f"{B.N_PARAMS}-weight neural net")
    known = brain.known_skills()
    print(f"  lessons so far: {len(brain.history['lessons'])}   "
          f"skills: {', '.join(T.PRETTY[t] for t in known) or 'none yet (newborn)'}")
    print("=" * 72)
    print("Type a task (e.g. 'find the food'), or 'help'.")


def ensure_calibrated():
    if (STATE / "calibration.npz").exists():
        return
    print("\nFirst run: measuring how the physics fly moves (one-time, a few minutes)...")
    import calibrate
    calibrate.main()


def progress_bar(g, total, sr, mr):
    w = 28
    k = int(w * g / total)
    print(f"\r    training [{'#' * k}{'.' * (w - k)}] gen {g:>3}/{total}  "
          f"success {sr:4.0%}  reward {mr:+.2f}", end="", flush=True)


def show_skills(brain):
    skills = brain.history["skills"]
    if not skills:
        print("  The fly hasn't learned anything yet. Try 'find the food'.")
        return
    print(f"  {'skill':<22}{'generations':>12}{'surrogate':>11}{'phys val':>10}{'phys test':>11}  status")
    for t in T.TASKS:
        if t in skills:
            s = skills[t]
            pv = f"{s['physics_val']:.0%}" if s.get("physics_val") is not None else "-"
            pt = f"{s['physics_test']:.0%}" if s.get("physics_test") is not None else "-"
            print(f"  {T.PRETTY[t]:<22}{s.get('generations', 0):>12}{s.get('surrogate_success', 0):>10.0%}"
                  f"{pv:>10}{pt:>11}  {'mastered' if s.get('mastered') else 'learning'}")
    print(f"  total generations trained: {brain.history['generations_total']}")


def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')
    except Exception:
        pass


def verdict_text(task, ok, traj, scen):
    x, y = traj["x"][-1], traj["y"][-1]
    if task == "turn":
        err = np.degrees(abs(T.wrap(traj["th"][-1] - scen.extra["target_heading"][0])))
        return (f"{'SUCCESS' if ok else 'not quite'}: finished {err:.0f} deg from the target heading")
    tx, ty = scen.extra["target"][0][0], scen.extra["target"][1][0]
    d0 = np.hypot(traj["x"][0] - tx, traj["y"][0] - ty)
    df = np.hypot(x - tx, y - ty)
    if task in ("odor_avoid", "light_avoid"):
        return f"{'SUCCESS' if ok else 'not quite'}: moved from {d0:.1f} mm to {df:.1f} mm away"
    return f"{'SUCCESS' if ok else 'not quite'}: ended {df:.1f} mm from the target (started {d0:.1f} mm away)"


def lesson(brain, body, command, task, params, args, generations=None, perform=True):
    known = brain.history["skills"].get(task, {})
    if generations is None:
        if known.get("mastered"):
            generations = 0          # physics-validated brain: perform, don't disturb it
        elif not known:
            generations = NEW_SENSORY_SKILL_GENS if task in SENSORY else NEW_SKILL_GENS
        elif known.get("surrogate_success", 0) >= 0.9:
            generations = REFRESH_GENS
        else:
            generations = IMPROVE_GENS
    n_lesson = len(brain.history["lessons"]) + 1
    status = ("mastered (validated in physics)" if generations == 0 else
              "new skill" if not known else f"practising (lesson #{known.get('lessons', 0) + 1} on it)")
    print(f"\nLesson {n_lesson}: {T.PRETTY[task]}  [{status}]")
    if generations:
        print(f"  1) training this skill's brain in the calibrated body: {generations} generations x 32 brains")
        theta, log = TR.train(brain.theta(task), body, task, generations, progress=progress_bar)
        brain.set_theta(task, theta)
        print(f"\n    success {log['success_before']:.0%} -> {log['success_after']:.0%}"
              f"   (reward {log['reward_before']:+.2f} -> {log['reward_after']:+.2f},"
              f" {log['seconds']:.0f}s)")
    else:
        sr, mr = TR.evaluate(body, brain.theta(task), task)
        log = {"generations": 0, "success_before": sr, "success_after": sr, "reward_before": mr,
               "reward_after": mr, "curve": [(0, sr, mr)], "seconds": 0.0}
        print("  1) no training needed: using its physics-validated brain "
              f"(held-out physics test {known.get('physics_test') or 0:.0%})")

    sk = brain.history["skills"].setdefault(task, {})
    sk["lessons"] = sk.get("lessons", 0) + 1
    sk["generations"] = sk.get("generations", 0) + generations
    sk["surrogate_success"] = log["success_after"]
    brain.history["generations_total"] += generations
    entry = {"n": n_lesson, "time": datetime.now().isoformat(timespec="seconds"),
             "command": command, "task": task, "params": {k: v for k, v in params.items()
                                                           if not k.startswith("_")},
             **{k: log[k] for k in ("generations", "success_before", "success_after",
                                    "reward_before", "reward_after", "curve", "seconds")}}

    if perform and not args.no_physics:
        print("  2) loading the brain into the NeuroMechFly physics body...")
        scen = T.sample(task, 1, np.random.default_rng(n_lesson), fixed={"deploy": True, **params})
        theta = brain.theta(task)
        pred = SG.rollout(body, theta[None], scen, rng=None, noise=0)
        import physics
        traj, frames = physics.deploy(theta, scen, record=not args.no_video,
                                      log=print if args.verbose else None)
        rew, ok = T.score(scen, traj["x"][None], traj["y"][None], traj["th"][None])
        ok = bool(ok[0])
        verdict = verdict_text(task, ok, traj, scen)
        print(f"    physics run: {scen.duration:.0f}s simulated in {traj['wall_seconds']:.0f}s  ->  {verdict}")
        sk["physics_success"] = ok
        entry.update(physics_success=ok, physics_reward=float(rew[0]), verdict=verdict)

        LESSONS.mkdir(parents=True, exist_ok=True)
        stem = f"{n_lesson:03d}_{task}"
        title = f'"{command}"'
        sub = f"{T.PRETTY[task]}  |  lesson {n_lesson}  |  brain trained {sk['generations']} generations on this skill"
        png = R.plot_trajectory(scen, traj, (pred[0][0, 0], pred[1][0, 0]),
                                f"{title}  ->  {verdict}", LESSONS / f"{stem}.png")
        entry["trajectory_png"] = f"lessons/{png.name}"
        if frames:
            print("  3) rendering video...")
            mp4 = R.make_video(frames, traj, title, sub, verdict, ok, LESSONS / f"{stem}.mp4")
            entry["video"] = f"lessons/{mp4.name}"
            brain.history["last_video"] = str(mp4)
            if not args.no_open:
                open_file(mp4)
        print(f"    saved: {LESSONS / stem}.*")

    brain.history["lessons"].append(entry)
    brain.save(task)
    R.plot_progress(brain.history, OUT / "learning_progress.png")
    write_diary(brain.history)
    return entry


def write_diary(history):
    """A local HTML page with every lesson: command, learning numbers, video, map."""
    rows = []
    for e in reversed(history["lessons"]):
        media = ""
        if e.get("video"):
            media += f'<video src="{e["video"]}" controls muted loop width="640"></video>'
        if e.get("trajectory_png"):
            media += f'<img src="{e["trajectory_png"]}" width="360">'
        phys = e.get("verdict", "surrogate only")
        badge = "ok" if e.get("physics_success") else "no"
        rows.append(f"""
<section><h3>Lesson {e['n']}: &ldquo;{e['command']}&rdquo; <small>{T.PRETTY[e['task']]} &middot; {e['time']}</small></h3>
<p>Trained {e['generations']} generations &middot; surrogate success {e['success_before']:.0%} &rarr; <b>{e['success_after']:.0%}</b>
&middot; physics: <span class="{badge}">{phys}</span></p><div class="media">{media}</div></section>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>FlyLab diary</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}}
section{{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:12px 16px;margin:14px 0}}
.media{{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}} video,img{{max-width:100%;border-radius:6px}}
small{{color:#777;font-weight:normal}} .ok{{color:#1a7f37;font-weight:600}} .no{{color:#b35900;font-weight:600}}
</style></head><body>
<h1>FlyLab diary</h1>
<p>A NeuroMechFly v2 physics fly, taught one command at a time. {len(history['lessons'])} lessons,
{history['generations_total']} training generations.</p>
<img src="learning_progress.png" style="width:100%">
{''.join(rows)}
</body></html>"""
    (OUT / "fly_diary.html").write_text(html, encoding="utf-8")


DEMO = ["walk forward 12 mm", "turn left 90", "go to 8, -6", "find the food",
        "go to the light", "hide from the light", "avoid the smell",
        "turn right 45", "find the food"]


def handle(brain, body, cmd, args):
    kind, what, params = T.parse(cmd)
    if kind == "unknown":
        print("  Hmm, the fly didn't understand that. Try 'help' for examples.")
    elif kind == "stop":
        print("  The fly stops: both descending drives set to 0 (its legs stand still).")
    elif kind == "task":
        ensure_calibrated()
        lesson(brain, body, cmd, what, params, args)
    elif what == "help":
        print(HELP)
    elif what == "skills":
        show_skills(brain)
    elif what == "show":
        last = brain.history.get("last_video")
        open_file(last) if last else print("  No video yet.")
    elif what == "reset":
        brain.reset()
        print("  Brain wiped. The fly is a newborn again.")
    elif what == "sleep":
        print("  Each skill now has its own brain, so there is nothing to consolidate --\n"
              "  run train_brains.py to keep improving every brain with physics validation.")
    elif what == "practice":
        ensure_calibrated()
        p = dict(params); task = p.pop("task"); gens = p.pop("_generations")
        lesson(brain, body, cmd, task, p, args, generations=gens, perform=False)
    elif what == "demo":
        ensure_calibrated()
        for c in DEMO:
            print(f"\n>>> {c}")
            handle(brain, body, c, args)
        print(f"\nDemo done. Open {OUT / 'fly_diary.html'} to review every lesson.")
    elif what == "quit":
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Teach a virtual fruit fly.")
    ap.add_argument("command", nargs="*", help="a task, e.g. find the food (omit for interactive)")
    ap.add_argument("--no-physics", action="store_true", help="train only; skip the physics run")
    ap.add_argument("--no-video", action="store_true", help="run physics but don't render video")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the video")
    ap.add_argument("--verbose", action="store_true", help="print the fly's state during physics")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
    brain = B.Brain(STATE)
    ensure_calibrated()
    body = SG.Body()

    if args.command:
        handle(brain, body, " ".join(args.command), args)
        return
    banner(brain)
    while True:
        try:
            cmd = input("\nfly> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd and not handle(brain, body, cmd, args):
            break
    print(f"Goodbye. The brain is saved in {STATE}; your diary is {OUT / 'fly_diary.html'}")


if __name__ == "__main__":
    main()
