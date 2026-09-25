"""
Continuous training of every FlyLab skill brain, with physics in the loop.

Each skill has its OWN brain and its OWN training process. A round is:

  1. SURROGATE TRAINING   100 generations of Evolution Strategies in the
                          calibrated, domain-randomized surrogate body, with
                          noise/step size annealed to how good the brain already is
  2. PHYSICS VALIDATION   the new weights drive the full NeuroMechFly body in
                          MuJoCo on a fixed set of validation arenas
  3. SELECTION            the deployed brain (state/brains/<skill>.npy) is only
                          replaced if physics performance improved
  4. HELD-OUT TEST        every new best is scored on separate test arenas that
                          are NEVER used for selection -> honest reported numbers
  5. PATIENCE             after several rounds without improvement the working
                          copy restarts from the best checkpoint
  6. RESEED               a lineage that still hasn't mastered even the surrogate
                          after PLATEAU rounds is stuck in a local optimum (fresh
                          brains master it in 1-2 rounds): the working copy starts
                          over from fresh weights (random, or a cloned innate
                          reflex -- see reflexes.py), with double patience.
                          The deployed brain stays until the new lineage beats it.
  7. PROGRESSIVE LEVELS   mastering a level (100% physics validation, >=95%
                          surrogate) raises the bar: the validation and test
                          sets double (6 -> 12 -> 24 arenas by default), the
                          champion is re-scored on them, and training goes on.
                          A brain is OPTIMAL only once it holds up at the top
                          level, so a lucky streak on a few arenas can't end it.

The loop always works on the weakest skill next, keeps going until every skill
is optimal (or the time budget runs out), and is fully resumable: stop it any
time and run it again to continue exactly where it left off.

Every round is appended to training/<skill>/log.jsonl, and each skill gets a
curve (curve.png) and a report (REPORT.md); training/TRAINING.md summarizes all.

    ..\\..\\.venv-sim\\Scripts\\python train_brains.py                   # all skills
    ..\\..\\.venv-sim\\Scripts\\python train_brains.py --skills turn goto  # a subset
    ..\\..\\.venv-sim\\Scripts\\python train_brains.py --hours 2 --val 6 --test 10
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # 2 trainers can share a 2-core CPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import brain as B
import physics
import reflexes as RX
import surrogate as SG
import tasks as T
import trainer as TR
from evaluate_physics import random_deploy_scenario

TRAIN_DIR = HERE / "training"
STATE = HERE / "state"
ROUND_GENS = 100
PATIENCE = 4                 # rounds without a new best -> restart from best
PLATEAU = 8                  # rounds a lineage gets to master the surrogate before a reseed
SUR_EVAL_N = 96
MASTERED_SUR = 0.95          # surrogate success needed to pass any level
# Progressive mastery: (x the base --val/--test arena counts, physics val needed)
LEVELS = [(1, 1.0), (2, 1.0), (4, 0.95)]


# ---------------------------------------------------------------------------
# physics evaluation on fixed arenas
# ---------------------------------------------------------------------------
def physics_eval(theta, task, n, seed):
    """Run the brain on n fixed random arenas in full physics.
    Returns (success rate, mean reward, per-arena details)."""
    rng = np.random.default_rng(seed)
    oks, rews, details = [], [], []
    for _ in range(n):
        scen = random_deploy_scenario(task, rng)
        traj, _ = physics.deploy(theta, scen, record=False, log=None)
        rew, ok = T.score(scen, traj["x"][None], traj["y"][None], traj["th"][None])
        oks.append(bool(ok[0])); rews.append(float(rew[0]))
        details.append(_error_of(task, scen, traj))
    return float(np.mean(oks)), float(np.mean(rews)), details


def _error_of(task, scen, traj):
    """A human-readable error for one physics run (mm or degrees)."""
    if task == "turn":
        return round(float(np.degrees(abs(T.wrap(traj["th"][-1] - scen.extra["target_heading"][0])))), 1)
    tx, ty = scen.extra["target"][0][0], scen.extra["target"][1][0]
    df = float(np.hypot(traj["x"][-1] - tx, traj["y"][-1] - ty))
    if task in ("odor_avoid", "light_avoid"):
        d0 = float(np.hypot(traj["x"][0] - tx, traj["y"][0] - ty))
        return round(df - d0, 1)                    # mm gained (want > 6)
    return round(df, 2)                             # mm from target (want < 2)


def val_seed(task):
    return 10_000 + T.TASKS.index(task)


def test_seed(task):
    return 20_000 + T.TASKS.index(task)


# ---------------------------------------------------------------------------
# per-skill state
# ---------------------------------------------------------------------------
class SkillRun:
    def __init__(self, task, brain: B.Brain):
        self.task = task
        self.brain = brain
        self.dir = TRAIN_DIR / task
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dir / "status.json"
        self.cand_path = self.dir / "candidate.npy"
        self.log_path = self.dir / "log.jsonl"
        self.status = (json.loads(self.status_path.read_text()) if self.status_path.exists() else
                       {"task": task, "round": 0, "generations": 0, "physics_runs": 0,
                        "stale": 0, "best": None, "test": None, "started": _now()})

    @property
    def best_key(self):
        b = self.status["best"]
        return (-1.0, -9.0, -1.0) if b is None else (b["physics_val"], b["physics_reward"], b["surrogate"])

    @property
    def level(self):
        return self.status.get("level", 0)

    @property
    def level_met(self):
        b = self.status["best"]
        return bool(b and b["physics_val"] >= LEVELS[self.level][1] and b["surrogate"] >= MASTERED_SUR)

    @property
    def proficient(self):
        """Passed the first level: good enough for FlyLab to just perform the skill."""
        return self.level > 0 or self.level_met

    @property
    def optimal(self):
        return self.level == len(LEVELS) - 1 and self.level_met

    def candidate(self):
        if self.cand_path.exists():
            return np.load(self.cand_path)
        return self.brain.theta(self.task).copy()

    def save(self):
        _atomic_write(self.status_path, json.dumps(self.status, indent=2))

    def log(self, row):
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def rows(self):
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# one training round for one skill
# ---------------------------------------------------------------------------
def level_up(run: SkillRun, n_val, n_test, say):
    """Raise the bar: re-score the champion on the next level's larger arena sets."""
    st, b = run.status, run.status["best"]
    st["level"] = run.level + 1
    k = LEVELS[run.level][0]
    theta = np.load(run.dir / "best.npy")
    pv, prew, pdet = physics_eval(theta, run.task, n_val * k, val_seed(run.task))
    b.update(physics_val=pv, physics_reward=prew, physics_errors=pdet, n=n_val * k)
    pt, ptrew, ptdet = physics_eval(theta, run.task, n_test * k, test_seed(run.task))
    st["test"] = {"round": b["round"], "physics_test": pt, "physics_reward": ptrew,
                  "physics_errors": ptdet, "n": n_test * k}
    st["physics_runs"] += (n_val + n_test) * k
    st["stale"] = 0
    run.save()
    say(f"[{run.task}] LEVEL UP -> {run.level + 1}/{len(LEVELS)}: champion re-scored on "
        f"{n_val * k} val arenas {pv:.0%} | {n_test * k} test arenas {pt:.0%}")


def run_round(run: SkillRun, body, n_val, n_test, say):
    st = run.status
    while run.level_met and run.level < len(LEVELS) - 1:
        level_up(run, n_val, n_test, say)
        write_skill_report(run)
        write_summary()
    if run.optimal:
        return
    k = LEVELS[run.level][0]
    n_val, n_test = n_val * k, n_test * k
    st["round"] += 1
    r = st["round"]
    theta0 = run.candidate()
    sur0, _ = TR.evaluate(body, theta0, run.task, n=SUR_EVAL_N, seed=999)
    sigma, lr = TR.annealed(sur0)
    t0 = time.time()
    theta, log = TR.train(theta0, body, run.task, ROUND_GENS,
                          seed=T.TASKS.index(run.task) * 100_000 + r, sigma=sigma, lr=lr)
    sur, sur_rew = TR.evaluate(body, theta, run.task, n=SUR_EVAL_N, seed=999)
    pv, prew, pdet = physics_eval(theta, run.task, n_val, val_seed(run.task))
    st["generations"] += ROUND_GENS
    st["physics_runs"] += n_val
    key = (pv, prew, sur)
    improved = key > run.best_key
    row = {"round": r, "time": _now(), "generations": st["generations"],
           "sigma": round(sigma, 4), "lr": round(lr, 4),
           "surrogate_before": sur0, "surrogate": sur, "surrogate_reward": sur_rew,
           "physics_val": pv, "physics_reward": prew, "physics_errors": pdet,
           "improved": improved, "level": run.level + 1, "n_val": n_val,
           "seconds": round(time.time() - t0, 1)}
    np.save(run.cand_path, theta)
    if improved:
        run.brain.set_theta(run.task, theta)
        run.brain.save(run.task)
        np.save(run.dir / "best.npy", theta)
        st["best"] = {"round": r, "physics_val": pv, "physics_reward": prew,
                      "surrogate": sur, "physics_errors": pdet, "n": n_val, "time": _now()}
        st["stale"] = 0
        if n_test:
            pt, ptrew, ptdet = physics_eval(theta, run.task, n_test, test_seed(run.task))
            st["physics_runs"] += n_test
            st["test"] = {"round": r, "physics_test": pt, "physics_reward": ptrew,
                          "physics_errors": ptdet, "n": n_test}
            row["physics_test"] = pt
    else:
        st["stale"] += 1
        lineage = [x["surrogate"] for x in run.rows() if x["round"] > st.get("reseed_round", 0)] + [sur]
        fresh = st.get("reseed_round", 0) > st["best"]["round"] if st["best"] else False
        if len(lineage) >= PLATEAU and max(lineage) < MASTERED_SUR:
            rng = np.random.default_rng(T.TASKS.index(run.task) * 100_000 + 50_000 + r)
            np.save(run.cand_path, RX.seed_theta(run.task, body, rng))   # reseed a fresh lineage
            st["reseed_round"] = r
            st["stale"] = 0
            row["reseeded"] = True
        elif st["stale"] >= PATIENCE * (2 if fresh else 1) and (run.dir / "best.npy").exists():
            np.save(run.cand_path, np.load(run.dir / "best.npy"))   # restart from best
            st["stale"] = 0
            row["restarted_from_best"] = True
    run.log(row)
    run.save()
    _update_flylab_history(run)
    write_skill_report(run)
    write_summary()
    tag = "NEW BEST" if improved else f"no gain ({st['stale']}/{PATIENCE})"
    test = f" | test {st['test']['physics_test']:.0%}" if improved and st.get("test") else ""
    say(f"[{run.task}] round {r}: surrogate {sur0:.0%}->{sur:.0%} | physics val {pv:.0%} "
        f"(reward {prew:+.2f}){test} | {tag} | {row['seconds']:.0f}s")


def _update_flylab_history(run: SkillRun):
    """Let the FlyLab console know how good each brain is."""
    b = run.status["best"]
    if not b:
        return
    run.brain.reload(run.task)
    sk = run.brain.history["skills"].setdefault(run.task, {})
    sk["generations"] = max(sk.get("generations", 0), run.status["generations"])
    sk["surrogate_success"] = b["surrogate"]
    sk["physics_val"] = b["physics_val"]
    sk["physics_test"] = (run.status.get("test") or {}).get("physics_test")
    sk["mastered"] = run.proficient
    run.brain.history["skills"][run.task] = sk
    _atomic_write(run.brain.h_path, json.dumps(run.brain.history, indent=2))


# ---------------------------------------------------------------------------
# documentation: per-skill report + summary
# ---------------------------------------------------------------------------
def write_skill_report(run: SkillRun):
    rows = run.rows()
    if not rows:
        return
    st = run.status
    r = [x["round"] for x in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(r, [100 * x["surrogate"] for x in rows], "-o", ms=3, label="surrogate (96 noisy arenas)")
    ax.plot(r, [100 * x["physics_val"] for x in rows], "-s", ms=4, label="physics validation")
    best_so_far, best = [], -1
    for x in rows:
        best = max(best, x["physics_val"]); best_so_far.append(100 * best)
    ax.step(r, best_so_far, where="post", color="k", lw=1, alpha=0.6, label="best physics (deployed)")
    tests = [(x["round"], 100 * x["physics_test"]) for x in rows if "physics_test" in x]
    if tests:
        ax.plot(*zip(*tests), "*", ms=12, color="crimson", label="held-out physics test")
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3)
    ax.set_xlabel("training round (100 ES generations each)"); ax.set_ylabel("success (%)")
    ax.set_title(f"{T.PRETTY[run.task]}: continuous training")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(run.dir / "curve.png", dpi=120); plt.close(fig)

    b, t = st["best"], st.get("test")
    lines = [f"# {T.PRETTY[run.task]} (`{run.task}`) — training report", "",
             f"*Auto-generated by `train_brains.py`, last updated {_now()}.*", "",
             "![curve](curve.png)", "",
             "| | |", "|---|---|",
             f"| rounds | {st['round']} |",
             f"| ES generations | {st['generations']} |",
             f"| physics runs | {st['physics_runs']} |",
             f"| status | {_status_text(st)} |"]
    if b:
        lines += [f"| best brain | round {b['round']}: physics validation **{b['physics_val']:.0%}** "
                  f"on {b.get('n', 6)} arenas, surrogate {b['surrogate']:.0%} |"]
    if t:
        lines += [f"| held-out physics test | **{t['physics_test']:.0%}** on {t['n']} unseen arenas "
                  f"(errors: {t['physics_errors']}) |"]
    unit = {"turn": "deg from target heading (< 20 = success)",
            "odor_avoid": "mm gained away from source (> 6 = success)",
            "light_avoid": "mm gained away from source (> 6 = success)"}.get(
        run.task, "mm from target at the end (< 2 = success)")
    lines += ["", f"Physics error per arena: {unit}.", "",
              "| round | level | gens | sigma | surrogate | physics val | val errors | test | note |",
              "|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for x in rows[-40:]:
        note = "new best" if x["improved"] else ("restart from best" if x.get("restarted_from_best") else
                                                 "reseed: fresh weights" if x.get("reseeded") else "")
        test = f"{x['physics_test']:.0%}" if "physics_test" in x else ""
        lines.append(f"| {x['round']} | {x.get('level', 1)} | {x['generations']} | {x['sigma']:.3f} | {x['surrogate']:.0%} | "
                     f"{x['physics_val']:.0%} | {x['physics_errors']} | {test} | {note} |")
    _atomic_write(run.dir / "REPORT.md", "\n".join(lines) + "\n")


def _status_text(st):
    b, lvl = st.get("best") or {}, st.get("level", 0)
    met = bool(b) and b.get("physics_val", 0) >= LEVELS[lvl][1] and b.get("surrogate", 0) >= MASTERED_SUR
    if met and lvl == len(LEVELS) - 1:
        return "**optimal**"
    return f"level {lvl + 1}/{len(LEVELS)}" + (" (passed)" if met else "")


def write_summary():
    TRAIN_DIR.mkdir(exist_ok=True)
    rows = []
    for task in T.TASKS:
        p = TRAIN_DIR / task / "status.json"
        if not p.exists():
            continue
        st = json.loads(p.read_text())
        b, t = st.get("best") or {}, st.get("test") or {}
        rows.append(f"| [{T.PRETTY[task]}]({task}/REPORT.md) | {st['round']} | {st['generations']} | "
                    f"{st['physics_runs']} | {b.get('surrogate', 0):.0%} | "
                    f"{b.get('physics_val', 0):.0%} of {b.get('n', 6)} | "
                    f"{('%.0f%% of %d' % (100 * t['physics_test'], t['n'])) if t else '-'} | "
                    f"{_status_text(st)} |")
    text = "\n".join([
        "# FlyLab brains — continuous training log", "",
        f"*Auto-generated by `train_brains.py`, last updated {_now()}.*", "",
        "Every skill has its own brain and its own training process: surrogate ES "
        "rounds, physics validation in MuJoCo, keep-the-best selection, and a held-out "
        "physics test that is never used for selection. See each skill's report for "
        "its full round-by-round history. Mastery is progressive: passing a level "
        f"(100% physics validation, >=95% surrogate) doubles the validation and test "
        f"arena sets; a brain is **optimal** once it holds up at level {len(LEVELS)}.", "",
        "| skill | rounds | ES generations | physics runs | surrogate | physics val | **physics test** | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|", *rows, ""])
    _atomic_write(TRAIN_DIR / "TRAINING.md", text)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills", nargs="+", default=T.TASKS, choices=T.TASKS)
    ap.add_argument("--hours", type=float, default=0, help="time budget (0 = until all optimal)")
    ap.add_argument("--val", type=int, default=6, help="physics validation arenas per round")
    ap.add_argument("--test", type=int, default=8, help="held-out physics test arenas per new best")
    ap.add_argument("--max-rounds", type=int, default=200, help="stop a skill after this many rounds")
    args = ap.parse_args()

    body = SG.Body()
    brain = B.Brain(STATE)
    runs = {t: SkillRun(t, brain) for t in args.skills}
    deadline = time.time() + args.hours * 3600 if args.hours else None
    say = lambda s: print(f"{_now()}  {s}", flush=True)
    say(f"continuous training: {', '.join(args.skills)} | val {args.val} / test {args.test} arenas")

    while True:
        active = [r for r in runs.values() if not r.optimal and r.status["round"] < args.max_rounds]
        if not active:
            say("every selected skill is optimal or has reached --max-rounds. Done.")
            break
        if deadline and time.time() > deadline:
            say("time budget reached. Run again to continue where this left off.")
            break
        run = min(active, key=lambda r: (r.level, r.best_key, r.status["round"]))   # weakest first
        run_round(run, body, args.val, args.test, say)


if __name__ == "__main__":
    main()
