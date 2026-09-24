"""
How well do the skills the brain learned in the surrogate transfer to the full
NeuroMechFly physics fly?  Runs the current brain on N random arenas per skill
in MuJoCo (no rendering) and compares with the surrogate on the same arenas.

    ..\\..\\.venv-sim\\Scripts\\python evaluate_physics.py            # 5 arenas per skill
    ..\\..\\.venv-sim\\Scripts\\python evaluate_physics.py --n 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import brain as B
import physics
import surrogate as SG
import tasks as T

HERE = Path(__file__).resolve().parent


def random_deploy_scenario(task, rng):
    """A random arena, re-expressed with the fly starting at the origin facing +x
    (the physics fly always spawns there)."""
    s = T.sample(task, 1, rng)
    th0 = float(s.th0[0])
    c, si = np.cos(-th0), np.sin(-th0)

    def rot(x, y):
        return c * x - si * y, si * x + c * y
    st = {k: v.copy() for k, v in s.stim.items()}
    for key in ("odor", "light", "goal"):
        st[f"{key}_x"], st[f"{key}_y"] = rot(st[f"{key}_x"], st[f"{key}_y"])
    extra = dict(s.extra)
    if "target" in extra:
        extra["target"] = rot(*extra["target"])
    if "target_heading" in extra:
        extra["target_heading"] = T.wrap(extra["target_heading"] - th0)
    return T.Scenario(task, np.zeros(1), np.zeros(1), np.zeros(1), st, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="random arenas per skill")
    ap.add_argument("--weights", default=None,
                    help="one shared weight file (legacy); default = each skill's own brain")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    brains = B.Brain(HERE / "state")
    shared = np.load(args.weights) if args.weights else None
    body = SG.Body()
    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for task in T.TASKS:
        theta = shared if shared is not None else brains.theta(task)
        ok_p, ok_s = [], []
        for _ in range(args.n):
            scen = random_deploy_scenario(task, rng)
            sx, sy, sth, _ = SG.rollout(body, theta[None], scen, rng=None, noise=0)
            ok_s.append(bool(T.score(scen, sx, sy, sth)[1][0, 0]))
            traj, _ = physics.deploy(theta, scen, record=False, log=None)
            ok_p.append(bool(T.score(scen, traj["x"][None], traj["y"][None], traj["th"][None])[1][0]))
        rows.append({"task": task, "skill": T.PRETTY[task], "n": args.n,
                     "surrogate_success": float(np.mean(ok_s)),
                     "physics_success": float(np.mean(ok_p))})
        print(f"  {T.PRETTY[task]:<22} surrogate {np.mean(ok_s):4.0%}   physics {np.mean(ok_p):4.0%}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    overall = np.mean([r["physics_success"] for r in rows])
    print(f"  {'OVERALL':<22} physics {overall:4.0%} over {args.n * len(T.TASKS)} runs")
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "physics_transfer.json").write_text(
        json.dumps({"n_per_skill": args.n, "overall_physics_success": overall, "skills": rows}, indent=2))


if __name__ == "__main__":
    main()
