"""Record reference runs from the Python training code for the browser parity test.

For every trained brain, on one fixed and one random arena, this steps the exact
functions used in training (world.sense, brain.act, surrogate.Body) with noise
off and records every sensory vector, drive and pose, plus tasks.score's verdict.
It first checks that this loop reproduces surrogate.rollout's drives.

    ..\\..\\..\\..\\.venv-sim\\Scripts\\python make_reference.py   ->  reference.json
    node check_core.mjs                                            ->  compares flybrain.js
"""
import json
import sys
from pathlib import Path

import numpy as np

FLYLAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FLYLAB))
import brain as B          # noqa: E402
import surrogate as SG     # noqa: E402
import tasks as T          # noqa: E402
from world import sense    # noqa: E402

body = SG.Body()
cases = []
for task in T.TASKS:
    theta = np.load(FLYLAB / "state" / "brains" / f"{task}.npy")
    for kind, scen in (("fixed", T.sample(task, 1, np.random.default_rng(0), fixed={"deploy": True})),
                       ("random", T.sample(task, 1, np.random.default_rng(7 + T.TASKS.index(task))))):
        steps = int(round(scen.duration / SG.DT))
        x, y, th = float(scen.x0[0]), float(scen.y0[0]), float(scen.th0[0])
        stim = {k: v[:1] for k, v in scen.stim.items()}
        cue = B.task_cue(task, (1, 1))
        d_eff, prev, alpha = np.zeros(2), None, SG.DT / body.tau
        feats_log, drives_log, pose = [], [], [[x, y, th]]
        for _ in range(steps):
            f, prev = sense(np.array([[x]]), np.array([[y]]), np.array([[th]]), stim, prev)
            d = B.act(theta[None], f, cue)[0, 0]
            feats_log.append(f[0, 0].tolist()); drives_log.append(d.tolist())
            d_eff += alpha * (d - d_eff)
            v, u, w = body.velocities(d_eff[0], d_eff[1])
            c, s = np.cos(th), np.sin(th)
            x, y, th = x + (v * c - u * s) * SG.DT, y + (v * s + u * c) * SG.DT, th + w * SG.DT
            pose.append([float(x), float(y), float(th)])
        # the loop above must match the official vectorized rollout
        _, _, _, ref_drives = SG.rollout(body, theta[None], scen, rng=None, noise=0)
        assert np.allclose(ref_drives[0, 0], np.array(drives_log), atol=1e-12), f"loop != rollout for {task}"
        P = np.array(pose)
        _, ok = T.score(scen, P[:, 0][None, None], P[:, 1][None, None], P[:, 2][None, None])
        arena = {"task": task, "x0": float(scen.x0[0]), "y0": float(scen.y0[0]), "th0": float(scen.th0[0]),
                 "stim": {k: float(v[0]) for k, v in scen.stim.items()}}
        if "target_heading" in scen.extra:
            arena["targetHeading"] = float(scen.extra["target_heading"][0])
        cases.append({"task": task, "kind": kind, "arena": arena, "feats": feats_log,
                      "drives": drives_log, "pose": pose, "success": bool(ok[0, 0])})
        print(f"{task:<13} {kind:<6} steps={steps:<4} python verdict={'pass' if ok[0, 0] else 'fail'}")

(Path(__file__).parent / "reference.json").write_text(json.dumps({"cases": cases}))
print(f"wrote tools/reference.json ({len(cases)} runs)")
