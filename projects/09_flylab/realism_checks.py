"""
Re-measure the body facts FlyLab's docs and senses rely on, in NeuroMechFly physics.

Read-only: it builds its own PhysicsFly, never touches training/, state/ or a
running trainer, and takes a few seconds of CPU (about 2 s of simulated time).

    ..\\..\\.venv-sim\\Scripts\\python realism_checks.py

Reference values (FlyGym 2.1.0 / MuJoCo 3.9, measured 2026-11-05):
    timestep 1e-4 s (10 kHz), controller every 10 steps (1 kHz), 42 actuated DoFs
    aristae (forward, left) 0.517 / +-0.188 mm        world.py uses 0.52 / 0.19
    full-stride (1.2, 1.2) speed 16.3 mm/s             calibration table 16.2
    gait yaw peak 11.8 Hz (CPG 12 Hz), yaw 4.4 deg rms surrogate uses 4 deg
    eye bodies 0.26 mm ahead of the thorax, 1.17 mm above ground
"""
from __future__ import annotations

import time

import numpy as np

import physics as PH
import world as W


def body_frame(pf: PH.PhysicsFly, index: int) -> tuple[float, float, float]:
    """(forward, left, height) of body segment `index` relative to the thorax, in mm."""
    P = np.asarray(pf.sim.get_body_positions(pf.fly.name))
    x, y, th = pf.pose()
    dx, dy = P[index][0] - x, P[index][1] - y
    return (float(dx * np.cos(th) + dy * np.sin(th)), float(-dx * np.sin(th) + dy * np.cos(th)), float(P[index][2]))


def main() -> None:
    t0 = time.time()
    pf = PH.PhysicsFly(record=False, colorize=False)
    names = [str(s.name) for s in pf.fly.get_bodysegs_order()]
    print(f"timestep {pf.sim.timestep:g} s, controller every {PH.CTRL_EVERY} steps, "
          f"{len(pf.dofs)} actuated DoFs")

    ar = [body_frame(pf, names.index(n)) for n in ("l_arista", "r_arista")]
    print(f"aristae (fwd, left) mm: {np.round([a[:2] for a in ar], 3).tolist()}   "
          f"world.py: ({W.ANTENNA_FORWARD}, +-{W.ANTENNA_HALF_SPAN})")
    eyes = [body_frame(pf, names.index(n)) for n in ("l_eye", "r_eye")]
    eye_fwd, eye_z = float(np.mean([e[0] for e in eyes])), float(np.mean([e[2] for e in eyes]))
    print(f"eyes {eye_fwd:.2f} mm ahead of the thorax, {eye_z:.2f} mm above ground -> the sensed lamp "
          f"is {eye_z + W.LIGHT_HEIGHT_MM:.2f} mm up (the film marker is drawn at 4.0 mm)")
    for r in (1.0, 2.0, 5.0):
        a = np.deg2rad(60.0)
        sx, sy = r * np.cos(a), r * np.sin(a)
        print(f"  light {r:.0f} mm away at 60 deg (from the thorax) is at "
              f"{np.degrees(np.arctan2(sy, sx - eye_fwd)):.1f} deg from the eyes")

    xs, ys, ths = [], [], []
    for _ in range(1500):                       # 1.5 s at full stride, sampled at 1 kHz
        pf.advance([1.2, 1.2], 0.001)
        x, y, th = pf.pose()
        xs.append(x); ys.append(y); ths.append(th)
    xs, ys, ths = np.asarray(xs), np.asarray(ys), np.unwrap(ths)
    k = 400                                     # skip the start-up transient
    speed = np.hypot(xs[-1] - xs[k], ys[-1] - ys[k]) / ((len(xs) - 1 - k) * 0.001)
    yaw = ths[k:] - np.polyval(np.polyfit(np.arange(len(ths) - k), ths[k:], 1), np.arange(len(ths) - k))
    f = np.fft.rfftfreq(len(yaw), 0.001)
    spec = np.abs(np.fft.rfft(yaw)); spec[f < 2] = 0
    print(f"full stride: {speed:.1f} mm/s, gait yaw peak {f[np.argmax(spec)]:.1f} Hz, "
          f"yaw wobble {np.degrees(yaw.std()):.2f} deg rms")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
