"""
Fast surrogate body: how the fly MOVES in response to its descending drives.

Full NeuroMechFly physics runs at ~10,000 steps per simulated second -- far too
slow to train a brain on a laptop CPU. So we measure the real physics once
(calibrate.py): for a grid of left/right drives we record the fly's forward
speed, sideways slip and turning rate. This module replays those measured
responses as a lightweight kinematic model that simulates thousands of flies in
parallel in milliseconds. The brain trains here, then is deployed back into the
full physics fly (physics.py) -- a sim-to-sim transfer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import brain as B
import tasks as T
from world import sense

DT = 0.05                     # brain / surrogate control step (20 Hz)
DR_GAIN = 0.2                 # domain randomization: +/-20% speed and turn gain
DR_VEER = 0.25                # rad/s constant veer while walking (std)
# Gait wobble measured on the physics fly at full stride (physics.py / README):
# thorax position oscillates ~0.15 mm rms and yaw ~3.6 deg per step. It is
# oscillatory (not a random walk), so it perturbs what the fly SENSES each step
# without accumulating into its path.
DR_JITTER_MM = 0.15
DR_YAW_WOBBLE = np.deg2rad(4.0)
STRIDE_DRIVE = 0.6            # |drive| of the "full stride" the wobble was measured at
CALIB_PATH = Path(__file__).resolve().parent / "state" / "calibration.npz"


class Body:
    def __init__(self, path: Path = CALIB_PATH):
        if path.exists():
            d = np.load(path)
            self.grid = d["grid"]
            self.v_fwd, self.v_lat, self.omega = d["v_fwd"], d["v_lat"], d["omega"]
            self.tau = float(d["tau"])
            self.calibrated = True
        else:
            # Placeholder until calibrate.py has measured the real fly.
            g = np.linspace(-0.5, 1.2, 6)
            L, R = np.meshgrid(g, g, indexing="ij")
            self.grid = g
            self.v_fwd = 9.0 * (L + R) / 2
            self.v_lat = np.zeros_like(L)
            self.omega = 2.5 * (R - L)
            self.tau = 0.15
            self.calibrated = False

    def _interp(self, table, dl, dr):
        g = self.grid
        fi = np.interp(dl, g, np.arange(len(g)))
        fj = np.interp(dr, g, np.arange(len(g)))
        i0 = np.clip(np.floor(fi).astype(int), 0, len(g) - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, len(g) - 2)
        a, b = fi - i0, fj - j0
        return ((1 - a) * (1 - b) * table[i0, j0] + a * (1 - b) * table[i0 + 1, j0]
                + (1 - a) * b * table[i0, j0 + 1] + a * b * table[i0 + 1, j0 + 1])

    def velocities(self, dl, dr):
        return (self._interp(self.v_fwd, dl, dr),
                self._interp(self.v_lat, dl, dr),
                self._interp(self.omega, dl, dr))


def rollout(body: Body, theta: np.ndarray, scen: T.Scenario,
            rng: np.random.Generator | None = None, noise: float = 1.0):
    """
    Simulate P brains x S scenarios in parallel.
    theta [P, N_PARAMS]. Returns xs, ys, ths [P, S, T+1] and drives [P, S, T, 2].
    """
    return _run(body, theta, scen.x0, scen.y0, scen.th0, scen.stim,
                B.task_cue(scen.task, (scen.n,)), int(round(scen.duration / DT)), rng, noise)


def rollout_many(body: Body, theta: np.ndarray, scens: list,
                 rng: np.random.Generator | None = None, noise: float = 1.0):
    """
    Simulate several task scenarios (e.g. a new lesson + rehearsed skills) in ONE
    vectorized pass. Returns a list of (xs, ys, ths) per scenario group, each
    truncated to that task's own duration.
    """
    x0 = np.concatenate([s.x0 for s in scens])
    y0 = np.concatenate([s.y0 for s in scens])
    th0 = np.concatenate([s.th0 for s in scens])
    stim = {k: np.concatenate([s.stim[k] for s in scens]) for k in scens[0].stim}
    cue = np.concatenate([B.task_cue(s.task, (s.n,)) for s in scens])
    steps = [int(round(s.duration / DT)) for s in scens]
    xs, ys, ths, _ = _run(body, theta, x0, y0, th0, stim, cue, max(steps), rng, noise)
    out, i = [], 0
    for s, n in zip(scens, steps):
        out.append((xs[:, i:i + s.n, :n + 1], ys[:, i:i + s.n, :n + 1], ths[:, i:i + s.n, :n + 1]))
        i += s.n
    return out


def _run(body, theta, x0, y0, th0, stim, cue_s, steps, rng, noise):
    P, S = theta.shape[0], len(x0)
    x = np.broadcast_to(x0, (P, S)).copy()
    y = np.broadcast_to(y0, (P, S)).copy()
    th = np.broadcast_to(th0, (P, S)).copy()
    d_eff = np.zeros((P, S, 2))
    cue = np.broadcast_to(cue_s, (P, S, cue_s.shape[-1]))
    xs = np.empty((P, S, steps + 1)); ys = np.empty_like(xs); ths = np.empty_like(xs)
    drives = np.empty((P, S, steps, 2))
    xs[..., 0], ys[..., 0], ths[..., 0] = x, y, th
    prev = None
    # Domain randomization: every training arena gets a slightly different body
    # (speed/turn gain, a constant veer, response lag, gait jitter). Shared across
    # the population (common random numbers) so ES compares brains fairly. This
    # forces robust closed-loop control that survives the jump to real physics.
    if rng is not None and noise > 0:
        g_v = rng.uniform(1 - DR_GAIN, 1 + DR_GAIN, S)
        g_w = rng.uniform(1 - DR_GAIN, 1 + DR_GAIN, S)
        veer = rng.normal(0, DR_VEER, S)
        alpha = DT / np.clip(body.tau * rng.uniform(0.7, 2.0, S), DT, None)
        alpha = alpha[None, :, None]
        jitter = DR_JITTER_MM
    else:
        g_v = g_w = 1.0
        veer = 0.0
        alpha = DT / body.tau
        jitter = 0.0
    # Common random numbers: every noise draw has shape (1, S) and is shared by
    # all P brains, so fitness differences come from the brains, not the dice.
    # Gait sway is part of the body, not domain randomization: it is always in
    # the recorded thorax path (a fixed seed keeps noise-free rollouts
    # repeatable), so "standing still" means the legs actually stopped.
    sway_rng = rng if rng is not None else np.random.default_rng(0)
    sway_x = np.zeros((P, S)); sway_y = np.zeros((P, S)); wob = np.zeros((P, S))
    for t in range(steps):
        if jitter:
            feats, prev = sense(x + sway_x, y + sway_y,
                                th + wob * sway_rng.normal(0, DR_YAW_WOBBLE, (1, S)), stim, prev)
        else:
            feats, prev = sense(x, y, th, stim, prev)
        if rng is not None and noise > 0:
            feats = feats + rng.normal(0, 0.05 * noise, (1, S, feats.shape[-1]))
        d_cmd = B.act(theta, feats, cue)
        drives[:, :, t] = d_cmd
        d_eff += alpha * (d_cmd - d_eff)
        v, u, w = body.velocities(d_eff[..., 0], d_eff[..., 1])
        v, w = v * g_v, w * g_w + veer * (np.abs(v) > 0.5)
        if rng is not None and noise > 0:
            v = v * (1 + rng.normal(0, 0.08 * noise, (1, S)))
            w = w + rng.normal(0, 0.4 * noise, (1, S))
        c, s = np.cos(th), np.sin(th)
        x = x + (v * c - u * s) * DT
        y = y + (v * s + u * c) * DT
        th = th + w * DT
        # Sway follows how hard the legs are stepping, not net speed: stepping
        # in place (one side backward) goes nowhere but rocks the thorax at
        # ~2 mm/s in physics, the same as 0.4 x a full stride's wobble.
        wob = np.clip(np.abs(d_eff).mean(axis=-1) / STRIDE_DRIVE, 0, 1.3)
        sway_x = wob * sway_rng.normal(0, DR_JITTER_MM, (1, S))
        sway_y = wob * sway_rng.normal(0, DR_JITTER_MM, (1, S))
        xs[..., t + 1], ys[..., t + 1], ths[..., t + 1] = x + sway_x, y + sway_y, th
    return xs, ys, ths, drives
