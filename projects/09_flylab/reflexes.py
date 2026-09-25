"""
Innate reflexes: hard-wired starting behaviors that learning then refines.

Real flies don't learn chemotaxis from scratch; they are born steering up a
smell gradient with their two antennae. Some skills here need the same head
start: a random brain discovers that CIRCLING the food earns almost full
reward (the body's tightest turning circle, ~1.6 mm, fits inside the 2 mm
goal zone) and evolution never climbs out of that basin to "walk up and stop".

For such skills a reseed doesn't start from random weights. It clones a simple
reflex into the network (behavior cloning on the reflex's own trajectories),
and ES + physics selection then improve it like any other brain.
"""
from __future__ import annotations

import numpy as np

import brain as B
import surrogate as SG
import tasks as T

CLONE_ARENAS = 512
CLONE_STEPS = 3000


def odor_taxis(feats):
    """Steer by the left-right smell difference, slow down as the smell gets
    stronger, stop on the source. feats = world.sense() features."""
    intensity = np.clip(feats[..., 0] / 2.0, 1e-3, 1.0)         # feature is 2 x exp(-r / 10 mm)
    dist = -10.0 * np.log(intensity)                             # mm to the source (estimate)
    fwd = np.clip(0.12 * dist, 0.1, 1.0)
    turn = np.clip(feats[..., 1], -0.6, 0.6)
    d = np.stack([fwd - turn, fwd + turn], axis=-1)
    return np.clip(d, -0.5, 1.2) * (dist > 0.5)[..., None]


REFLEXES = {"odor_seek": odor_taxis}


def _gate(d):
    return d * np.clip((np.abs(d) - B.DRIVE_GATE_LO) / (B.DRIVE_GATE_HI - B.DRIVE_GATE_LO), 0.0, 1.0)


def clone(task: str, body: SG.Body, rng: np.random.Generator) -> np.ndarray:
    """Weights that reproduce the task's reflex (least squares on its own rollouts)."""
    reflex, xs, ys = REFLEXES[task], [], []

    def act(theta, feats, cue):
        d = reflex(feats)
        xs.append(np.concatenate([feats, cue], axis=-1)[0]); ys.append(d[0])
        return _gate(d)

    real_act, B.act = B.act, act
    try:
        SG.rollout(body, np.zeros((1, B.N_PARAMS)), T.sample(task, CLONE_ARENAS, rng), rng, noise=1.0)
    finally:
        B.act = real_act
    X, Y = np.concatenate(xs), np.concatenate(ys)

    theta = B.init_theta(rng) * 0.3
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    for it in range(1, CLONE_STEPS + 1):
        idx = rng.integers(0, len(X), 4096)
        x, y = X[idx], Y[idx]
        W1, b1, W2, b2 = (a[0] for a in B.unflatten(theta[None]))
        h = np.tanh(x @ W1 + b1)
        t = np.tanh(h @ W2 + b2)
        g_o = 2 * (B.DRIVE_CENTER + B.DRIVE_SPAN * t - y) / len(x) * B.DRIVE_SPAN * (1 - t ** 2)
        g_h = g_o @ W2.T * (1 - h ** 2)
        g = np.concatenate([(x.T @ g_h).ravel(), g_h.sum(0), (h.T @ g_o).ravel(), g_o.sum(0)])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta -= 0.003 * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return theta


def seed_theta(task: str, body: SG.Body, rng: np.random.Generator) -> np.ndarray:
    """Starting weights for a fresh lineage: the skill's reflex if it has one, else random."""
    return clone(task, body, rng) if task in REFLEXES else B.init_theta(rng)
