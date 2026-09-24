"""
Training one skill brain with Evolution Strategies (OpenAI-ES).

Each generation perturbs the brain's weights in many random directions
(antithetic pairs), lets every perturbed brain attempt the task in the
calibrated, domain-randomized surrogate body, and moves the weights toward the
perturbations that did better (centered-rank fitness shaping + Adam).

Common random numbers: all candidate brains in a generation face the same
arenas, bodies and noise, so fitness differences come from the brains, not luck.
Annealing: exploration noise (sigma) and step size shrink as the skill improves,
so a good brain is polished rather than shaken.
"""
from __future__ import annotations

import time

import numpy as np

import surrogate as SG
import tasks as T

POP_PAIRS = 16          # 32 perturbed brains per generation
SIGMA = 0.08
LR = 0.03
WEIGHT_DECAY = 0.005
SCENARIOS = 16


def _ranks(f):
    """Centered rank transform -> robust to reward scale."""
    r = np.empty_like(f)
    r[np.argsort(f)] = np.arange(len(f))
    return r / (len(f) - 1) - 0.5


def evaluate(body, theta, task, n=48, seed=123):
    """Success rate and mean reward of one brain on n random (noisy) arenas."""
    rng = np.random.default_rng(seed)
    scen = T.sample(task, n, rng)
    xs, ys, ths, _ = SG.rollout(body, theta[None], scen, rng, noise=1.0)
    rew, ok = T.score(scen, xs, ys, ths)
    return float(ok.mean()), float(rew.mean())


def annealed(success: float):
    """Exploration noise and learning rate for a brain at this success level."""
    k = 1.0 - 0.7 * np.clip(success, 0, 1)          # 1.0 -> 0.3 as success -> 100%
    return SIGMA * k, LR * k


def train(theta: np.ndarray, body: SG.Body, task: str, generations: int,
          seed: int | None = None, sigma: float = SIGMA, lr: float = LR,
          progress=None, eval_every: int = 10):
    """Train one skill brain. Returns (new_theta, log)."""
    rng = np.random.default_rng(seed if seed is not None else int(time.time()))
    theta = np.asarray(theta, dtype=float).copy()
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    before = evaluate(body, theta, task)
    curve = [(0, before[0], before[1])]
    t0 = time.time()

    for g in range(1, generations + 1):
        noise = rng.normal(0, 1, (POP_PAIRS, theta.size))
        pop = np.concatenate([theta + sigma * noise, theta - sigma * noise])  # [2K, N]
        seed_g = int(rng.integers(1 << 31))
        scen = T.sample(task, SCENARIOS, np.random.default_rng(seed_g))
        xs, ys, ths, _ = SG.rollout(body, pop, scen, np.random.default_rng(seed_g + 1))
        shaped = _ranks(T.score(scen, xs, ys, ths)[0].mean(axis=1))
        grad = -(shaped[:POP_PAIRS] - shaped[POP_PAIRS:]) @ noise / (2 * POP_PAIRS * sigma)
        grad = grad + WEIGHT_DECAY * theta           # minimize negative fitness
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        theta = theta - lr * (m / (1 - b1 ** g)) / (np.sqrt(v / (1 - b2 ** g)) + eps)

        if g % eval_every == 0 or g == generations:
            sr, mr = evaluate(body, theta, task)
            curve.append((g, sr, mr))
            if progress:
                progress(g, generations, sr, mr)

    after = evaluate(body, theta, task)
    return theta, {"task": task, "generations": generations,
                   "success_before": before[0], "success_after": after[0],
                   "reward_before": before[1], "reward_after": after[1],
                   "curve": curve, "sigma": sigma, "lr": lr,
                   "seconds": round(time.time() - t0, 1)}
