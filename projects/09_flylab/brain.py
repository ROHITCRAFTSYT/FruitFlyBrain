"""
The fly's brains: one small neural network PER SKILL, from senses to descending
commands.

    senses (10) + task cue (7)  ->  hidden layer (32, tanh)  ->  2 descending drives

The two outputs are the LEFT and RIGHT descending drives -- the same interface
real flies use, where descending neurons carry brain commands to the ventral
nerve cord that patterns the legs. Positive drive = that side's legs step
forward, larger = faster; negative = that side steps backward (spot turns).

Why one brain per skill? An earlier version shared ONE network across all
skills (with rehearsal and a "sleep" consolidation step). Measured in physics,
skills still drifted: walk-forward went from a 0.5 mm stop right after its
lesson to 1/5 physics success after eight more lessons. Separate brains --
loosely like dedicated circuits for different behaviors -- cannot overwrite each
other, so every skill can be trained to its own best independently.

Each brain's weights live in one flat vector so Evolution Strategies can train
them, and every brain is saved to disk (state/brains/<skill>.npy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tasks import N_TASKS, TASKS
from world import N_SENSES

N_IN = N_SENSES + N_TASKS
N_HID = 32
N_OUT = 2
DRIVE_CENTER, DRIVE_SPAN = 0.35, 0.85       # drive in [-0.5, 1.2]
DRIVE_GATE_LO, DRIVE_GATE_HI = 0.02, 0.10  # |drive|<0.02 -> exactly 0; full strength from 0.10

SHAPES = [(N_IN, N_HID), (N_HID,), (N_HID, N_OUT), (N_OUT,)]
N_PARAMS = int(sum(np.prod(s) for s in SHAPES))


def unflatten(theta: np.ndarray):
    """theta [..., N_PARAMS] -> list of arrays with leading batch dims."""
    lead = theta.shape[:-1]
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s))
        out.append(theta[..., i:i + n].reshape(*lead, *s))
        i += n
    return out


def task_cue(task: str, shape) -> np.ndarray:
    cue = np.zeros((*shape, N_TASKS))
    cue[..., TASKS.index(task)] = 1.0
    return cue


def act(theta: np.ndarray, senses: np.ndarray, cue: np.ndarray) -> np.ndarray:
    """
    theta  [P, N_PARAMS]  (P brains, e.g. an ES population; P=1 for deployment)
    senses [P, S, N_SENSES], cue [P, S, N_TASKS]
    returns drives [P, S, 2]
    """
    W1, b1, W2, b2 = unflatten(theta)
    x = np.concatenate([senses, cue], axis=-1)
    h = np.tanh(np.einsum("psi,pih->psh", x, W1) + b1[:, None, :])
    o = np.einsum("psh,pho->pso", h, W2) + b2[:, None, :]
    d = DRIVE_CENTER + DRIVE_SPAN * np.tanh(o)
    # Motor gate: tiny drives become a clean "halt" (without it, near-zero
    # commands make the physics legs shuffle and the fly creeps off its goal).
    # A ramp instead of a hard deadband keeps small weight changes visible to
    # the optimizer, so learning never stalls on a flat "standing still" plateau.
    return d * np.clip((np.abs(d) - DRIVE_GATE_LO) / (DRIVE_GATE_HI - DRIVE_GATE_LO), 0.0, 1.0)


def init_theta(rng: np.random.Generator) -> np.ndarray:
    W1 = rng.normal(0, 1 / np.sqrt(N_IN), (N_IN, N_HID))
    W2 = rng.normal(0, 0.1 / np.sqrt(N_HID), (N_HID, N_OUT))
    return np.concatenate([W1.ravel(), np.zeros(N_HID), W2.ravel(), np.zeros(N_OUT)])


class Brain:
    """The fly's set of per-skill brains + shared history, persisted on disk."""

    def __init__(self, state_dir: Path, seed: int = 0):
        self.dir = Path(state_dir)
        self.bdir = self.dir / "brains"
        self.bdir.mkdir(parents=True, exist_ok=True)
        self.h_path = self.dir / "brain_history.json"
        self.seed = seed
        self.history = json.loads(self.h_path.read_text()) if self.h_path.exists() else {}
        self.history.setdefault("lessons", [])
        self.history.setdefault("skills", {})
        self.history.setdefault("generations_total", 0)
        self._theta = {}

    def weights_path(self, task: str) -> Path:
        return self.bdir / f"{task}.npy"

    def theta(self, task: str) -> np.ndarray:
        if task not in self._theta:
            p = self.weights_path(task)
            legacy = self.dir / "brain_weights.npy"      # older single shared brain
            if p.exists():
                self._theta[task] = np.load(p)
            elif legacy.exists() and np.load(legacy).size == N_PARAMS:
                self._theta[task] = np.load(legacy)      # warm-start from what it already knew
            else:
                self._theta[task] = init_theta(np.random.default_rng(self.seed + TASKS.index(task)))
        return self._theta[task]

    def set_theta(self, task: str, theta: np.ndarray):
        self._theta[task] = np.asarray(theta, dtype=float)

    def reload(self, task: str):
        """Re-read a brain from disk (e.g. after the continuous trainer improved it)."""
        self._theta.pop(task, None)
        if self.h_path.exists():
            self.history = json.loads(self.h_path.read_text())

    def save(self, task: str | None = None):
        for t in ([task] if task else list(self._theta)):
            np.save(self.weights_path(t), self._theta[t])
        self.h_path.write_text(json.dumps(self.history, indent=2))

    def reset(self):
        for p in self.bdir.glob("*.npy"):
            p.unlink()
        self._theta = {}
        self.history = {"lessons": [], "skills": {}, "generations_total": 0}
        self.h_path.write_text(json.dumps(self.history, indent=2))

    def known_skills(self):
        return [t for t, s in self.history["skills"].items() if s.get("generations", 0) > 0]
