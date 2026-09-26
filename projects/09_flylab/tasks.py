"""
The fly's task library + the plain-English command parser.

Every task knows how to:
  * sample randomized training scenarios (so skills generalize),
  * build the one specific scenario you asked for (for the physics run),
  * score a trajectory (reward for learning, success for reporting).

Seek/avoid tasks give the brain NO goal direction -- the fly has to use its
antennae (odor) or eyes (light), exactly like a real fly doing chemotaxis or
phototaxis. Navigation tasks (go to / walk / turn) provide a goal direction,
like the fly's internal compass / path integration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from world import wrap

SUCCESS_RADIUS_MM = 1.5  # "arrived": thorax within ~0.6 body lengths of the target
STOPPED_MM_S = 1.5       # ...and actually standing there (real flies stop to feed)
STOP_WINDOW_S = 0.5
AVOID_GAIN_MM = 6.0      # escapes must gain this much distance...
AVOID_SLACK_MM = 0.5     # ...without ever approaching the source by more than this
TURN_MAX_DISP_MM = 6.0   # a "turn" must not wander off (the gait can't pivot perfectly)
STEP_S = 0.05            # trajectories are sampled at the 20 Hz brain rate
# Bumped whenever a task's definition of success changes; the trainer then
# re-scores the deployed brain under the new rules before training on.
# Light v3: the lamp moved overhead (world.LIGHT_HEIGHT_MM), so scores taken
# with the old eye-level point light no longer describe the brain.
TASK_VERSION = {"walk_forward": 2, "turn": 1, "goto": 2, "odor_seek": 2,
                "odor_avoid": 2, "light_seek": 3, "light_avoid": 3}

TASKS = ["walk_forward", "turn", "goto", "odor_seek", "odor_avoid",
         "light_seek", "light_avoid"]
N_TASKS = len(TASKS)

PRETTY = {
    "walk_forward": "Walk forward",
    "turn": "Turn in place",
    "goto": "Go to a location",
    "odor_seek": "Find food by smell",
    "odor_avoid": "Escape a bad smell",
    "light_seek": "Walk toward light",
    "light_avoid": "Hide from light",
}

DURATION_S = {
    "walk_forward": 5.0, "turn": 3.0, "goto": 7.0, "odor_seek": 8.0,
    "odor_avoid": 4.0, "light_seek": 7.0, "light_avoid": 4.0,
}


def _empty(S):
    z = np.zeros(S)
    return {"odor_on": z.copy(), "odor_x": z.copy(), "odor_y": z.copy(),
            "light_on": z.copy(), "light_x": z.copy(), "light_y": z.copy(),
            "goal_on": z.copy(), "goal_x": z.copy(), "goal_y": z.copy(),
            "goal_heading_only": z.copy()}


def _polar(r, a):
    return r * np.cos(a), r * np.sin(a)


@dataclass
class Scenario:
    task: str
    x0: np.ndarray
    y0: np.ndarray
    th0: np.ndarray
    stim: dict
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.x0)

    @property
    def duration(self):
        return DURATION_S[self.task]


def sample(task: str, S: int, rng: np.random.Generator, fixed: dict | None = None) -> Scenario:
    """Randomized scenarios for training; `fixed` pins parameters for a real run."""
    fixed = fixed or {}
    x0 = np.zeros(S); y0 = np.zeros(S)
    th0 = rng.uniform(-np.pi, np.pi, S) if not fixed else np.zeros(S)
    st = _empty(S)
    extra = {}

    if task == "walk_forward":
        D = np.full(S, fixed["distance"]) if "distance" in fixed else rng.uniform(5, 15, S)
        st["goal_on"][:] = 1
        st["goal_x"], st["goal_y"] = x0 + D * np.cos(th0), y0 + D * np.sin(th0)
        extra["target"] = (st["goal_x"], st["goal_y"])

    elif task == "turn":
        if "angle_deg" in fixed:
            A = np.full(S, np.deg2rad(fixed["angle_deg"]))
        else:
            A = rng.choice([-1, 1], S) * rng.uniform(np.deg2rad(45), np.pi, S)
        target_heading = wrap(th0 + A)
        st["goal_on"][:] = 1
        st["goal_heading_only"][:] = 1
        st["goal_x"], st["goal_y"] = _polar(1000.0, target_heading)
        extra["target_heading"] = target_heading

    elif task == "goto":
        if "goal" in fixed:
            gx, gy = fixed["goal"]
            st["goal_x"][:], st["goal_y"][:] = gx, gy
        else:
            st["goal_x"], st["goal_y"] = _polar(rng.uniform(6, 16, S), rng.uniform(-np.pi, np.pi, S))
        st["goal_on"][:] = 1
        extra["target"] = (st["goal_x"], st["goal_y"])

    elif task in ("odor_seek", "odor_avoid"):
        if fixed:
            r = np.full(S, 11.0 if task == "odor_seek" else 3.5)
        else:
            r = rng.uniform(8, 15, S) if task == "odor_seek" else rng.uniform(2.5, 5, S)
        a = rng.uniform(-np.pi, np.pi, S) if not fixed else np.full(S, fixed.get("bearing", np.deg2rad(60)))
        st["odor_on"][:] = 1
        st["odor_x"], st["odor_y"] = _polar(r, a)
        extra["target"] = (st["odor_x"], st["odor_y"])

    elif task in ("light_seek", "light_avoid"):
        if fixed:
            r = np.full(S, 12.0 if task == "light_seek" else 5.0)
        else:
            r = rng.uniform(8, 16, S) if task == "light_seek" else rng.uniform(4, 8, S)
        a = rng.uniform(-np.pi, np.pi, S) if not fixed else np.full(S, fixed.get("bearing", np.deg2rad(-70)))
        st["light_on"][:] = 1
        st["light_x"], st["light_y"] = _polar(r, a)
        extra["target"] = (st["light_x"], st["light_y"])
    else:
        raise ValueError(task)

    return Scenario(task, x0, y0, th0, st, extra)


def score(scen: Scenario, xs, ys, ths):
    """
    xs, ys, ths: arrays [..., S, T+1] (leading dims = population).
    Returns (reward [..., S], success [..., S] bool).
    """
    t = scen.task
    xf, yf, thf = xs[..., -1], ys[..., -1], ths[..., -1]
    if t == "turn":
        err = np.abs(wrap(thf - scen.extra["target_heading"]))
        disp = np.hypot(xf - xs[..., 0], yf - ys[..., 0])
        reward = 1.0 - err / np.pi - 0.6 * np.minimum(disp / TURN_MAX_DISP_MM, 1.5)
        return reward, (err < np.deg2rad(20)) & (disp < TURN_MAX_DISP_MM)

    tx, ty = scen.extra["target"]
    d0 = np.hypot(xs[..., 0] - tx, ys[..., 0] - ty)
    df = np.hypot(xf - tx, yf - ty)
    d = np.hypot(xs - tx[..., None], ys - ty[..., None])
    if t in ("odor_avoid", "light_avoid"):
        # A blind straight sprint used to pass ~100% (the source starts 2.5-8 mm
        # away in a random direction and 4 s of sprinting covers 60 mm). A real
        # escape has to sense the source: it may not walk toward it first.
        gain = df - d0
        approach = np.maximum(d0 - d.min(axis=-1), 0.0)
        reward = np.clip(gain / 8.0, -1, 1.5) - np.clip(approach / 2.0, 0, 1.5)
        return reward, (gain > AVOID_GAIN_MM) & (approach <= AVOID_SLACK_MM)

    # go-to style: walk_forward, goto, odor_seek, light_seek
    # Reward arriving AND staying still: dwell only counts while the fly is
    # slow, because the body's ~1.6 mm tightest circle fits inside the goal
    # zone and brains learned to orbit the target instead of stopping on it.
    speed = np.hypot(np.diff(xs, axis=-1), np.diff(ys, axis=-1)) / STEP_S
    inside = d[..., 1:] < SUCCESS_RADIUS_MM
    still = np.exp(-speed / STOPPED_MM_S)
    tail = int(0.6 * speed.shape[-1])
    dwell = (inside * still)[..., tail:].mean(axis=-1)
    progress = np.clip((d0 - df) / d0, -1, 1)
    reward = progress + 1.0 * dwell + 0.3 * inside.any(axis=-1)
    stopped = speed[..., -int(round(STOP_WINDOW_S / STEP_S)):].mean(axis=-1) < STOPPED_MM_S
    return reward, (df < SUCCESS_RADIUS_MM) & stopped


# ---------------------------------------------------------------------------
# Plain-English command parser
# ---------------------------------------------------------------------------
ODOR_WORDS = r"(food|odou?r|smell|scent|banana|fruit|vinegar|yeast|sugar|apple)"
BAD_ODOR_WORDS = r"(odou?r|smell|scent|repellent|poison|smoke|bad smell|danger|vinegar)"
LIGHT_WORDS = r"(light|sun|lamp|bright|brightness)"
NUM = r"(-?\d+(?:\.\d+)?)"


def parse(cmd: str):
    """
    Returns one of:
      ("task", task_name, fixed_params)
      ("stop", None, {})
      ("meta", keyword, {})   keyword in help/skills/quit/reset/demo/show/practice
      ("unknown", None, {})
    """
    c = cmd.strip().lower()
    if not c:
        return ("unknown", None, {})
    if c in ("help", "?", "h"):
        return ("meta", "help", {})
    if c in ("quit", "exit", "q", "bye"):
        return ("meta", "quit", {})
    if c in ("skills", "status", "progress", "stats"):
        return ("meta", "skills", {})
    if c.startswith("reset"):
        return ("meta", "reset", {})
    if c.startswith("demo"):
        return ("meta", "demo", {})
    if c in ("show", "video", "replay", "open"):
        return ("meta", "show", {})
    if re.search(r"\b(sleep|nap|consolidate|dream)\b", c):
        return ("meta", "sleep", {})
    m = re.match(r"^(practice|train)\s+(.+?)(?:\s+(\d+))?$", c)
    if m:
        inner = parse(m.group(2))
        if inner[0] == "task":
            params = dict(inner[2]); params["_generations"] = int(m.group(3) or 150)
            return ("meta", "practice", {"task": inner[1], **params})

    if re.search(r"\b(stop|stand still|freeze|rest|halt|stay)\b", c):
        return ("stop", None, {})

    avoid = re.search(r"\b(avoid|escape|flee|run away|hide|away from|get away)\b", c)
    if re.search(LIGHT_WORDS, c):
        if avoid or "dark" in c or "shade" in c:
            return ("task", "light_avoid", {})
        return ("task", "light_seek", {})
    if avoid and re.search(BAD_ODOR_WORDS, c):
        return ("task", "odor_avoid", {})
    if re.search(ODOR_WORDS, c) or re.search(r"\b(eat|hungry|forage|sniff)\b", c):
        return ("task", "odor_seek", {})

    m = re.search(r"turn(?:\s+to\s+the)?\s+(left|right)(?:\s+(?:by\s+)?" + NUM + r")?", c)
    if m:
        ang = float(m.group(2) or 90)
        return ("task", "turn", {"angle_deg": ang if m.group(1) == "left" else -ang})
    if re.search(r"turn\s*around|u-?turn|about face", c):
        return ("task", "turn", {"angle_deg": 180.0})

    m = re.search(r"(?:go|walk|move|head)\s+to\s*\(?\s*" + NUM + r"\s*[, ]\s*" + NUM, c)
    if m:
        return ("task", "goto", {"goal": (float(m.group(1)), float(m.group(2)))})

    m = re.search(r"(?:walk|go|move|run)\s+(?:straight\s+)?(?:forward|ahead|straight)(?:\s+" + NUM + r")?", c)
    if m or re.fullmatch(r"(walk|forward|go forward|move forward)", c):
        dist = float(m.group(1)) if (m and m.group(1)) else 10.0
        return ("task", "walk_forward", {"distance": max(2.0, min(dist, 25.0))})

    return ("unknown", None, {})
