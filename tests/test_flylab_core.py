"""FlyLab (project 09): senses, task scoring, brain, surrogate body, statistics.

None of this needs MuJoCo. Several tests are deliberate tripwires: if they
fail, a change has altered what the trained brains sense, how they move or how
they are scored, which would silently invalidate every published score. Such a
change needs a TASK_VERSION bump and retraining, not a test edit.
"""
from __future__ import annotations

import numpy as np
import pytest

import brain as B
import surrogate as SG
import tasks as T
import train_brains as TB
import world as W
from conftest import FLYLAB


def _stim(S=3):
    return {"odor_on": np.ones(S), "odor_x": np.array([5.0, 0.0, -3.0]), "odor_y": np.array([2.0, 6.0, 1.0]),
            "light_on": np.ones(S), "light_x": np.array([0.0, 8.0, 2.0]), "light_y": np.array([-4.0, 1.0, 9.0]),
            "goal_on": np.array([1.0, 1.0, 0.0]), "goal_x": np.array([10.0, -2.0, 0.0]),
            "goal_y": np.array([0.0, 3.0, 0.0]), "goal_heading_only": np.zeros(S)}


def _one(**kw):
    """A single-fly stimulus dict with everything off unless given."""
    st = {k: np.zeros(1) for k in _stim()}
    st.update({k: np.atleast_1d(np.asarray(v, float)) for k, v in kw.items()})
    return st


# --- senses -------------------------------------------------------------------
def test_sense_vector_layout():
    assert W.N_SENSES == len(W.SENSE_NAMES) == 10
    f, prev = W.sense(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), _stim())
    assert f.shape == (1, 3, 10) and set(prev) == {"filt"}


def test_antennae_sit_at_measured_arista_offsets():
    (lx, ly), (rx, ry) = W.antenna_positions(0.0, 0.0, 0.0)
    assert (lx, ly) == pytest.approx((0.52, 0.19)) and (rx, ry) == pytest.approx((0.52, -0.19))
    (lx, ly), _ = W.antenna_positions(1.0, 2.0, np.pi / 2)        # facing +y: left is -x
    assert (lx, ly) == pytest.approx((1.0 - 0.19, 2.0 + 0.52))


def test_bilateral_contrast_points_to_the_stimulus_side():
    left = W.sense(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), _one(odor_on=1, odor_y=5.0))[0][0, 0]
    right = W.sense(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), _one(odor_on=1, odor_y=-5.0))[0][0, 0]
    assert left[1] > 0 > right[1] and left[1] == pytest.approx(-right[1])
    lamp = W.sense(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), _one(light_on=1, light_y=6.0))[0][0, 0]
    assert lamp[4] > 0 and lamp[3] > 0


def test_nearer_source_smells_and_looks_stronger():
    near = W.odor_intensity(0.0, 0.0, 2.0, 0.0)
    far = W.odor_intensity(0.0, 0.0, 12.0, 0.0)
    assert 1 > near > far > 0
    assert W.light_intensity(0, 0, 0, 0) == 1.0 and W.light_intensity(0, 0, 30, 0) == pytest.approx(0.5)


def test_wrap_range():
    a = np.linspace(-10, 10, 401)
    w = W.wrap(a)
    assert (w >= -np.pi).all() and (w < np.pi).all()
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)


def test_no_goal_never_looks_like_arrived():
    f = W.sense(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), _one())[0][0, 0]
    assert f[9] == 0.0 and f[8] == 0.0          # "goal known" off, distance 0
    g = W.sense(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), _one(goal_on=1, goal_x=0.01))[0][0, 0]
    assert g[9] == 1.0


GOLDEN_FIRST = [[1.2241594576972328, 0.15480743542289505, 0.0, 0.5274097828898889, -1.807313787548748, 0.0, 0.0,
                 1.0, 0.7615941559557649, 1.0],
                [1.18635274512466, 0.27737552111162056, 0.0, 0.9900626144339968, -1.5184774395017557, 0.0,
                 0.9923290240127155, 0.12362486846088669, 0.37180195173610736, 1.0],
                [1.660038649646058, -0.28587433559960046, 0.0, 0.013848657003944236, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
GOLDEN_SECOND = [[1.2342455115611428, 0.14843994147320114, 1.0086053863910038, 0.4674680729011413,
                  -1.7661696539782705, -3.0, -0.03172040488870056, 0.9987402862938443, 0.7564744415469395, 1.0],
                 [1.1753112795540437, 0.2706706652391765, -1.104146557061636, 0.9559356415437625,
                  -1.6098928805674557, -3.0, 0.9861936732172564, 0.15948778118800017, 0.38399784444442736, 1.0],
                 [1.6409769730990498, -0.3067376336472697, -1.9061676547008188, 0.013737714519652994, 0.0,
                  -0.011094248429124182, 0.0, 0.0, 0.0, 0.0]]


def test_tripwire_sensing_is_unchanged():
    """What the trained brains perceive, bit for bit (incl. the leaky integrator)."""
    x, y, th = np.array([[0.0, 1.0, -1.0]]), np.array([[0.0, 0.5, 2.0]]), np.array([[0.0, 1.0, -2.0]])
    f1, prev = W.sense(x, y, th, _stim())
    f2, _ = W.sense(x + 0.3, y - 0.2, th + 0.1, _stim(), prev)
    np.testing.assert_allclose(f1[0], GOLDEN_FIRST, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f2[0], GOLDEN_SECOND, rtol=1e-12, atol=1e-12)


# --- tasks: parser ----------------------------------------------------------------
@pytest.mark.parametrize("cmd, expected", [
    ("find the food", ("task", "odor_seek", {})),
    ("follow the smell", ("task", "odor_seek", {})),
    ("avoid the smell", ("task", "odor_avoid", {})),
    ("go to the light", ("task", "light_seek", {})),
    ("hide from the light", ("task", "light_avoid", {})),
    ("turn left 45", ("task", "turn", {"angle_deg": 45.0})),
    ("turn right", ("task", "turn", {"angle_deg": -90.0})),
    ("turn around", ("task", "turn", {"angle_deg": 180.0})),
    ("go to 10, 5", ("task", "goto", {"goal": (10.0, 5.0)})),
    ("walk forward 12", ("task", "walk_forward", {"distance": 12.0})),
    ("walk forward 100", ("task", "walk_forward", {"distance": 25.0})),   # clipped
    ("stop", ("stop", None, {})),
    ("help", ("meta", "help", {})),
    ("", ("unknown", None, {})),
    ("sing a song", ("unknown", None, {})),
])
def test_parse(cmd, expected):
    assert T.parse(cmd) == expected


def test_parse_practice():
    kind, word, params = T.parse("practice find the food 200")
    assert (kind, word, params["task"], params["_generations"]) == ("meta", "practice", "odor_seek", 200)


# --- tasks: scoring rules ------------------------------------------------------------
def _path(points, steps):
    """Piecewise-linear path through `points`, sampled into `steps` + 1 positions."""
    pts = np.asarray(points, float)
    s = np.linspace(0, len(pts) - 1, steps + 1)
    xs = np.interp(s, np.arange(len(pts)), pts[:, 0])
    ys = np.interp(s, np.arange(len(pts)), pts[:, 1])
    return xs[None, :], ys[None, :]


def _steps(task):
    return int(round(T.DURATION_S[task] / T.STEP_S))


def test_seek_success_needs_arriving_and_standing_still():
    scen = T.sample("goto", 1, np.random.default_rng(0), fixed={"goal": (6.0, 0.0)})
    n = _steps("goto")
    xs, ys = _path([(0, 0), (6, 0), (6, 0), (6, 0)], n)
    reward, ok = T.score(scen, xs, ys, np.zeros_like(xs))
    assert ok[0] and reward[0] > 1.5

    # orbiting the goal inside the success radius: arrives but never stops
    t = np.arange(n + 1) * T.STEP_S
    ox, oy = 6 + 1.0 * np.cos(8 * t), 1.0 * np.sin(8 * t)      # 8 mm/s on a 1 mm circle
    _, ok_orbit = T.score(scen, ox[None], oy[None], np.zeros((1, n + 1)))
    assert not ok_orbit[0]

    xs, ys = _path([(0, 0), (3, 0)], n)                        # stops short
    _, ok_short = T.score(scen, xs, ys, np.zeros_like(xs))
    assert not ok_short[0]


def test_escape_must_gain_distance_without_approaching():
    scen = T.sample("odor_avoid", 1, np.random.default_rng(0), fixed={"bearing": 0.0})   # source at (3.5, 0)
    n = _steps("odor_avoid")
    xs, ys = _path([(0, 0), (-10, 0)], n)                       # straight away
    _, ok = T.score(scen, xs, ys, np.zeros_like(xs))
    assert ok[0]
    xs, ys = _path([(0, 0), (1.0, 0), (-10, 0)], n)             # 1 mm toward it first (> 0.5 mm slack)
    _, ok = T.score(scen, xs, ys, np.zeros_like(xs))
    assert not ok[0]
    xs, ys = _path([(0, 0), (0, 3)], n)                         # moves, but gains < 6 mm
    _, ok = T.score(scen, xs, ys, np.zeros_like(xs))
    assert not ok[0]


def test_turn_success_needs_heading_and_little_drift():
    scen = T.sample("turn", 1, np.random.default_rng(0), fixed={"angle_deg": 90.0})
    n = _steps("turn")
    ths = np.linspace(0, np.deg2rad(85), n + 1)[None]
    xs, ys = np.zeros((1, n + 1)), np.zeros((1, n + 1))
    assert T.score(scen, xs, ys, ths)[1][0]                      # 5 deg error, no drift
    assert not T.score(scen, xs, ys, ths * 0.5)[1][0]            # 47 deg short
    far_x, far_y = _path([(0, 0), (7, 0)], n)
    assert not T.score(scen, far_x, far_y, ths)[1][0]           # wandered 7 mm


def test_sampled_arenas_follow_task_ranges():
    rng = np.random.default_rng(3)
    s = T.sample("odor_seek", 500, rng)
    r = np.hypot(s.stim["odor_x"], s.stim["odor_y"])
    assert r.min() >= 8 and r.max() <= 15 and s.stim["goal_on"].sum() == 0     # no compass for smell
    s = T.sample("light_avoid", 500, rng)
    r = np.hypot(s.stim["light_x"], s.stim["light_y"])
    assert r.min() >= 4 and r.max() <= 8
    s = T.sample("turn", 500, rng)
    assert (s.stim["goal_heading_only"] == 1).all()


def test_tripwire_task_rules_version():
    """Bump TASK_VERSION (and this test) only together with a deliberate rule change."""
    assert T.TASK_VERSION == {"walk_forward": 2, "turn": 1, "goto": 2, "odor_seek": 2,
                              "odor_avoid": 2, "light_seek": 3, "light_avoid": 3}
    assert (T.SUCCESS_RADIUS_MM, T.STOPPED_MM_S, T.STOP_WINDOW_S) == (1.5, 1.5, 0.5)
    assert (T.AVOID_GAIN_MM, T.AVOID_SLACK_MM, T.TURN_MAX_DISP_MM) == (6.0, 0.5, 6.0)


# --- brain -----------------------------------------------------------------------------
def test_checkpoints_match_the_network_shape():
    assert B.N_PARAMS == (W.N_SENSES + T.N_TASKS) * B.N_HID + B.N_HID + B.N_HID * B.N_OUT + B.N_OUT == 642
    for task in T.TASKS:
        theta = np.load(FLYLAB / "state" / "brains" / f"{task}.npy")
        assert theta.shape == (B.N_PARAMS,) and np.isfinite(theta).all(), task


def test_drives_stay_in_range_and_tiny_drives_halt():
    rng = np.random.default_rng(0)
    thetas = rng.normal(0, 3, (16, B.N_PARAMS))
    feats = rng.normal(0, 2, (16, 32, W.N_SENSES))
    d = B.act(thetas, feats, B.task_cue("goto", (16, 32)))
    assert d.shape == (16, 32, 2) and d.min() >= -0.5 - 1e-12 and d.max() <= 1.2 + 1e-12

    theta = np.zeros((1, B.N_PARAMS))
    assert B.act(theta, np.zeros((1, 1, 10)), B.task_cue("goto", (1, 1)))[0, 0] == pytest.approx([0.35, 0.35])
    theta[0, -2:] = np.arctanh((0.01 - B.DRIVE_CENTER) / B.DRIVE_SPAN)    # raw drive 0.01 -> gate closes
    assert (B.act(theta, np.zeros((1, 1, 10)), B.task_cue("goto", (1, 1))) == 0.0).all()


def test_unflatten_roundtrip():
    theta = np.arange(B.N_PARAMS, dtype=float)
    parts = B.unflatten(theta)
    assert [p.shape for p in parts] == B.SHAPES
    assert np.array_equal(np.concatenate([p.ravel() for p in parts]), theta)


# --- surrogate body -----------------------------------------------------------------------
def test_surrogate_reproduces_its_calibration_grid():
    body = SG.Body()
    assert body.calibrated and body.v_fwd.shape == (len(body.grid), len(body.grid))
    for i, dl in enumerate(body.grid):
        for j, dr in enumerate(body.grid):
            v, u, w = body.velocities(np.array(dl), np.array(dr))
            assert (v, u, w) == pytest.approx((body.v_fwd[i, j], body.v_lat[i, j], body.omega[i, j]))
    # a full stride is fast and nearly straight; halting really halts
    i12 = int(np.argmin(abs(body.grid - 1.2)))
    i0 = int(np.argmin(abs(body.grid)))
    assert body.v_fwd[i12, i12] > 10 and abs(np.degrees(body.omega[i12, i12])) < 20
    assert abs(body.v_fwd[i0, i0]) < 0.1


GOLDEN_ROLLOUT = {
    "goto": ([-0.685664303, -3.271526789, -0.982416677], [2.964268703, 2.783134016, 3.997823]),
    "odor_seek": ([-7.425474561, -11.481007384, -6.994322894], [3.117047421, 4.27441924, 6.510535448]),
    "light_avoid": ([16.151562711, 13.98572346, 19.613544097], [-13.653612851, 13.169300654, -0.654250436]),
}


@pytest.mark.parametrize("task", sorted(GOLDEN_ROLLOUT))
def test_tripwire_surrogate_rollout_is_unchanged(task):
    """Fixed synthetic brain, fixed arenas, noise off: senses + brain + body end to end."""
    theta = B.init_theta(np.random.default_rng(123))
    theta[-2:] = [0.4, 0.1]
    scen = T.sample(task, 3, np.random.default_rng(7))
    xs, ys, ths, drives = SG.rollout(SG.Body(), theta[None], scen, rng=None, noise=0.0)
    gx, gy = GOLDEN_ROLLOUT[task]
    np.testing.assert_allclose(xs[0, :, -1], gx, atol=1e-6)
    np.testing.assert_allclose(ys[0, :, -1], gy, atol=1e-6)
    assert drives.shape == (1, 3, _steps(task), 2)


def test_rollout_many_groups_by_task_and_is_repeatable():
    # (gait sway is drawn per batch, so a joint pass is not bitwise equal to separate
    # passes; what it guarantees is per-task truncation and determinism)
    theta = B.init_theta(np.random.default_rng(5))[None]
    body = SG.Body()
    scens = [T.sample("goto", 2, np.random.default_rng(1)), T.sample("turn", 3, np.random.default_rng(2))]
    a = SG.rollout_many(body, theta, scens, rng=None, noise=0.0)
    b = SG.rollout_many(body, theta, scens, rng=None, noise=0.0)
    for s, (xs, _ys, ths), (xs2, _, _) in zip(scens, a, b):
        assert xs.shape == (1, s.n, _steps(s.task) + 1)
        assert xs[0, :, 0].tolist() == s.x0.tolist() and ths[0, :, 0].tolist() == s.th0.tolist()
        np.testing.assert_array_equal(xs, xs2)


# --- statistics + optimal rule -------------------------------------------------------------
def test_wilson_interval_matches_published_table():
    lo, hi = TB.wilson(1.0, 32)
    assert (round(100 * lo), round(100 * hi)) == (89, 100)       # README: 100% of 32 (89%-100%)
    lo, hi = TB.wilson(31 / 32, 32)
    assert (round(100 * lo), round(100 * hi)) == (84, 99)        # README: 97% of 32 (84%-99%)
    assert TB.wilson(0.5, 0) == (0.0, 1.0)
    lo, hi = TB.wilson(0.0, 10)
    assert lo == 0.0 and 0 < hi < 0.35


def _status(level=2, val=1.0, sur=1.0, test=1.0, version=None, task="goto"):
    return {"task": task, "level": level, "task_version": version or T.TASK_VERSION[task],
            "best": {"physics_val": val, "surrogate": sur}, "test": {"physics_test": test, "n": 32}}


def test_optimal_requires_top_level_validation_surrogate_test_and_current_rules():
    assert TB.is_optimal(_status())
    assert TB.is_optimal(_status(val=0.95, test=0.90))
    assert not TB.is_optimal(_status(level=1))                   # not at the last level
    assert not TB.is_optimal(_status(val=0.94))                  # level-3 validation bar is 95%
    assert not TB.is_optimal(_status(sur=0.90))                  # surrogate mastery is 95%
    assert not TB.is_optimal(_status(test=0.875))                # held-out test bar is 90%
    assert not TB.is_optimal(_status(version=T.TASK_VERSION["goto"] - 1))   # scored under old rules
    assert TB.LEVELS == [(1, 1.0), (2, 1.0), (4, 0.95)] and TB.TEST_OPTIMAL == 0.90


def test_validation_and_test_arenas_never_overlap():
    for task in T.TASKS:
        assert TB.val_seed(task) != TB.test_seed(task)
    assert len({TB.val_seed(t) for t in T.TASKS} | {TB.test_seed(t) for t in T.TASKS}) == 2 * len(T.TASKS)
