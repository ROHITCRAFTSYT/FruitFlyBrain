"""
Bridge to the full physics fly: NeuroMechFly v2 in MuJoCo (FlyGym 2.x).

    brain (20 Hz)  ->  left/right descending drives
    HybridTurningController (1 kHz)  ->  CPGs + leg-coordination rules  ->  joint targets
    MuJoCo physics (10 kHz)  ->  42 actuated leg joints, adhesion, ground contact

The body is EPFL's NeuroMechFly v2 (Wang-Chen et al., Nature Methods 2024),
built from a micro-CT scan of a real adult female Drosophila. Its locomotion
controller is the official FlyGym turning controller (Apache-2.0).

Performance note: the leg controller is updated every 10 physics steps (1 kHz)
instead of every step; this preserves the gait (verified: 12.72 vs 12.73 mm
forward in 1 s) while running ~20x faster on a laptop CPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (HybridControllerObservation,
                                         HybridTurningController, LocomotionAction,
                                         PreprogrammedSteps, apply_locomotion_action,
                                         make_locomotion_fly)

import brain as B
import tasks as T
from surrogate import DT
from world import sense

CTRL_EVERY = 10                     # physics steps per controller update (1 kHz)
GEOM = mj.mjtGeom


def _yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_x(angle):
    return [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]


class PhysicsFly:
    """One NeuroMechFly in a flat arena, optionally with task markers + cameras."""

    def __init__(self, markers=(), overhead=None, record=False, res=(360, 480),
                 playback_speed=0.5, fps=25, colorize=True):
        self.fly = make_locomotion_fly(name="fly", add_adhesion=True, colorize=colorize)
        cams = []
        if record:
            cams.append(self.fly.add_tracking_camera(
                name="closeup", pos_offset=(-1.0, -9.0, 4.0),
                rotation=Rotation3D("euler", (1.2, 0.0, 0.0)), fovy=40.0))
        self.world = FlatGroundWorld()
        wb = self.world.mjcf_root.worldbody
        for mk in markers:
            self._add_marker(wb, **mk)
        if record and overhead is not None:
            cx, cy, h = overhead
            tilt = np.deg2rad(28)
            cams.insert(0, wb.add_camera(name="overhead", pos=[cx, cy - h * np.tan(tilt), h],
                                         quat=_quat_x(tilt), fovy=45.0))
        self.world.add_fly(self.fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                           bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                           add_ground_contact_sensors=False)
        self.sim = Simulation(self.world)
        self.cam_names = []
        if record:
            self.sim.set_renderer(cams, camera_res=res, playback_speed=playback_speed,
                                  output_fps=fps)
            self.cam_names = list(self.sim.renderer.frames.keys())
        self.steps = PreprogrammedSteps()
        self.dofs = self.fly.get_actuated_jointdofs_order("position")
        self.ctrl = HybridTurningController(timestep=self.sim.timestep * CTRL_EVERY,
                                            preprogrammed_steps=self.steps,
                                            output_dof_order=self.dofs)
        segs = self.fly.get_bodysegs_order()
        self.i_thorax = segs.index(BodySegment("c_thorax"))
        self.i_arista = (segs.index(BodySegment("l_arista")), segs.index(BodySegment("r_arista")))
        self.record = record
        self.reset()

    @staticmethod
    def _add_marker(wb, kind, x, y):
        if kind == "food":        # a ripe drop of banana: yellow-green, glossy
            wb.add_geom(type=GEOM.mjGEOM_SPHERE, size=[0.9, 0, 0], pos=[x, y, 0.9],
                        rgba=[0.75, 0.9, 0.1, 1], contype=0, conaffinity=0)
            wb.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[3.0, 0.01, 0], pos=[x, y, 0.02],
                        rgba=[0.6, 0.85, 0.2, 0.25], contype=0, conaffinity=0)
        elif kind == "repellent":
            wb.add_geom(type=GEOM.mjGEOM_SPHERE, size=[0.9, 0, 0], pos=[x, y, 0.9],
                        rgba=[0.55, 0.1, 0.6, 1], contype=0, conaffinity=0)
            wb.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[4.0, 0.01, 0], pos=[x, y, 0.02],
                        rgba=[0.55, 0.1, 0.6, 0.25], contype=0, conaffinity=0)
        elif kind == "light":     # a glowing lamp on a thin post
            wb.add_geom(type=GEOM.mjGEOM_SPHERE, size=[1.1, 0, 0], pos=[x, y, 4.0],
                        rgba=[1.0, 0.95, 0.4, 1], contype=0, conaffinity=0)
            wb.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.12, 2.0, 0], pos=[x, y, 2.0],
                        rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
            wb.add_light(pos=[x, y, 5.0], dir=[0, 0, -1], diffuse=[0.9, 0.85, 0.5],
                         castshadow=False, cutoff=60)
        elif kind == "goal":      # a red flag
            wb.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[2.0, 0.01, 0], pos=[x, y, 0.02],
                        rgba=[0.9, 0.2, 0.2, 0.35], contype=0, conaffinity=0)
            wb.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.1, 1.8, 0], pos=[x, y, 1.8],
                        rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0)
            wb.add_geom(type=GEOM.mjGEOM_BOX, size=[0.8, 0.02, 0.5], pos=[x + 0.8, y, 3.1],
                        rgba=[0.9, 0.15, 0.15, 1], contype=0, conaffinity=0)

    # -- basic control -------------------------------------------------------
    def reset(self):
        self.sim.reset()
        self.ctrl.reset(seed=0)
        apply_locomotion_action(self.sim, self.fly.name, LocomotionAction(
            joint_angles=self.steps.default_pose_by_dof_order(self.dofs),
            adhesion_onoff=np.ones(6, dtype=bool)))
        self.sim.warmup()
        self.n = 0

    def pose(self):
        p = self.sim.get_body_positions(self.fly.name)[self.i_thorax]
        q = np.asarray(self.sim.get_body_rotations(self.fly.name))[self.i_thorax]
        return float(p[0]), float(p[1]), float(_yaw_from_quat(q))

    def advance(self, drive, seconds):
        """Hold a descending drive [L, R] for `seconds` of physics."""
        drive = np.asarray(drive, dtype=float)
        for _ in range(int(round(seconds / self.sim.timestep))):
            if self.n % CTRL_EVERY == 0:
                obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
                apply_locomotion_action(self.sim, self.fly.name, self.ctrl.step(drive, obs))
            self.sim.step()
            if self.record:
                self.sim.render_as_needed()
            self.n += 1

    def antenna_offsets(self):
        """Measured arista positions in the thorax frame (forward, left) in mm."""
        P = self.sim.get_body_positions(self.fly.name)
        x, y, th = self.pose()
        out = []
        for i in self.i_arista:
            dx, dy = P[i][0] - x, P[i][1] - y
            out.append((dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)))
        return out


# ---------------------------------------------------------------------------
# Calibration: measure how the real body moves for each pair of drives
# ---------------------------------------------------------------------------
def calibrate(grid=(-0.5, 0.0, 0.15, 0.3, 0.6, 0.9, 1.2), trial_s=1.3, skip_s=0.4, log=print):
    fly = PhysicsFly(record=False, colorize=False)
    g = np.asarray(grid, float)
    n = len(g)
    v_fwd = np.zeros((n, n)); v_lat = np.zeros((n, n)); omega = np.zeros((n, n))
    step_t = 0.05
    t_start = time.time()
    for i, dl in enumerate(g):
        for j, dr in enumerate(g):
            fly.reset()
            fly.advance([dl, dr], skip_s)
            xs, ys, ths = [], [], []
            for _ in range(int(round((trial_s - skip_s) / step_t))):
                x, y, th = fly.pose(); xs.append(x); ys.append(y); ths.append(th)
                fly.advance([dl, dr], step_t)
            xs, ys, ths = map(np.asarray, (xs, ys, ths))
            ths = np.unwrap(ths)
            dx, dy = np.diff(xs), np.diff(ys)
            th_mid = 0.5 * (ths[1:] + ths[:-1])
            fwd = dx * np.cos(th_mid) + dy * np.sin(th_mid)
            lat = -dx * np.sin(th_mid) + dy * np.cos(th_mid)
            dur = step_t * len(dx)
            v_fwd[i, j] = fwd.sum() / dur
            v_lat[i, j] = lat.sum() / dur
            omega[i, j] = (ths[-1] - ths[0]) / dur
            log(f"  drive L={dl:+.1f} R={dr:+.1f}:  forward {v_fwd[i, j]:6.2f} mm/s   "
                f"side {v_lat[i, j]:+5.2f} mm/s   turn {np.degrees(omega[i, j]):+7.1f} deg/s")
    # Step-response lag of the CPG amplitudes ([0,0] -> [1,1]).
    fly.reset(); fly.advance([0.0, 0.0], 0.2)
    x0, y0, _ = fly.pose(); dist = []
    for _ in range(20):
        fly.advance([1.0, 1.0], 0.025)
        x, y, _ = fly.pose(); dist.append(np.hypot(x - x0, y - y0))
    speed = np.gradient(np.asarray(dist), 0.025)
    target = 0.63 * np.median(speed[-6:])
    k = int(np.argmax(speed >= target)) if np.any(speed >= target) else len(speed) - 1
    tau = float(np.clip((k + 1) * 0.025, 0.05, 0.4))
    antenna = fly.antenna_offsets()
    log(f"  CPG lag tau = {tau:.3f} s   |   arista offsets (fwd, left) mm: "
        f"{np.round(antenna, 2).tolist()}   |   {time.time() - t_start:.0f} s total")
    return {"grid": g, "v_fwd": v_fwd, "v_lat": v_lat, "omega": omega, "tau": tau,
            "antenna": np.asarray(antenna)}


# ---------------------------------------------------------------------------
# Deployment: run a trained brain inside the physics fly
# ---------------------------------------------------------------------------
def markers_for(scen: T.Scenario):
    t = scen.task
    if t in ("odor_seek",):
        return [dict(kind="food", x=float(scen.stim["odor_x"][0]), y=float(scen.stim["odor_y"][0]))]
    if t == "odor_avoid":
        return [dict(kind="repellent", x=float(scen.stim["odor_x"][0]), y=float(scen.stim["odor_y"][0]))]
    if t in ("light_seek", "light_avoid"):
        return [dict(kind="light", x=float(scen.stim["light_x"][0]), y=float(scen.stim["light_y"][0]))]
    if t in ("goto", "walk_forward"):
        return [dict(kind="goal", x=float(scen.stim["goal_x"][0]), y=float(scen.stim["goal_y"][0]))]
    return []


def overhead_for(scen: T.Scenario):
    if scen.task == "turn":
        return (0.0, 0.0, 22.0)
    if "target" in scen.extra:
        tx, ty = float(scen.extra["target"][0][0]), float(scen.extra["target"][1][0])
    else:
        tx, ty = 0.0, 0.0
    span = np.hypot(tx, ty)
    far = scen.task in ("odor_avoid", "light_avoid")
    cx, cy = (0.0, 0.0) if far else (tx / 2, ty / 2)
    h = max(24.0, 1.35 * span + (26.0 if far else 12.0))
    return (cx, cy, h)


def deploy(theta: np.ndarray, scen: T.Scenario, record=True, log=print):
    """Run the brain in the physics fly for the scenario's duration (S must be 1)."""
    pf = PhysicsFly(markers=markers_for(scen), overhead=overhead_for(scen), record=record)
    steps = int(round(scen.duration / DT))
    cue = B.task_cue(scen.task, (1, 1))
    xs, ys, ths, drives, feats_log = [], [], [], [], []
    prev = None
    t0 = time.time()
    for k in range(steps):
        x, y, th = pf.pose()
        xs.append(x); ys.append(y); ths.append(th)
        stim = {key: val[:1] for key, val in scen.stim.items()}
        feats, prev = sense(np.array([[x]]), np.array([[y]]), np.array([[th]]), stim, prev)
        d = B.act(theta[None], feats, cue)[0, 0]
        drives.append(d); feats_log.append(feats[0, 0])
        pf.advance(d, DT)
        if log and (k + 1) % 20 == 0:
            log(f"    t={(k + 1) * DT:4.1f}s  pos=({x:6.2f},{y:6.2f})mm  "
                f"heading={np.degrees(th):6.1f}  drive L={d[0]:+.2f} R={d[1]:+.2f}")
    x, y, th = pf.pose()
    xs.append(x); ys.append(y); ths.append(th)
    traj = {k: np.asarray(v) for k, v in
            dict(x=xs, y=ys, th=np.unwrap(ths), drives=drives, senses=feats_log).items()}
    traj["wall_seconds"] = time.time() - t0
    frames = {}
    if record:
        frames = {name: list(pf.sim.renderer.frames[name]) for name in pf.cam_names}
    pf.sim.close() if hasattr(pf.sim, "close") else None
    return traj, frames
