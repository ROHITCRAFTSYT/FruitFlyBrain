"""
Stimulus fields and the fly's senses.

Both the fast surrogate and the full NeuroMechFly physics simulation call the
SAME functions here to turn a pose (x, y, heading) into sensory input. Identical
sensing on both sides is what lets a brain trained in the surrogate transfer to
the physics fly.

Units: millimetres, radians, seconds. Heading 0 = +x, counter-clockwise positive.
"""
from __future__ import annotations

import numpy as np

# Where the sense organs sit relative to the thorax (mm, body frame).
# Measured on the NeuroMechFly model: the left/right aristae sit 0.52 mm ahead
# of the thorax origin and 0.19 mm to either side (see calibrate.py output).
ANTENNA_FORWARD = 0.52
ANTENNA_HALF_SPAN = 0.19
EYE_AXIS = np.deg2rad(30.0)  # each compound eye's preferred direction, +/- from heading

ODOR_DECAY_MM = 10.0         # odor intensity ~ exp(-r / 10 mm)
LIGHT_SCALE_MM = 30.0        # light intensity ~ 1 / (1 + (r/30)^2)
REACH_RADIUS_MM = 1.5        # "arrived" if the thorax is within this of a target
SENSE_BETA = 0.4             # leaky sensory integration per 50 ms step (tau ~ 0.1 s)


def odor_intensity(px, py, sx, sy):
    r = np.hypot(px - sx, py - sy)
    return np.exp(-r / ODOR_DECAY_MM)


def light_intensity(px, py, sx, sy):
    r = np.hypot(px - sx, py - sy)
    return 1.0 / (1.0 + (r / LIGHT_SCALE_MM) ** 2)


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def antenna_positions(x, y, th):
    """World positions of the left and right antennae."""
    c, s = np.cos(th), np.sin(th)
    hx, hy = x + ANTENNA_FORWARD * c, y + ANTENNA_FORWARD * s
    # left = +90 deg from heading
    lx, ly = hx - ANTENNA_HALF_SPAN * s, hy + ANTENNA_HALF_SPAN * c
    rx, ry = hx + ANTENNA_HALF_SPAN * s, hy - ANTENNA_HALF_SPAN * c
    return (lx, ly), (rx, ry)


def sense(x, y, th, stim, prev=None):
    """
    Compute the brain's sensory vector for a batch of flies.

    `stim` is a dict of arrays (broadcastable to x):
        odor_on, odor_x, odor_y     odor source (odor_on = 0/1)
        light_on, light_x, light_y  light source
        goal_on, goal_x, goal_y     navigation goal (a point to reach / face)
    `prev` is the previous step's raw intensities (for temporal change), or None.

    Returns (features [..., 9], raw) where raw holds intensities for the next call.
    """
    (lx, ly), (rx, ry) = antenna_positions(x, y, th)

    # --- olfaction: two antennae --------------------------------------------
    oL = odor_intensity(lx, ly, stim["odor_x"], stim["odor_y"]) * stim["odor_on"]
    oR = odor_intensity(rx, ry, stim["odor_x"], stim["odor_y"]) * stim["odor_on"]
    o_mean = 0.5 * (oL + oR)
    o_contrast = (oL - oR) / (oL + oR + 1e-9)

    # --- vision: two compound eyes, cosine-tuned to +/-30 deg ---------------
    I = light_intensity(x, y, stim["light_x"], stim["light_y"]) * stim["light_on"]
    bearing_light = wrap(np.arctan2(stim["light_y"] - y, stim["light_x"] - x) - th)
    vL = I * np.maximum(0.0, np.cos(bearing_light - EYE_AXIS))
    vR = I * np.maximum(0.0, np.cos(bearing_light + EYE_AXIS))
    v_mean = 0.5 * (vL + vR)
    v_contrast = (vL - vR) / (vL + vR + 1e-6)

    # --- goal direction (path integration / "where I want to go") -----------
    gdx, gdy = stim["goal_x"] - x, stim["goal_y"] - y
    g_bear = wrap(np.arctan2(gdy, gdx) - th)
    # heading-only goals ("face this way") report distance 0: "you're already
    # here, just orient" -- so turning reuses the stop-on-arrival behavior.
    g_dist = np.hypot(gdx, gdy) * (1 - stim.get("goal_heading_only", 0))
    g_on = stim["goal_on"]

    # Sensory integration: a leaky integrator (~0.1 s at 20 Hz) smooths the
    # step-by-step wobble of the walking body, like the temporal filtering of
    # real olfactory and visual neurons. Temporal change is taken on the
    # smoothed signal.
    now = np.stack([o_mean, o_contrast, v_mean, v_contrast,
                    np.sin(g_bear) * g_on, np.cos(g_bear) * g_on,
                    np.tanh(g_dist / 10.0) * g_on], axis=-1)
    if prev is None:
        filt = now
        d_o = np.zeros_like(o_mean)
        d_v = np.zeros_like(v_mean)
    else:
        filt = prev["filt"] + SENSE_BETA * (now - prev["filt"])
        d_o = filt[..., 0] - prev["filt"][..., 0]
        d_v = filt[..., 2] - prev["filt"][..., 2]

    feats = np.stack([
        filt[..., 0] * 2.0,
        filt[..., 1] * 20.0,          # tiny bilateral differences, amplified
        np.clip(d_o * 200.0, -3, 3),  # is the smell getting stronger?
        filt[..., 2] * 2.0,
        filt[..., 3] * 2.0,
        np.clip(d_v * 200.0, -3, 3),
        filt[..., 4],
        filt[..., 5],
        filt[..., 6],
        np.broadcast_to(g_on, filt[..., 6].shape),   # "do I have a goal?" -- so "no goal"
    ], axis=-1)                                      # never looks like "arrived"
    return feats, {"filt": filt}


N_SENSES = 10
SENSE_NAMES = ["odor", "odor L-R", "odor dt", "light", "light L-R", "light dt",
               "goal sin", "goal cos", "goal dist", "goal known"]
