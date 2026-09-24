"""
Measure how the real NeuroMechFly body moves for each pair of descending drives,
and save it as the surrogate body's lookup table (state/calibration.npz).

Run once (FlyLab also runs it automatically on first launch):
    ..\\..\\.venv-sim\\Scripts\\python calibrate.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
OUT = HERE / "outputs"


def plot_calibration(c, path):
    g = c["grid"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, key, title, cmap, scale, unit in [
        (ax[0], "v_fwd", "Forward speed", "viridis", 1, "mm/s"),
        (ax[1], "omega", "Turning rate", "coolwarm", 180 / np.pi, "deg/s"),
        (ax[2], "v_lat", "Sideways slip", "PuOr", 1, "mm/s"),
    ]:
        data = c[key] * scale
        lim = np.abs(data).max() if cmap != "viridis" else None
        mid = (g[1:] + g[:-1]) / 2                       # uneven grid -> explicit cell edges
        edges = np.concatenate([[g[0] - (mid[0] - g[0])], mid, [g[-1] + (g[-1] - mid[-1])]])
        im = a.pcolormesh(edges, edges, data.T, cmap=cmap, shading="flat",
                          vmin=-lim if lim else None, vmax=lim)
        a.set_xticks(g); a.set_yticks(g)
        a.set_xlabel("LEFT descending drive"); a.set_ylabel("RIGHT descending drive")
        a.set_title(f"{title} ({unit})")
        fig.colorbar(im, ax=a)
    fig.suptitle("How the NeuroMechFly body responds to its brain's two descending drives "
                 "(measured in MuJoCo physics)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    import physics
    STATE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    print("Calibrating the surrogate body against NeuroMechFly physics "
          "(49 left/right drive combinations, ~3 minutes)...")
    c = physics.calibrate()
    np.savez(STATE / "calibration.npz", **c)
    plot_calibration(c, OUT / "body_calibration.png")
    print(f"Saved {STATE / 'calibration.npz'} and {OUT / 'body_calibration.png'}")


if __name__ == "__main__":
    main()
