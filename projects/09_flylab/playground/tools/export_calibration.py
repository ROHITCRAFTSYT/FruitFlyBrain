"""Export the calibrated body (state/calibration.npz) to playground/calibration.json.

Re-run after calibrate.py changes the calibration:
    ..\\..\\..\\..\\.venv-sim\\Scripts\\python export_calibration.py
"""
import json
from pathlib import Path

import numpy as np

FLYLAB = Path(__file__).resolve().parents[2]
cal = np.load(FLYLAB / "state" / "calibration.npz")
out = {k: cal[k].tolist() for k in ("grid", "v_fwd", "v_lat", "omega")}
out["tau"] = float(cal["tau"])
out["units"] = "grid: descending drive; v_fwd, v_lat: mm/s; omega: rad/s; tau: s"
(FLYLAB / "playground" / "calibration.json").write_text(json.dumps(out))
print("wrote playground/calibration.json")
