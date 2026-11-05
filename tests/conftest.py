"""Shared test setup: import paths, a stand-in for the MuJoCo bridge, tiny fixtures.

No test downloads data or needs FlyGym/MuJoCo:
  * projects/NN_name/*.py scripts are loaded by file path (their folder names
    start with digits, so they are not importable as modules);
  * FlyLab's `physics` module (NeuroMechFly in MuJoCo) is replaced by a stub
    when FlyGym is not installed, so the trainer/scoring code still imports;
  * hemibrain / FlyWire inputs come from tests/fixtures/.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
FLYLAB = PROJECTS / "09_flylab"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

for p in (PROJECTS, FLYLAB, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:  # the real bridge needs flygym + mujoco; CI does not install them
    import flygym  # noqa: F401
except ImportError:
    stub = types.ModuleType("physics")

    def _no_physics(*_args, **_kwargs):
        raise RuntimeError("MuJoCo physics is not available in the test environment")

    stub.deploy = _no_physics
    stub.calibrate = _no_physics
    stub.PhysicsFly = _no_physics
    sys.modules["physics"] = stub


def load_script(relpath: str) -> types.ModuleType:
    """Import a project script (e.g. '01_connectome_graph_analysis/analyze.py') by path."""
    path = PROJECTS / relpath
    name = "proj_" + path.parent.name + "_" + path.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def hemibrain_paths() -> tuple[Path, Path]:
    d = FIXTURES / "hemibrain"
    return d / "traced-total-connections.csv", d / "traced-neurons.csv"


@pytest.fixture()
def tiny_hemibrain(hemibrain_paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn_p, neur_p = hemibrain_paths
    return pd.read_csv(conn_p), pd.read_csv(neur_p)


@pytest.fixture()
def tiny_flywire() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "flywire" / "neurons.tsv", sep="\t")
