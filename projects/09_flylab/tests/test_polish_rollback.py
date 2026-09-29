"""
Polish mode must never make a deployed brain worse.

train_brains.py --polish keeps training brains that are already optimal. A
polished candidate that beats the champion on physics validation but then falls
under the optimal bar on the held-out test must be rolled back; a candidate that
stays optimal must be kept. Physics is stubbed out, and the test works on a
temporary copy of the goto brain's state, so it never touches live training.

    ..\\..\\..\\.venv-sim\\Scripts\\python test_polish_rollback.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FLYLAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYLAB))
import brain as B                  # noqa: E402
import surrogate as SG             # noqa: E402
import train_brains as TB          # noqa: E402

SKILL = "goto"


def make_sandbox():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "training" / SKILL).mkdir(parents=True)
    (tmp / "state" / "brains").mkdir(parents=True)
    shutil.copy(FLYLAB / "training" / SKILL / "status.json", tmp / "training" / SKILL / "status.json")
    shutil.copy(FLYLAB / "state" / "brains" / f"{SKILL}.npy", tmp / "state" / "brains" / f"{SKILL}.npy")
    TB.TRAIN_DIR, TB.STATE = tmp / "training", tmp / "state"
    TB.write_skill_report = TB.write_summary = TB._update_flylab_history = lambda *a, **k: None
    TB.ROUND_GENS = 1
    return tmp


def polish_once(tmp, test_rate):
    """One polish round where validation improves and the held-out test scores `test_rate`."""
    run = TB.SkillRun(SKILL, B.Brain(TB.STATE))
    assert run.optimal, f"{SKILL} must start optimal for this test"
    best_reward = run.status["best"]["physics_reward"]
    TB.physics_eval = lambda theta, task, n, seed: (
        (test_rate if seed == TB.test_seed(task) else 1.0), best_reward + 0.5, [0.1] * n)
    before = np.load(tmp / "state" / "brains" / f"{SKILL}.npy")
    TB.run_round(run, SG.Body(), 6, 8, say=lambda m: None, polish=True)
    after = np.load(tmp / "state" / "brains" / f"{SKILL}.npy")
    return run, best_reward, before, after


def main():
    tmp = make_sandbox()

    run, reward, before, after = polish_once(tmp, test_rate=0.5)
    assert run.optimal, "a rejected polish must leave the brain optimal"
    assert np.array_equal(before, after), "a rejected polish must not change the deployed brain"
    assert run.status["best"]["physics_reward"] == reward, "a rejected polish must not change the record"
    last = json.loads((tmp / "training" / SKILL / "log.jsonl").read_text().splitlines()[-1])
    assert last.get("polish_rejected") and not last["improved"], "the round must be logged as rejected"
    print("ok  polish that would drop the held-out test to 50% is rolled back")

    run, reward, before, after = polish_once(tmp, test_rate=1.0)
    assert run.optimal and not np.array_equal(before, after), "a good polish must be deployed"
    assert run.status["best"]["physics_reward"] > reward, "a good polish must raise the recorded reward"
    print("ok  polish that stays optimal is kept")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
