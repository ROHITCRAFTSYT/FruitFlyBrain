"""
Real physics footage of every deployed skill brain, for the training dashboard.

For each skill this loads the deployed brain (state/brains/<skill>.npy), runs ONE
full NeuroMechFly v2 / MuJoCo episode on a fixed, seeded demo arena and writes

    dashboard/videos/<skill>.mp4   H.264, browser-playable, a few seconds long
    dashboard/videos/<skill>.png   poster (last frame, outcome visible)
    dashboard/videos/index.json    {"videos": [...]}, one entry per skill

Nothing here is staged: the arena is drawn by the same generator the trainer
uses (evaluate_physics.random_deploy_scenario), but from seed 30000+i so it is
never one of the validation (10000+i) or test (20000+i) arenas. A failed
episode is kept as-is: it is what the brain really does.

    ..\\..\\.venv-sim\\Scripts\\python render_brains.py                 # all skills
    ..\\..\\.venv-sim\\Scripts\\python render_brains.py --skills goto turn
    ..\\..\\.venv-sim\\Scripts\\python render_brains.py --changed-only  # only new best brains

Runs at BELOW_NORMAL priority so it never slows down the continuous trainer.
"""
from __future__ import annotations

import sys

if sys.platform == "win32":                       # yield the CPU to the trainer
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import physics                                     # noqa: E402
import tasks as T                                  # noqa: E402
from evaluate_physics import random_deploy_scenario  # noqa: E402
from render import FPS as REC_FPS, PLAYBACK as REC_PLAYBACK, _bar, _font, _sense_text  # noqa: E402
from surrogate import DT                           # noqa: E402

BRAINS = HERE / "state" / "brains"
TRAIN = HERE / "training"
OUT = HERE / "dashboard" / "videos"
DEMO_SEED = 30_000                                 # + skill index; val = 10000+i, test = 20000+i
OUT_FPS = 25
HUD_H = 120
MAX_BYTES = 3 * 1024 * 1024

UNITS = {"turn": "deg", "odor_avoid": "mm gained", "light_avoid": "mm gained"}
MARKER = {"walk_forward": "goal flag", "goto": "goal flag", "odor_seek": "food",
          "odor_avoid": "repellent", "light_seek": "lamp", "light_avoid": "lamp"}


def demo_seed(task):
    return DEMO_SEED + T.TASKS.index(task)


def unit_of(task):
    return UNITS.get(task, "mm")


# ---------------------------------------------------------------------------
# outcome (same definitions as train_brains._error_of and tasks.score)
# ---------------------------------------------------------------------------
def outcome(task, scen, traj):
    """(success, final_error, sentence). final_error uses the trainer's units."""
    _, ok = T.score(scen, traj["x"][None], traj["y"][None], traj["th"][None])
    ok = bool(ok[0])
    x, y, th = traj["x"], traj["y"], traj["th"]
    if task == "turn":
        err = float(np.degrees(abs(T.wrap(th[-1] - scen.extra["target_heading"][0]))))
        drift = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        return ok, round(err, 1), (f"heading error {err:.1f} deg, drift {drift:.1f} mm "
                                   f"(pass: < 20 deg and < {T.TURN_MAX_DISP_MM:.0f} mm)")
    tx, ty = scen.extra["target"][0][0], scen.extra["target"][1][0]
    df = float(np.hypot(x[-1] - tx, y[-1] - ty))
    if task in ("odor_avoid", "light_avoid"):
        gain = df - float(np.hypot(x[0] - tx, y[0] - ty))
        return ok, round(gain, 1), f"moved {gain:+.1f} mm further away (pass: > {T.AVOID_GAIN_MM:.0f} mm, never approaching)"
    return ok, round(df, 2), (f"ended {df:.2f} mm from the {MARKER[task]} "
                              f"(pass: < {T.SUCCESS_RADIUS_MM} mm and standing still)")


def describe_arena(task, scen):
    if task == "turn":
        a = float(np.degrees(scen.extra["target_heading"][0]))
        return f"turn {abs(a):.0f} deg {'left' if a > 0 else 'right'} in place"
    tx, ty = float(scen.extra["target"][0][0]), float(scen.extra["target"][1][0])
    r, b = float(np.hypot(tx, ty)), float(np.degrees(np.arctan2(ty, tx)))
    side = "ahead" if abs(b) < 10 else (f"{abs(b):.0f} deg to the {'left' if b > 0 else 'right'}")
    return f"{MARKER[task]} {r:.1f} mm away, {side}"


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
def _minimap(d, box, scen, traj, upto):
    """Top-down trace (fly path so far, start, target) drawn into box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(28, 28, 34), outline=(80, 80, 90))
    xs, ys = traj["x"][:upto + 1], traj["y"][:upto + 1]
    px, py = list(traj["x"]), list(traj["y"])
    if "target" in scen.extra:
        px.append(float(scen.extra["target"][0][0])); py.append(float(scen.extra["target"][1][0]))
    cx, cy = (max(px) + min(px)) / 2, (max(py) + min(py)) / 2
    span = max(max(px) - min(px), max(py) - min(py), 8.0) + 4.0
    s = min(x1 - x0, y1 - y0) / span

    def P(x, y):
        return ((x0 + x1) / 2 + (x - cx) * s, (y0 + y1) / 2 - (y - cy) * s)
    if "target" in scen.extra:
        tx, ty = P(float(scen.extra["target"][0][0]), float(scen.extra["target"][1][0]))
        if scen.task not in ("odor_avoid", "light_avoid"):
            r = T.SUCCESS_RADIUS_MM * s
            d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=(230, 80, 80))
        col = {"odor_avoid": (170, 60, 190), "odor_seek": (170, 220, 40),
               "light_seek": (255, 230, 90), "light_avoid": (255, 230, 90)}.get(scen.task, (230, 60, 60))
        d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=col)
    if scen.task == "turn":
        th_t = float(scen.extra["target_heading"][0])
        fx, fy = P(xs[-1], ys[-1])
        d.line([fx, fy, fx + 22 * np.cos(th_t), fy - 22 * np.sin(th_t)], fill=(230, 80, 80), width=2)
    pts = [P(a, b) for a, b in zip(xs, ys)]
    if len(pts) > 1:
        d.line(pts, fill=(120, 190, 255), width=2)
    sx, sy = P(traj["x"][0], traj["y"][0])
    d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(200, 200, 200))
    fx, fy = pts[-1]
    th = traj["th"][upto]
    d.line([fx, fy, fx + 12 * np.cos(th), fy - 12 * np.sin(th)], fill=(255, 255, 255), width=2)


def write_video(frames, traj, scen, task, header, verdict, ok, mp4, png):
    names = list(frames.keys())
    n = min(len(frames[c]) for c in names)
    # The recorder runs at REC_PLAYBACK x speed (0.5x -> 50 frames per sim second).
    # Long episodes are shown in real time (every 2nd frame), short ones at 0.5x
    # so every clip lasts ~5-10 s.
    stride = 1 if scen.duration < 5 else 2
    speed = REC_PLAYBACK * stride
    h, w = np.asarray(frames[names[0]][0]).shape[:2]
    W = w * len(names)
    W16, H16 = W - W % 16, (h + HUD_H) - (h + HUD_H) % 16
    f_big, f_small = _font(20), _font(14)
    drives = traj["drives"]
    writer = imageio.get_writer(str(mp4), fps=OUT_FPS, codec="libx264", quality=None,
                                pixelformat="yuv420p", macro_block_size=16,
                                output_params=["-crf", "27", "-preset", "slow",
                                               "-movflags", "+faststart"])
    last = None
    ks = list(range(0, n, stride))
    for j, k in enumerate(ks):
        top = np.hstack([np.asarray(frames[c][k])[..., :3] for c in names])
        canvas = Image.new("RGB", (W, h + HUD_H), (18, 18, 22))
        canvas.paste(Image.fromarray(top), (0, 0))
        d = ImageDraw.Draw(canvas)
        t_sim = k * REC_PLAYBACK / REC_FPS
        idx = min(int(t_sim / DT), len(drives) - 1)
        for i, c in enumerate(names):
            d.text((i * w + 10, 8), "ARENA (overhead)" if "overhead" in c else "FLY (close-up)",
                   font=f_small, fill=(255, 255, 255))
        d.text((W - 250, 8), "NeuroMechFly v2 + MuJoCo physics", font=f_small, fill=(210, 210, 210))
        y0 = h
        mm = HUD_H - 12
        _minimap(d, (8, y0 + 6, 8 + mm, y0 + 6 + mm), scen, traj, min(idx + 1, len(traj["x"]) - 1))
        tx0 = 8 + mm + 12
        d.text((tx0, y0 + 6), header, font=f_big, fill=(255, 255, 255))
        d.text((tx0, y0 + 34), f"arena: {describe_arena(task, scen)}  (unseen demo seed {demo_seed(task)})",
               font=f_small, fill=(170, 200, 255))
        d.text((tx0, y0 + 56), f"t = {t_sim:4.2f} / {scen.duration:.0f} s  ({speed:g}x)  "
               f"{_sense_text(traj['senses'][idx])}", font=f_small, fill=(200, 200, 200))
        final = j >= len(ks) - int(OUT_FPS * 1.5)
        d.text((tx0, y0 + 80), (("SUCCESS - " if ok else "FAILED - ") + verdict) if final
               else "outcome shown at the end of the episode",
               font=f_small, fill=((120, 230, 120) if ok else (240, 150, 90)) if final else (120, 120, 120))
        bx = W - 250
        dl, dr = drives[idx]
        d.text((bx, y0 + 6), "descending drives (brain -> legs)", font=f_small, fill=(200, 200, 200))
        _bar(d, bx, y0 + 50, 230, 14, dl, -0.5, 1.2, (90, 170, 255), "LEFT ", f_small)
        _bar(d, bx, y0 + 92, 230, 14, dr, -0.5, 1.2, (255, 140, 90), "RIGHT", f_small)
        last = canvas.crop((0, 0, W16, H16))
        writer.append_data(np.asarray(last))
    writer.close()
    last.save(png, optimize=True)
    return len(ks) / OUT_FPS, speed


def probe(mp4):
    """(frames, duration_s) read back from the file, to prove it decodes."""
    r = imageio.get_reader(str(mp4))
    meta = r.get_meta_data()
    count = sum(1 for _ in r)
    r.close()
    return count, count / meta.get("fps", OUT_FPS), meta.get("codec"), meta.get("size")


# ---------------------------------------------------------------------------
def read_json(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_index(entries):
    OUT.mkdir(parents=True, exist_ok=True)
    order = {t: i for i, t in enumerate(T.TASKS)}
    vids = sorted(entries.values(), key=lambda v: order.get(v["skill"], 99))
    tmp = OUT / "index.json.tmp"
    tmp.write_text(json.dumps({"videos": vids}, indent=2), encoding="utf-8")
    os.replace(tmp, OUT / "index.json")


def load_brain(task):
    """Deployed weights + the status they belong to (re-read if the trainer swaps mid-read)."""
    for _ in range(5):
        st = read_json(TRAIN / task / "status.json", {})
        theta = np.load(BRAINS / f"{task}.npy")
        st2 = read_json(TRAIN / task / "status.json", {})
        if (st.get("best") or {}).get("round") == (st2.get("best") or {}).get("round"):
            return theta, st2
        time.sleep(2)
    return theta, st2


def render_skill(task, log=print):
    t0 = time.time()
    theta, st = load_brain(task)
    best = st.get("best") or {}
    rnd, level = best.get("round"), st.get("level", 0)
    rng = np.random.default_rng(demo_seed(task))
    scen = random_deploy_scenario(task, rng)
    log(f"[{task}] brain round {rnd}, level {level + 1}; arena: {describe_arena(task, scen)}")
    traj, frames = physics.deploy(theta, scen, record=True, log=None)
    ok, err, verdict = outcome(task, scen, traj)
    header = f"{T.PRETTY[task]}  |  brain from training round {rnd}  |  level {level + 1}"
    mp4, png = OUT / f"{task}.mp4", OUT / f"{task}.png"
    tmp = OUT / f"{task}.tmp.mp4"
    secs, speed = write_video(frames, traj, scen, task, header, verdict, ok, tmp, png)
    os.replace(tmp, mp4)
    nfr, dur, codec, size = probe(mp4)
    nbytes = mp4.stat().st_size
    wall = time.time() - t0
    log(f"[{task}] {'SUCCESS' if ok else 'FAILED'}: {verdict}; {nfr} frames, {dur:.1f} s, "
        f"{codec} {size}, {nbytes / 1024:.0f} KB, rendered in {wall:.0f} s")
    if nbytes > MAX_BYTES:
        log(f"[{task}] WARNING: {nbytes} bytes > 3 MB budget")
    entry = {"skill": task, "file": mp4.name, "poster": png.name, "brain_round": rnd,
             "level": level, "arena": describe_arena(task, scen), "success": ok,
             "final_error": err, "unit": unit_of(task), "outcome": verdict,
             "seed": demo_seed(task), "sim_seconds": scen.duration, "video_seconds": round(dur, 2),
             "playback_speed": speed, "rendered_at": datetime.now().isoformat(timespec="seconds")}
    return entry, {"wall_s": round(wall, 1), "bytes": nbytes, "frames": nfr, "codec": codec}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skills", nargs="+", default=list(T.TASKS), choices=T.TASKS)
    ap.add_argument("--changed-only", action="store_true",
                    help="skip skills whose video already shows the current best brain")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    entries = {v["skill"]: v for v in read_json(OUT / "index.json", {}).get("videos", [])
               if isinstance(v, dict) and v.get("skill")}
    t0 = time.time()
    for task in args.skills:
        if not (BRAINS / f"{task}.npy").exists():
            print(f"[{task}] no deployed brain yet, skipped", flush=True)
            continue
        if args.changed_only:
            cur = (read_json(TRAIN / task / "status.json", {}).get("best") or {}).get("round")
            have = entries.get(task, {}).get("brain_round")
            if have is not None and have == cur and (OUT / f"{task}.mp4").exists():
                print(f"[{task}] video already shows round {cur}, skipped", flush=True)
                continue
        try:
            entry, _ = render_skill(task, log=lambda m: print(m, flush=True))
        except Exception as e:                      # one broken skill must not stop the rest
            print(f"[{task}] render failed: {e!r}", flush=True)
            continue
        entries[task] = entry
        write_index(entries)                        # after every skill: partial runs still publish
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
