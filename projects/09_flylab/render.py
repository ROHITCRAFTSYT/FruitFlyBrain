"""
Turn a physics run into things you can watch: a composite MP4 (overhead arena +
close-up of the fly + live brain HUD), a top-down trajectory map, and the
brain's learning-progress chart.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tasks as T
from surrogate import DT
from world import light_intensity, odor_intensity

PLAYBACK = 0.5
FPS = 25


def _font(size):
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bar(draw, x, y, w, h, value, lo, hi, color, label, font):
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90), fill=(30, 30, 30))
    zero = x + w * (0 - lo) / (hi - lo)
    v = x + w * (np.clip(value, lo, hi) - lo) / (hi - lo)
    draw.rectangle([min(zero, v), y + 2, max(zero, v), y + h - 2], fill=color)
    draw.line([zero, y, zero, y + h], fill=(200, 200, 200))
    draw.text((x, y - 18), f"{label} {value:+.2f}", font=font, fill=(220, 220, 220))


def _sense_text(f):
    """Human-readable version of the brain's sensory vector (undo feature scaling)."""
    parts = []
    if f[0] > 1e-4:
        side = "left" if f[1] > 0 else "right"
        parts.append(f"smell {f[0] / 2:.2f} (stronger {side})")
    if f[3] > 1e-4:
        side = "left" if f[4] > 0 else "right"
        parts.append(f"light {f[3] / 2:.2f} (brighter {side})")
    if abs(f[6]) + abs(f[7]) > 1e-6:
        parts.append(f"goal at {np.degrees(np.arctan2(f[6], f[7])):+.0f} deg")
    return ",  ".join(parts) or "-"


def make_video(frames: dict, traj: dict, title: str, subtitle: str, verdict: str,
               ok: bool, path: Path):
    names = list(frames.keys())
    if not names:
        return None
    n = min(len(frames[k]) for k in names)
    h, w = np.asarray(frames[names[0]][0]).shape[:2]
    hud_h = 120
    f_big, f_small = _font(22), _font(15)
    drives = traj["drives"]
    writer = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7,
                                macro_block_size=8)
    for k in range(n):
        tiles = [np.asarray(frames[c][k]) for c in names]
        top = np.hstack(tiles)
        canvas = Image.new("RGB", (top.shape[1], top.shape[0] + hud_h), (18, 18, 22))
        canvas.paste(Image.fromarray(top), (0, 0))
        d = ImageDraw.Draw(canvas)
        t_sim = k * PLAYBACK / FPS
        idx = min(int(t_sim / DT), len(drives) - 1)
        dl, dr = drives[idx]
        # labels over the camera tiles
        for i, c in enumerate(names):
            d.text((i * w + 10, 8), "ARENA (overhead)" if "overhead" in c else "FLY (close-up)",
                   font=f_small, fill=(255, 255, 255))
        y0 = top.shape[0]
        d.text((14, y0 + 8), title, font=f_big, fill=(255, 255, 255))
        d.text((14, y0 + 40), subtitle, font=f_small, fill=(170, 200, 255))
        d.text((14, y0 + 64), f"t = {t_sim:4.2f} s   (video at {PLAYBACK:.1f}x speed)   "
               f"senses: {_sense_text(traj['senses'][idx])}",
               font=f_small, fill=(200, 200, 200))
        if k >= n - FPS * 1.2:
            d.text((14, y0 + 88), verdict, font=f_small,
                   fill=(120, 230, 120) if ok else (240, 170, 90))
        bx = top.shape[1] - 330
        d.text((bx, y0 + 6), "descending drives (brain -> legs)", font=f_small, fill=(200, 200, 200))
        _bar(d, bx, y0 + 50, 300, 16, dl, -0.5, 1.2, (90, 170, 255), "LEFT ", f_small)
        _bar(d, bx, y0 + 92, 300, 16, dr, -0.5, 1.2, (255, 140, 90), "RIGHT", f_small)
        writer.append_data(np.asarray(canvas))
    writer.close()
    return path


def mp4_to_gif(mp4: Path, gif: Path, width=560, fps=10, max_seconds=12):
    """Downscaled GIF of a lesson video (for README embedding)."""
    reader = imageio.get_reader(str(mp4))
    src_fps = reader.get_meta_data().get("fps", FPS)
    step = max(1, int(round(src_fps / fps)))
    frames = []
    for i, fr in enumerate(reader):
        if i % step:
            continue
        if len(frames) >= fps * max_seconds:
            break
        im = Image.fromarray(fr)
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    reader.close()
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return gif


def plot_trajectory(scen: T.Scenario, traj: dict, pred: tuple | None, title: str, path: Path):
    xs, ys = traj["x"], traj["y"]
    pts = [xs, ys]
    if "target" in scen.extra:
        tx, ty = float(scen.extra["target"][0][0]), float(scen.extra["target"][1][0])
        pts += [np.array([tx]), np.array([ty])]
    allx = np.concatenate([pts[0]] + ([pts[2]] if len(pts) > 2 else []))
    ally = np.concatenate([pts[1]] + ([pts[3]] if len(pts) > 2 else []))
    pad = 5
    x0, x1 = allx.min() - pad, allx.max() + pad
    y0, y1 = ally.min() - pad, ally.max() + pad
    span = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1, y0, y1 = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    st = scen.stim
    if st["odor_on"][0]:
        field = odor_intensity(gx, gy, st["odor_x"][0], st["odor_y"][0])
        ax.imshow(field, origin="lower", extent=[x0, x1, y0, y1],
                  cmap="Greens" if scen.task == "odor_seek" else "Purples", alpha=0.55)
    if st["light_on"][0]:
        field = light_intensity(gx, gy, st["light_x"][0], st["light_y"][0])
        ax.imshow(field, origin="lower", extent=[x0, x1, y0, y1], cmap="YlOrBr", alpha=0.6)
    if pred is not None:
        ax.plot(pred[0], pred[1], "--", color="#555", lw=1.5, label="surrogate prediction")
    sc = ax.scatter(xs, ys, c=np.linspace(0, 1, len(xs)), cmap="plasma", s=9, zorder=3)
    ax.plot(xs, ys, color="#333", lw=0.8, alpha=0.6, label="physics fly (NeuroMechFly)")
    ax.annotate("", xy=(xs[0] + 1.5 * np.cos(traj["th"][0]), ys[0] + 1.5 * np.sin(traj["th"][0])),
                xytext=(xs[0], ys[0]), arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    ax.plot(xs[0], ys[0], "ko", ms=6, label="start")
    if "target" in scen.extra:
        label = {"odor_seek": "food", "odor_avoid": "repellent", "light_seek": "light",
                 "light_avoid": "light"}.get(scen.task, "goal")
        ax.plot(tx, ty, marker="*", ms=18, color="crimson", mec="k", label=label, zorder=4)
        if scen.task not in ("odor_avoid", "light_avoid"):
            ax.add_patch(plt.Circle((tx, ty), T.SUCCESS_RADIUS_MM, fill=False, ls=":", color="crimson"))
    if scen.task == "turn":
        th_t = float(scen.extra["target_heading"][0])
        ax.annotate("", xy=(xs[-1] + 4 * np.cos(th_t), ys[-1] + 4 * np.sin(th_t)),
                    xytext=(xs[-1], ys[-1]),
                    arrowprops=dict(arrowstyle="->", lw=2, color="crimson", ls="--"))
        ax.annotate("", xy=(xs[-1] + 3 * np.cos(traj["th"][-1]), ys[-1] + 3 * np.sin(traj["th"][-1])),
                    xytext=(xs[-1], ys[-1]), arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4"))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="time (normalized)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_progress(history: dict, path: Path):
    lessons = history.get("lessons", [])
    if not lessons:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    # left: every lesson's surrogate learning curve, stitched end-to-end
    offset = 0
    colors = {t: plt.get_cmap("tab10")(i) for i, t in enumerate(T.TASKS)}
    seen = set()
    for les in lessons:
        g = np.array([c[0] for c in les["curve"]]) + offset
        s = np.array([c[1] for c in les["curve"]]) * 100
        lab = T.PRETTY[les["task"]] if les["task"] not in seen else None
        seen.add(les["task"])
        ax[0].plot(g, s, "-o", ms=3, color=colors[les["task"]], label=lab)
        offset = g[-1]
    ax[0].set_xlabel("total training generations (all lessons)")
    ax[0].set_ylabel("success in surrogate (%)")
    ax[0].set_ylim(-5, 105); ax[0].grid(alpha=0.3)
    ax[0].set_title("The brain learning, lesson after lesson")
    ax[0].legend(fontsize=7, loc="lower right")
    # right: skill snapshot after the latest lesson
    skills = history.get("skills", {})
    names = [t for t in T.TASKS if t in skills]
    sur = [skills[t].get("surrogate_success", 0) * 100 for t in names]
    phy = [100 if skills[t].get("physics_success") else 0 for t in names]
    y = np.arange(len(names))
    ax[1].barh(y - 0.18, sur, height=0.36, color="#4e79a7", label="surrogate (48 random trials)")
    ax[1].barh(y + 0.18, phy, height=0.36, color="#f28e2b", label="last physics run")
    ax[1].set_yticks(y); ax[1].set_yticklabels([T.PRETTY[t] for t in names])
    ax[1].set_xlim(0, 105); ax[1].set_xlabel("success (%)")
    ax[1].set_title("Skills the fly has learned")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
