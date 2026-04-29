"""Render a top-down (bird's-eye-view) MP4 of the synth-forklift scene.

Shows:
  - oriented forklift body + heading arrow
  - 3-second future ghost (constant-velocity prediction)
  - adaptive red zone circle (radius = 1.5 + v + v^2/3)
  - all persons as dots (red when inside red zone or predicted breach)
  - other (real) vehicles in scene as gray dots
  - frame counter / time / alarm banner

Default: first 60 seconds at 15 fps. Override with --seconds / --fps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle, FancyArrow, Rectangle

DATA_ROOT = Path("./data/physicalai/MTMC_Tracking_2025")
SYNTH_GT = DATA_ROOT / "train/Warehouse_006_synth/ground_truth.json"
PERSONS_TRAJ = Path("./outputs/trajectories.csv")
OUT_PATH = Path("./outputs/synth_bev.mp4")
SOURCE_SCENE = "Warehouse_006"
HORIZON_S = 3.0
REACTION_S = 1.0
BRAKE_DECEL = 1.5
MIN_RADIUS = 1.5
GT_DIST = 3.0


def adaptive_radius(v: float) -> float:
    v = max(0.0, v)
    return MIN_RADIUS + REACTION_S * v + (v ** 2) / (2 * BRAKE_DECEL)


def load_synth_forklift() -> pd.DataFrame:
    with open(SYNTH_GT, encoding="utf-8") as f:
        gt = json.load(f)
    rows = []
    for fid_str, objs in gt.items():
        if not isinstance(objs, list):
            continue
        fid = int(fid_str)
        for o in objs:
            if o.get("object type") != "Forklift":
                continue
            loc = o["3d location"]; rot = o["3d bounding box rotation"]; scl = o["3d bounding box scale"]
            rows.append({
                "frame_id": fid, "veh_id": int(o["object id"]),
                "x": float(loc[0]), "y": float(loc[1]), "yaw": float(rot[2]),
                "w": float(scl[0]), "l": float(scl[1]),
            })
    df = pd.DataFrame(rows).sort_values(["veh_id","frame_id"]).reset_index(drop=True)
    g = df.groupby("veh_id", sort=False)
    dt = g.frame_id.diff() / 30.0
    df["vx"] = g.x.diff() / dt
    df["vy"] = g.y.diff() / dt
    df["speed"] = np.hypot(df.vx, df.vy)
    return df


def load_persons() -> pd.DataFrame:
    cols = ["scene","frame_id","object_id","object_type","x","y","vx","vy","speed"]
    parts = []
    for chunk in pd.read_csv(PERSONS_TRAJ, usecols=cols, chunksize=500_000):
        m = (chunk.scene == SOURCE_SCENE) & (chunk.object_type == "Person")
        if m.any():
            parts.append(chunk.loc[m, ["frame_id","object_id","x","y","vx","vy","speed"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--source-fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    fk_all = load_synth_forklift()
    moving_id = fk_all.groupby("veh_id").speed.max().idxmax()
    fk = fk_all[fk_all.veh_id == moving_id].set_index("frame_id")
    other_fk = fk_all[fk_all.veh_id != moving_id]
    print(f"main forklift id={moving_id}, frames={len(fk):,}, max speed={fk.speed.max():.2f} m/s")

    persons = load_persons().set_index("frame_id")
    print(f"persons rows: {len(persons):,}")

    step = max(1, int(round(args.source_fps / args.fps)))
    n_render = int(args.seconds * args.fps)
    src_frames = [i * step for i in range(n_render)]
    src_frames = [f for f in src_frames if f in fk.index]
    print(f"will render {len(src_frames)} frames at {args.fps} fps -> {args.seconds:.0f} s")

    all_x = pd.concat([fk.x, persons.x]); all_y = pd.concat([fk.y, persons.y])
    pad = 5
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_facecolor("#101418")
    fig.patch.set_facecolor("#0a0c0f")
    ax.tick_params(colors="#777"); ax.spines[:].set_color("#333")
    ax.grid(True, color="#222", lw=0.5)

    # static elements
    ax.set_title("Forklift Safety BEV — Warehouse_006 + synth forklift",
                 color="#eee", fontsize=12)

    # full forklift path (faint guide)
    ax.plot(fk.x, fk.y, color="#444", lw=0.7, ls="--", label="forklift path")

    # other (stationary) forklift
    if not other_fk.empty:
        ax.scatter(other_fk.x.iloc[0], other_fk.y.iloc[0], marker="s",
                   s=120, c="#555", edgecolors="#888", label="static forklift")

    fk_body = Rectangle((0,0), 1, 1, fc="#ffd166", ec="#222", zorder=4)
    fk_ghost = Rectangle((0,0), 1, 1, fc="none", ec="#ffd166", ls=":", zorder=3, alpha=0.6)
    red_zone = Circle((0,0), 1.0, fc="#ff5c5c", alpha=0.18, ec="#ff5c5c", lw=1.2, zorder=2)
    arrow = ax.annotate("", xy=(0,0), xytext=(0,0),
                        arrowprops=dict(arrowstyle="->", color="#ffd166", lw=2), zorder=5)
    persons_safe = ax.scatter([], [], s=28, c="#7adfff", zorder=4, label="person")
    persons_warn = ax.scatter([], [], s=70, c="#ff5c5c", ec="white", lw=0.8, zorder=6, label="alarm")

    ax.add_patch(red_zone); ax.add_patch(fk_body); ax.add_patch(fk_ghost)

    info = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                   color="#eee", fontsize=10, family="monospace",
                   bbox=dict(facecolor="#000a", edgecolor="#333", pad=4))
    banner = ax.text(0.5, 0.94, "", transform=ax.transAxes, va="center", ha="center",
                     color="white", fontsize=18, fontweight="bold", visible=False,
                     bbox=dict(facecolor="#cc1f1f", edgecolor="white", pad=6))
    ax.legend(loc="lower right", facecolor="#101418", edgecolor="#333", labelcolor="#ccc")

    def set_oriented_rect(patch: Rectangle, x: float, y: float, yaw: float,
                          w: float = 1.5, l: float = 3.0):
        patch.set_width(l); patch.set_height(w)
        patch.set_xy((x - l/2, y - w/2))
        patch.angle = float(np.degrees(yaw))
        # rotate around center
        from matplotlib.transforms import Affine2D
        t = Affine2D().rotate_deg_around(x, y, np.degrees(yaw)) + ax.transData
        patch.set_transform(t)

    def update(idx):
        f = src_frames[idx]
        if f not in fk.index:
            return ()
        row = fk.loc[f]
        x, y, yaw = float(row.x), float(row.y), float(row.yaw)
        v = float(row.speed) if pd.notna(row.speed) else 0.0
        vx = float(row.vx) if pd.notna(row.vx) else 0.0
        vy = float(row.vy) if pd.notna(row.vy) else 0.0
        r = adaptive_radius(v)

        set_oriented_rect(fk_body, x, y, yaw, w=row.w if pd.notna(row.w) else 1.5,
                          l=row.l if pd.notna(row.l) else 3.0)
        fx = x + vx * HORIZON_S; fy = y + vy * HORIZON_S
        set_oriented_rect(fk_ghost, fx, fy, yaw, w=row.w if pd.notna(row.w) else 1.5,
                          l=row.l if pd.notna(row.l) else 3.0)

        red_zone.set_center((x, y)); red_zone.set_radius(r)

        # arrow
        arrow.xy = (x + vx * 1.5, y + vy * 1.5)
        arrow.set_position((x, y))

        # persons in this frame
        try:
            pframe = persons.loc[[f]] if f in persons.index else persons.iloc[0:0]
        except KeyError:
            pframe = persons.iloc[0:0]

        if not pframe.empty:
            px = pframe.x.values; py = pframe.y.values
            pvx = pframe.vx.fillna(0).values; pvy = pframe.vy.fillna(0).values
            d_now = np.hypot(px - x, py - y)
            fxp = px + pvx * HORIZON_S; fyp = py + pvy * HORIZON_S
            d_fut = np.hypot(fxp - fx, fyp - fy)
            warn = (v >= 0.3) & ((d_now <= r) | (d_fut <= r))
            persons_safe.set_offsets(np.column_stack([px[~warn], py[~warn]])
                                     if (~warn).any() else np.empty((0,2)))
            persons_warn.set_offsets(np.column_stack([px[warn], py[warn]])
                                     if warn.any() else np.empty((0,2)))
            n_warn = int(warn.sum())
        else:
            persons_safe.set_offsets(np.empty((0,2)))
            persons_warn.set_offsets(np.empty((0,2)))
            n_warn = 0

        info.set_text(f"frame {f:>4d}  t={f/30.0:5.1f}s  speed={v:4.2f} m/s  red r={r:4.2f} m  alarms={n_warn}")
        banner.set_visible(n_warn > 0)
        if n_warn:
            banner.set_text(f"⚠  ALARM × {n_warn}")
        return (fk_body, fk_ghost, red_zone, persons_safe, persons_warn, info, banner)

    print("rendering ...")
    anim = FuncAnimation(fig, update, frames=len(src_frames), interval=1000/args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"],
    )
    anim.save(args.out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
