"""Extract per-frame kinematics for every tracked object across all 2025 scenes.

Reads ground_truth.json files under data/physicalai/MTMC_Tracking_2025/ and writes
outputs/trajectories.csv with positions, velocities, accelerations, headings and
N-second constant-velocity forecasts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("./data/physicalai/MTMC_Tracking_2025")
OUT_PATH = Path("./outputs/trajectories.csv")
SUMMARY_PATH = Path("./outputs/scene_summary.csv")
FPS = 30.0
HORIZON_SEC = 3.0


def parse_gt(gt_path: Path) -> list[dict]:
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    rows = []
    for fid_str, objs in gt.items():
        if not isinstance(objs, list):
            continue
        fid = int(fid_str)
        for o in objs:
            loc = o.get("3d location", [0.0, 0.0, 0.0])
            rot = o.get("3d bounding box rotation", [0.0, 0.0, 0.0])
            scl = o.get("3d bounding box scale", [0.0, 0.0, 0.0])
            cams = o.get("2d bounding box visible", {}) or {}
            rows.append({
                "frame_id": fid,
                "object_id": int(o.get("object id", -1)),
                "object_type": o.get("object type", "?"),
                "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2]),
                "yaw": float(rot[2]),
                "width": float(scl[0]), "length": float(scl[1]), "height": float(scl[2]),
                "n_cameras_visible": len(cams) if isinstance(cams, dict) else 0,
            })
    return rows


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["scene", "object_type", "object_id", "frame_id"]).reset_index(drop=True)
    g = df.groupby(["scene", "object_type", "object_id"], sort=False)
    dt = g["frame_id"].diff() / FPS
    df["vx"] = g["x"].diff() / dt
    df["vy"] = g["y"].diff() / dt
    df["speed"] = np.hypot(df["vx"], df["vy"])
    df["heading"] = np.arctan2(df["vy"], df["vx"])
    df["ax"] = g["vx"].diff() / dt
    df["ay"] = g["vy"].diff() / dt
    df["x_future"] = df["x"] + df["vx"].fillna(0.0) * HORIZON_SEC
    df["y_future"] = df["y"] + df["vy"].fillna(0.0) * HORIZON_SEC
    return df


def main() -> None:
    gt_paths = sorted(DATA_ROOT.rglob("ground_truth.json"))
    if not gt_paths:
        raise SystemExit(f"No ground_truth.json under {DATA_ROOT}")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for p in gt_paths:
        scene = p.parent.name
        rows = parse_gt(p)
        for r in rows:
            r["scene"] = scene
        all_rows.extend(rows)
        df_scene = pd.DataFrame(rows)
        n_frames = df_scene["frame_id"].nunique()
        type_counts = df_scene.groupby("object_type")["object_id"].nunique().to_dict()
        summaries.append({"scene": scene, "n_frames": n_frames, **type_counts})
        print(f"  {scene}: {n_frames} frames, types={type_counts}")

    df = pd.DataFrame(all_rows)
    df = add_kinematics(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    pd.DataFrame(summaries).fillna(0).to_csv(SUMMARY_PATH, index=False)
    print(f"\n{OUT_PATH}: {len(df):,} rows")
    print(f"{SUMMARY_PATH}: {len(summaries)} scenes")
    print("\nMotion check (objects with > 0.5 m total range):")
    g = df.groupby(["scene", "object_type", "object_id"]).agg(
        x_rng=("x", lambda s: s.max() - s.min()),
        y_rng=("y", lambda s: s.max() - s.min()),
    )
    g["moving"] = (g["x_rng"] + g["y_rng"]) > 0.5
    print(g.groupby("object_type")["moving"].agg(["sum", "count"]).rename(
        columns={"sum": "moving", "count": "total"}))


if __name__ == "__main__":
    main()
