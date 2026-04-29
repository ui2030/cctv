"""Phase 3 — synthesize a moving forklift in a real warehouse scene.

Take an existing 2025 ground_truth.json (e.g. Warehouse_006) and replace one
of its stationary forklifts with a procedurally-driven trajectory that walks
through the bounding box of pedestrian activity. Yaw follows the heading.

Output: data/physicalai/MTMC_Tracking_2025/synth/<Scene>/ground_truth.json
After running, point extract_tracks.py / compute_pairs.py / predict_alarm.py
at this synth file (see SYNTH_DATA_ROOT in those scripts) — the easy way is
to just copy this file under DATA_ROOT and run extract_tracks.py.

Defaults:
  - source scene  : Warehouse_006   (100 persons, plenty of intersections)
  - source forklift: first stationary forklift in scene
  - speed: 1.5 m/s (~5.4 km/h indoor cruise)
  - turn radius: 2 m at corners
  - waypoints: along principal aisles inferred from pedestrian bbox
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path("./data/physicalai/MTMC_Tracking_2025")
SOURCE_SCENE = "Warehouse_006"
SYNTH_SCENE_NAME = f"{SOURCE_SCENE}_synth"
SYNTH_OUT = DATA_ROOT / "train" / SYNTH_SCENE_NAME / "ground_truth.json"

FPS = 30
SPEED_MS = 1.5
TURN_RADIUS_M = 2.0


def load_gt(scene: str) -> dict:
    candidates = list(DATA_ROOT.rglob(f"{scene}/ground_truth.json"))
    if not candidates:
        raise SystemExit(f"ground_truth.json for {scene} not found under {DATA_ROOT}")
    with open(candidates[0], encoding="utf-8") as f:
        return json.load(f)


def find_first_forklift(gt: dict) -> tuple[int, dict]:
    for fid, objs in gt.items():
        if not isinstance(objs, list):
            continue
        for o in objs:
            if o.get("object type") == "Forklift":
                return int(o.get("object id")), o
    raise SystemExit(f"No Forklift in scene")


def pedestrian_bounds(gt: dict) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for fid, objs in gt.items():
        if not isinstance(objs, list):
            continue
        for o in objs:
            if o.get("object type") == "Person":
                loc = o.get("3d location", [0,0,0])
                xs.append(float(loc[0])); ys.append(float(loc[1]))
    return min(xs), max(xs), min(ys), max(ys)


def build_waypoints(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    pad = 2.0
    xmin += pad; xmax -= pad; ymin += pad; ymax -= pad
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return np.array([
        [xmin, cy],
        [cx,   cy],
        [cx,   ymin],
        [cx,   ymax],
        [cx,   cy],
        [xmax, cy],
        [xmax, ymin],
        [xmin, ymin],
        [xmin, cy],
    ], dtype=float)


def trajectory_along(waypoints: np.ndarray, speed: float, n_frames: int, fps: int) -> np.ndarray:
    pts = [waypoints[0]]
    yaws = [0.0]
    seg_idx = 0
    cur = waypoints[0].astype(float).copy()
    dt = 1.0 / fps
    while len(pts) < n_frames:
        if seg_idx >= len(waypoints) - 1:
            seg_idx = 0
            cur = waypoints[0].astype(float).copy()
        target = waypoints[seg_idx + 1]
        diff = target - cur
        d = float(np.hypot(*diff))
        if d < 1e-6:
            seg_idx += 1
            continue
        step = speed * dt
        if step >= d:
            cur = target.astype(float).copy()
            seg_idx += 1
        else:
            cur = cur + diff * (step / d)
        pts.append(cur.copy())
        yaws.append(np.arctan2(diff[1], diff[0]))
    arr = np.column_stack([np.array(pts[:n_frames]), np.array(yaws[:n_frames])])
    return arr  # columns: x, y, yaw


def main() -> None:
    gt = load_gt(SOURCE_SCENE)
    fk_id, fk_template = find_first_forklift(gt)
    xmin, xmax, ymin, ymax = pedestrian_bounds(gt)
    print(f"pedestrian bbox: x [{xmin:.1f}, {xmax:.1f}], y [{ymin:.1f}, {ymax:.1f}]")

    waypoints = build_waypoints(xmin, xmax, ymin, ymax)
    print(f"waypoints: {len(waypoints)}")

    n_frames = max(int(k) for k in gt.keys() if isinstance(gt.get(k), list)) + 1
    print(f"frames: {n_frames}")
    traj = trajectory_along(waypoints, SPEED_MS, n_frames, FPS)

    new_id = max(
        (o["object id"] for objs in gt.values() if isinstance(objs, list) for o in objs),
        default=0,
    ) + 100
    print(f"adding synthetic Forklift with object id = {new_id}, replacing static fk {fk_id}")

    template_scale = fk_template.get("3d bounding box scale", [1.5, 3.0, 2.0])
    template_z = float(fk_template.get("3d location", [0,0,0])[2])

    new_gt: dict = {}
    for fid_str, objs in gt.items():
        if not isinstance(objs, list):
            new_gt[fid_str] = objs
            continue
        fid = int(fid_str)
        kept = [o for o in objs if not (o.get("object type") == "Forklift" and int(o.get("object id", -1)) == fk_id)]
        x, y, yaw = traj[fid]
        kept.append({
            "object type": "Forklift",
            "object id": int(new_id),
            "3d location": [float(x), float(y), template_z],
            "3d bounding box scale": list(template_scale),
            "3d bounding box rotation": [0.0, 0.0, float(yaw)],
            "2d bounding box visible": {},
        })
        new_gt[fid_str] = kept

    SYNTH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTH_OUT, "w", encoding="utf-8") as f:
        json.dump(new_gt, f)
    print(f"wrote {SYNTH_OUT}")
    print(f"\nNow re-run extract_tracks.py to include {SYNTH_SCENE_NAME}.")


if __name__ == "__main__":
    main()
