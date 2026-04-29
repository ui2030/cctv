"""Phase 3 — focused mini-pipeline for the synthetic-forklift scene only.

Loads the synth ground_truth.json (created by synth_forklift.py), computes
kinematics for the forklift, pairs with the scene's persons (loaded from the
already-extracted outputs/trajectories.csv), reuses the same predictive-alarm
logic, and prints metrics.

Run this after synth_forklift.py so we don't have to redo extract_tracks.py
on the full 8.9 M-row trajectory CSV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SYNTH_GT = Path("./data/physicalai/MTMC_Tracking_2025/train/Warehouse_006_synth/ground_truth.json")
TRAJ_PATH = Path("./outputs/trajectories.csv")
SOURCE_SCENE = "Warehouse_006"
OUT_PAIRS = Path("./outputs/synth_pairs.csv")
OUT_EVENTS = Path("./outputs/synth_predicted_events.csv")
OUT_METRICS = Path("./outputs/synth_metrics.csv")
FPS = 30.0
HORIZON_SEC = 3.0
HORIZONS = [1.0, 2.0, 3.0]
RADIUS_M = 15.0
GROUND_TRUTH_DIST = 3.0
GROUND_TRUTH_TTC_MAX = 1.5
REACTION_S = 1.0
BRAKE_DECEL = 1.5
MIN_RADIUS = 1.5
GAP_FRAMES = int(FPS)


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
            loc = o["3d location"]; rot = o["3d bounding box rotation"]
            rows.append({
                "frame_id": fid,
                "veh_id": int(o["object id"]),
                "veh_type": "Forklift",
                "veh_x": float(loc[0]), "veh_y": float(loc[1]),
                "veh_yaw": float(rot[2]),
            })
    df = pd.DataFrame(rows).sort_values(["veh_id","frame_id"]).reset_index(drop=True)
    g = df.groupby("veh_id", sort=False)
    dt = g["frame_id"].diff() / FPS
    df["veh_vx"] = g["veh_x"].diff() / dt
    df["veh_vy"] = g["veh_y"].diff() / dt
    df["veh_speed"] = np.hypot(df.veh_vx, df.veh_vy)
    df["veh_x_future"] = df.veh_x + df.veh_vx.fillna(0) * HORIZON_SEC
    df["veh_y_future"] = df.veh_y + df.veh_vy.fillna(0) * HORIZON_SEC
    return df


def load_persons() -> pd.DataFrame:
    cols = ["scene","frame_id","object_id","object_type","x","y","vx","vy","speed","x_future","y_future"]
    it = pd.read_csv(TRAJ_PATH, usecols=cols, chunksize=500_000)
    parts = []
    for chunk in it:
        m = (chunk.scene == SOURCE_SCENE) & (chunk.object_type == "Person")
        if m.any():
            parts.append(chunk.loc[m].drop(columns=["scene","object_type"]))
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={
        "object_id": "ps_id", "x": "ps_x", "y": "ps_y",
        "vx": "ps_vx", "vy": "ps_vy", "speed": "ps_speed",
        "x_future": "ps_x_future", "y_future": "ps_y_future",
    })
    return df


def adaptive_red_radius(v: pd.Series) -> pd.Series:
    v = v.fillna(0).clip(lower=0)
    return MIN_RADIUS + REACTION_S * v + (v ** 2) / (2.0 * BRAKE_DECEL)


def predict_dist(df: pd.DataFrame, T: float) -> pd.Series:
    fx_v = df.veh_x + df.veh_vx.fillna(0) * T
    fy_v = df.veh_y + df.veh_vy.fillna(0) * T
    fx_p = df.ps_x + df.ps_vx.fillna(0) * T
    fy_p = df.ps_y + df.ps_vy.fillna(0) * T
    return np.hypot(fx_p - fx_v, fy_p - fy_v)


def main() -> None:
    fk = load_synth_forklift()
    print(f"forklift frames: {len(fk):,}, mean speed: {fk.veh_speed.mean():.2f} m/s, max: {fk.veh_speed.max():.2f}")
    ps = load_persons()
    print(f"person rows for {SOURCE_SCENE}: {len(ps):,}")

    pairs = fk.merge(ps, on="frame_id")
    pairs["dx"] = pairs.ps_x - pairs.veh_x
    pairs["dy"] = pairs.ps_y - pairs.veh_y
    pairs["distance"] = np.hypot(pairs.dx, pairs.dy)
    pairs = pairs[pairs.distance <= RADIUS_M].copy()
    print(f"pairs within {RADIUS_M} m: {len(pairs):,}")

    veh_vx = pairs.veh_vx.fillna(0); veh_vy = pairs.veh_vy.fillna(0)
    ps_vx = pairs.ps_vx.fillna(0); ps_vy = pairs.ps_vy.fillna(0)
    rel_vx = ps_vx - veh_vx; rel_vy = ps_vy - veh_vy
    safe_d = pairs.distance.replace(0, np.nan)
    pairs["closing_speed"] = -(pairs.dx*rel_vx + pairs.dy*rel_vy) / safe_d
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = pairs.distance / pairs.closing_speed
    pairs["ttc_seconds"] = np.where(pairs.closing_speed > 0, ttc, np.nan)
    pairs["red_radius"] = adaptive_red_radius(pairs.veh_speed)

    pairs["pred_d_t1"] = predict_dist(pairs, HORIZONS[0])
    pairs["pred_d_t2"] = predict_dist(pairs, HORIZONS[1])
    pairs["pred_d_t3"] = predict_dist(pairs, HORIZONS[2])

    moving = pairs.veh_speed.fillna(0).ge(0.3)
    pred_breach = (
        (pairs.distance <= pairs.red_radius)
        | (pairs.pred_d_t1 <= pairs.red_radius)
        | (pairs.pred_d_t2 <= pairs.red_radius)
        | (pairs.pred_d_t3 <= pairs.red_radius)
    )
    pairs["predicted_alarm"] = moving & pred_breach
    pairs["gt_close_encounter"] = moving & (
        (pairs.distance <= GROUND_TRUTH_DIST)
        | ((pairs.ttc_seconds > 0) & (pairs.ttc_seconds <= GROUND_TRUTH_TTC_MAX))
    )

    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_PAIRS, index=False)

    keep = pairs[pairs.predicted_alarm | pairs.gt_close_encounter].copy()
    keep = keep.sort_values(["veh_id","ps_id","frame_id"]).reset_index(drop=True)
    grp = keep.groupby(["veh_id","ps_id"], sort=False)
    keep["gap"] = grp.frame_id.diff()
    keep["start"] = (keep.gap.isna() | (keep.gap > GAP_FRAMES)).astype(int)
    keep["local_evt"] = grp.start.cumsum()
    key = (keep.veh_id.astype(str) + "|" + keep.ps_id.astype(str) + "|" + keep.local_evt.astype(str))
    keep["event_id"] = pd.factorize(key)[0] + 1

    rows = []
    for evt_id, sub in keep.groupby("event_id", sort=False):
        gt_f = sub.loc[sub.gt_close_encounter, "frame_id"]
        pred_f = sub.loc[sub.predicted_alarm, "frame_id"]
        first_gt = gt_f.min() if not gt_f.empty else np.nan
        first_pred = pred_f.min() if not pred_f.empty else np.nan
        if not np.isnan(first_gt) and not np.isnan(first_pred):
            kind = "true_positive"; lead = (first_gt - first_pred) / FPS
        elif not np.isnan(first_gt):
            kind = "missed"; lead = np.nan
        else:
            kind = "false_alarm"; lead = np.nan
        rows.append({
            "event_id": int(evt_id),
            "veh_id": int(sub.veh_id.iloc[0]),
            "ps_id": int(sub.ps_id.iloc[0]),
            "first_pred_frame": first_pred if not np.isnan(first_pred) else None,
            "first_gt_frame": first_gt if not np.isnan(first_gt) else None,
            "n_frames": len(sub),
            "min_distance": float(sub.distance.min()),
            "max_veh_speed": float(sub.veh_speed.fillna(0).max()),
            "kind": kind,
            "lead_time_s": lead,
        })
    events = pd.DataFrame(rows)
    events.to_csv(OUT_EVENTS, index=False)
    print(f"\n{OUT_EVENTS}: {len(events)} events")

    n_tp = (events.kind == "true_positive").sum()
    n_miss = (events.kind == "missed").sum()
    n_fa = (events.kind == "false_alarm").sum()
    n_gt = n_tp + n_miss
    miss_rate = n_miss / max(n_gt, 1)
    avg_lead = events.loc[events.kind == "true_positive", "lead_time_s"].mean()
    hours = (fk.frame_id.max() + 1) / FPS / 3600.0
    fa_per_hour = n_fa / max(hours, 1e-9)

    metrics = pd.DataFrame([{
        "scene": "Warehouse_006_synth",
        "n_pairs": len(pairs),
        "n_true_positive": int(n_tp),
        "n_missed": int(n_miss),
        "n_false_alarm": int(n_fa),
        "miss_rate": float(miss_rate),
        "false_alarms_per_hour": float(fa_per_hour),
        "mean_lead_time_s": float(avg_lead) if pd.notna(avg_lead) else None,
        "scene_hours": float(hours),
        "veh_mean_speed": float(fk.veh_speed.mean()),
        "veh_max_speed": float(fk.veh_speed.max()),
    }])
    metrics.to_csv(OUT_METRICS, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
