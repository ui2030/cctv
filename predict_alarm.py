"""Predictive alarm with adaptive red zone and proper safety metrics.

For every (scene, vehicle, person) frame:
  1. predict each agent's position at horizon T using constant-velocity (CV)
     extrapolation; optionally include yaw-rate for the vehicle (turn intent).
  2. compute predicted distance at horizon T.
  3. compute adaptive red-zone radius r(v) based on vehicle speed
     (reaction + simple braking proxy).
  4. trigger predicted_alarm if predicted_distance <= r(v) at any horizon in
     HORIZONS or if the *current* distance is already <= MIN_RADIUS.

Reads outputs/vehicle_person_pairs.csv (must be produced by compute_pairs.py
with veh_x_future / ps_x_future columns), writes
  outputs/predicted_alarm_events.csv  — per-frame predicted alarm with lead info
  outputs/predict_metrics.csv         — false-alarm/hour, miss rate, mean lead time
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAIRS_PATH = Path("./outputs/vehicle_person_pairs.csv")
ALARM_PATH = Path("./outputs/predicted_alarm_events.csv")
METRICS_PATH = Path("./outputs/predict_metrics.csv")
FPS = 30.0
HORIZONS = [1.0, 2.0, 3.0]            # seconds
GROUND_TRUTH_DIST = 3.0               # m  -- "true close encounter" definition
GROUND_TRUTH_TTC_MAX = 1.5            # s  -- and either close OR closing within 1.5s
REACTION_S = 1.0                      # operator reaction time
BRAKE_DECEL = 1.5                     # m/s^2 conservative for indoor industrial vehicle
MIN_RADIUS = 1.5                      # never below this base radius
GAP_FRAMES = int(FPS)


def adaptive_red_radius(veh_speed: pd.Series) -> pd.Series:
    v = veh_speed.fillna(0.0).clip(lower=0.0)
    return MIN_RADIUS + REACTION_S * v + (v ** 2) / (2.0 * BRAKE_DECEL)


def predict_distance_at(pairs: pd.DataFrame, T: float) -> pd.Series:
    veh_vx = pairs.veh_vx.fillna(0.0); veh_vy = pairs.veh_vy.fillna(0.0)
    ps_vx = pairs.ps_vx.fillna(0.0); ps_vy = pairs.ps_vy.fillna(0.0)
    fx_v = pairs.veh_x + veh_vx * T
    fy_v = pairs.veh_y + veh_vy * T
    fx_p = pairs.ps_x + ps_vx * T
    fy_p = pairs.ps_y + ps_vy * T
    return np.hypot(fx_p - fx_v, fy_p - fy_v)


def label_events(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    df = df.sort_values(key_cols + ["frame_id"]).copy()
    grp = df.groupby(key_cols, sort=False)
    gaps = grp["frame_id"].diff()
    starts = (gaps.isna() | (gaps > GAP_FRAMES)).astype(int)
    local = grp["new_event"].cumsum() if False else None  # placeholder
    df["__start"] = starts
    df["__local"] = grp["__start"].cumsum()
    keys = "|".join("{}").join([])
    keys = df[key_cols].astype(str).agg("|".join, axis=1) + "|" + df["__local"].astype(str)
    return pd.Series(pd.factorize(keys)[0] + 1, index=df.index)


def main() -> None:
    df = pd.read_csv(PAIRS_PATH)
    print(f"pairs: {len(df):,}")

    df["red_radius"] = adaptive_red_radius(df["veh_speed"])
    df["pred_dist_t1"] = predict_distance_at(df, HORIZONS[0])
    df["pred_dist_t2"] = predict_distance_at(df, HORIZONS[1])
    df["pred_dist_t3"] = predict_distance_at(df, HORIZONS[2])

    pred_breach = (
        (df.distance <= df.red_radius)
        | (df.pred_dist_t1 <= df.red_radius)
        | (df.pred_dist_t2 <= df.red_radius)
        | (df.pred_dist_t3 <= df.red_radius)
    )
    df["predicted_alarm"] = pred_breach & df.veh_speed.fillna(0).ge(0.3)

    df["gt_close_encounter"] = (
        df.veh_speed.fillna(0).ge(0.3)
        & (
            (df.distance <= GROUND_TRUTH_DIST)
            | ((df.ttc_seconds > 0) & (df.ttc_seconds <= GROUND_TRUTH_TTC_MAX))
        )
    )

    keep = df[df.predicted_alarm | df.gt_close_encounter].copy()
    keep = keep.sort_values(["scene","veh_type","veh_id","ps_id","frame_id"]).reset_index(drop=True)

    grp = keep.groupby(["scene","veh_type","veh_id","ps_id"], sort=False)
    keep["gap"] = grp["frame_id"].diff()
    keep["start"] = (keep["gap"].isna() | (keep["gap"] > GAP_FRAMES)).astype(int)
    keep["local_evt"] = grp["start"].cumsum()
    key = (keep.scene.astype(str) + "|" + keep.veh_type.astype(str) + "|"
           + keep.veh_id.astype(str) + "|" + keep.ps_id.astype(str)
           + "|" + keep.local_evt.astype(str))
    keep["event_id"] = pd.factorize(key)[0] + 1

    rows = []
    for evt_id, sub in keep.groupby("event_id", sort=False):
        gt_frames = sub.loc[sub.gt_close_encounter, "frame_id"]
        pred_frames = sub.loc[sub.predicted_alarm, "frame_id"]
        if gt_frames.empty and pred_frames.empty:
            continue
        first_gt = gt_frames.min() if not gt_frames.empty else np.nan
        first_pred = pred_frames.min() if not pred_frames.empty else np.nan
        if not np.isnan(first_gt) and not np.isnan(first_pred):
            kind = "true_positive"
            lead_s = (first_gt - first_pred) / FPS
        elif not np.isnan(first_gt) and np.isnan(first_pred):
            kind = "missed"
            lead_s = np.nan
        else:
            kind = "false_alarm"
            lead_s = np.nan
        rows.append({
            "event_id": int(evt_id),
            "scene": sub.scene.iloc[0],
            "veh_type": sub.veh_type.iloc[0],
            "veh_id": int(sub.veh_id.iloc[0]),
            "ps_id": int(sub.ps_id.iloc[0]),
            "first_pred_frame": first_pred if not np.isnan(first_pred) else None,
            "first_gt_frame": first_gt if not np.isnan(first_gt) else None,
            "n_frames": len(sub),
            "min_distance": float(sub.distance.min()),
            "min_red_radius": float(sub.red_radius.min()),
            "max_veh_speed": float(sub.veh_speed.fillna(0).max()),
            "kind": kind,
            "lead_time_s": lead_s,
        })
    events = pd.DataFrame(rows)

    n_tp = (events.kind == "true_positive").sum()
    n_miss = (events.kind == "missed").sum()
    n_fa = (events.kind == "false_alarm").sum()
    n_gt_total = n_tp + n_miss
    miss_rate = n_miss / max(n_gt_total, 1)
    avg_lead = events.loc[events.kind == "true_positive", "lead_time_s"].mean()

    n_scenes = df.scene.nunique()
    total_hours = n_scenes * (df.frame_id.max() + 1) / FPS / 3600.0
    fa_per_hour = n_fa / max(total_hours, 1e-9)

    metrics = pd.DataFrame([{
        "horizons_s": ",".join(str(h) for h in HORIZONS),
        "reaction_s": REACTION_S,
        "brake_decel": BRAKE_DECEL,
        "min_radius": MIN_RADIUS,
        "n_true_positive": int(n_tp),
        "n_missed": int(n_miss),
        "n_false_alarm": int(n_fa),
        "miss_rate": float(miss_rate),
        "false_alarms_per_hour": float(fa_per_hour),
        "mean_lead_time_s": float(avg_lead) if pd.notna(avg_lead) else None,
        "total_scene_hours": float(total_hours),
    }])

    ALARM_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(ALARM_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)

    print(f"\n{ALARM_PATH}: {len(events)} events")
    print(f"{METRICS_PATH}:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
