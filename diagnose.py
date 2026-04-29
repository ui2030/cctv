"""Filter vehicle-person pairs into 'real alarm' frames using simple thresholds:
  - vehicle is moving (speed >= 0.3 m/s)
  - distance <= 3 m  OR  TTC in (0, 3] s
Then group consecutive risky frames into events and write a summary.

Reads outputs/vehicle_person_pairs.csv, writes outputs/real_alarm_events.csv,
outputs/event_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAIRS_PATH = Path("./outputs/vehicle_person_pairs.csv")
EVENTS_PATH = Path("./outputs/real_alarm_events.csv")
SUMMARY_PATH = Path("./outputs/event_summary.csv")
FPS = 30.0
DIST_M = 3.0
TTC_S = 3.0
GAP_FRAMES = int(FPS)


def main() -> None:
    pairs = pd.read_csv(PAIRS_PATH)
    print(f"전체 pair 행: {len(pairs):,}")
    pairs["veh_moving"] = pairs["veh_speed"].fillna(0) >= 0.3

    moving = pairs[pairs.veh_moving]
    risky = moving[
        (moving.distance <= DIST_M)
        | ((moving.ttc_seconds > 0) & (moving.ttc_seconds <= TTC_S))
    ].copy()
    print(f"움직이는 vehicle pair: {len(moving):,}")
    print(f"위험 frame: {len(risky):,}")
    if risky.empty:
        EVENTS_PATH.write_text("")
        SUMMARY_PATH.write_text("")
        print("위험 frame 없음 — 빈 결과 저장")
        return

    risky = risky.sort_values(["scene","veh_type","veh_id","ps_id","frame_id"]).copy()
    grp = risky.groupby(["scene","veh_type","veh_id","ps_id"], sort=False)
    risky["gap"] = grp["frame_id"].diff()
    risky["new_event"] = (risky["gap"].isna() | (risky["gap"] > GAP_FRAMES)).astype(int)
    risky["local_event"] = grp["new_event"].cumsum()
    keys = (risky.scene.astype(str) + "|" + risky.veh_type.astype(str) + "|"
            + risky.veh_id.astype(str) + "|" + risky.ps_id.astype(str)
            + "|" + risky.local_event.astype(str))
    risky["event_id"] = pd.factorize(keys)[0] + 1
    print(f"고유 위험 이벤트: {risky.event_id.nunique():,}")

    summary = (risky.groupby("event_id").agg(
        scene=("scene","first"),
        veh_type=("veh_type","first"),
        veh_id=("veh_id","first"),
        ps_id=("ps_id","first"),
        start_frame=("frame_id","min"),
        end_frame=("frame_id","max"),
        n_frames=("frame_id","count"),
        min_distance=("distance","min"),
        min_ttc=("ttc_seconds","min"),
        peak_closing_speed=("closing_speed","max"),
        veh_max_speed=("veh_speed","max"),
        ps_max_speed=("ps_speed","max"),
    ).reset_index())
    summary["duration_s"] = summary.n_frames / FPS

    risky.to_csv(EVENTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n{EVENTS_PATH}: {len(risky):,} rows")
    print(f"{SUMMARY_PATH}: {len(summary):,} events")
    print("\nVehicle 타입별 이벤트 분포:")
    print(summary.veh_type.value_counts())
    print("\n이벤트 지속시간 (초):")
    print(summary.duration_s.describe().round(2))


if __name__ == "__main__":
    main()
