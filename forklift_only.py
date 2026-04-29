from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import argparse

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

DATA_ROOT = Path("./data/physicalai")
OUTPUT_DIR = Path("./outputs")
HF_REPO = "nvidia/PhysicalAI-SmartSpaces"
REPO_TYPE = "dataset"


def list_ground_truth_files() -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO, repo_type=REPO_TYPE)
    return sorted([f for f in files if f.endswith("ground_truth.json")])


def download_ground_truth_files(gt_paths: Iterable[str]) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        local_dir=str(DATA_ROOT),
        allow_patterns=list(gt_paths),
    )


def load_ground_truth(gt_path: Path) -> list[dict]:
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    rows = []
    for frame_id_str, obj_list in gt.items():
        if not isinstance(obj_list, list):
            continue
        frame_id = int(frame_id_str)
        for obj in obj_list:
            if obj.get("object type") == "Forklift":
                loc = obj.get("3d location", [0.0, 0.0, 0.0])
                rows.append(
                    {
                        "frame_id": frame_id,
                        "object_id": int(obj["object id"]),
                        "x": float(loc[0]),
                        "y": float(loc[1]),
                        "z": float(loc[2]),
                    }
                )
    return rows


def analyze_scene(scene: str, gt_path: Path) -> tuple[int, int, float]:
    rows = load_ground_truth(gt_path)
    if not rows:
        return 0, 0, 0.0

    df = pd.DataFrame(rows)
    df = df.sort_values(["object_id", "frame_id"]).reset_index(drop=True)
    df["dx"] = df.groupby("object_id")["x"].diff()
    df["dy"] = df.groupby("object_id")["y"].diff()
    df["dt"] = df.groupby("object_id")["frame_id"].diff() / 30.0
    df["speed"] = np.hypot(df["dx"], df["dy"]) / df["dt"]
    df["speed"] = df["speed"].fillna(0.0)

    forklift_count = int(df["object_id"].nunique())
    moving_frames = int((df["speed"] > 0.3).sum())
    max_speed = float(df["speed"].max())
    return forklift_count, moving_frames, max_speed


def save_forklift_tracks(gt_paths: Iterable[str]) -> None:
    records = []
    for rel in gt_paths:
        scene = rel.split("/")[2]
        gt_path = DATA_ROOT / rel
        rows = load_ground_truth(gt_path)
        for row in rows:
            row["scene"] = scene
            records.append(row)
    if records:
        df = pd.DataFrame(records)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / "forklift_tracks.csv", index=False)


def main(arg_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forklift-only scene analyzer")
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated list of scene names to analyze (e.g. Warehouse_000,Warehouse_001).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading ground truth files and use already downloaded data only.",
    )
    args = parser.parse_args(arg_list)

    gt_paths = list_ground_truth_files()
    if not gt_paths:
        raise RuntimeError("No ground_truth.json files found in the dataset repository.")

    if args.scenes:
        target_scenes = {scene.strip() for scene in args.scenes.split(",") if scene.strip()}
        gt_paths = [path for path in gt_paths if path.split("/")[2] in target_scenes]
        if not gt_paths:
            raise RuntimeError(f"No matching scenes found for: {args.scenes}")

    if not args.skip_download:
        download_ground_truth_files(gt_paths)

    summaries = []
    for rel in gt_paths:
        scene = rel.split("/")[2]
        gt_path = DATA_ROOT / rel
        forklift_count, moving_frames, max_speed = analyze_scene(scene, gt_path)
        summaries.append(
            {
                "scene": scene,
                "forklift_count": forklift_count,
                "moving_frames_gt_0_3": moving_frames,
                "max_speed": max_speed,
            }
        )

    df_summary = pd.DataFrame(summaries)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(OUTPUT_DIR / "forklift_scene_summary.csv", index=False)
    save_forklift_tracks(gt_paths)

    print(df_summary.to_string(index=False))
    print("\noutputs/forklift_scene_summary.csv and outputs/forklift_tracks.csv created")


if __name__ == "__main__":
    main()
