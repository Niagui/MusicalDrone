"""Evaluate generated drone choreography folders.

Usage:
    python evaluate.py data/intenseSpanish.mp3
    python evaluate.py evaluation/song01 --out evaluation/song01/results

The input can be either:
    1. a single run folder containing trajectory.csv and optional json/*.json
    2. a song folder whose immediate subfolders are conditions, each with a
       trajectory.csv file

Outputs are written to <song_folder>/evaluation by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ANCHORS = ["happy", "sad", "sleepy", "brave", "grumpy", "scared", "shy"]
FEATURE_COLUMNS = [
    "mean_speed",
    "max_speed",
    "mean_accel",
    "mean_jerk",
    "mean_altitude",
    "altitude_range",
    "swarm_spread",
    "min_pairwise_dist",
    "boundary_margin",
    "centroid_path_length",
    "heading_change",
    "formation_change",
]
BOUNDS_LO = np.array([-1.05, -1.05, 0.5], dtype=float)
BOUNDS_HI = np.array([1.05, 1.05, 1.2], dtype=float)


@dataclass
class Run:
    name: str
    path: Path
    trajectory_path: Path


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def discover_runs(song_folder: Path) -> list[Run]:
    direct = find_trajectory(song_folder)
    if direct is not None:
        return [Run(song_folder.name, song_folder, direct)]

    runs: list[Run] = []
    for child in sorted(song_folder.iterdir()):
        if not child.is_dir():
            continue
        child_traj = find_trajectory(child)
        if child_traj is not None:
            runs.append(Run(child.name, child, child_traj))
    if not runs:
        raise FileNotFoundError(
            f"No trajectory.csv found in {song_folder} or its immediate subfolders"
        )
    return runs


def find_trajectory(folder: Path) -> Path | None:
    for name in ("trajectory.csv", "trajectories.csv", "newtraj.csv"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def read_trajectory(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(v) for v in row[:11]])
            except ValueError:
                # Header row.
                continue
    if not rows:
        raise ValueError(f"No numeric trajectory rows found in {path}")
    return np.asarray(rows, dtype=float)


def load_segments(run_path: Path, traj: np.ndarray) -> list[dict[str, Any]]:
    json_dir = run_path / "json"
    clap_weights = load_json(json_dir / "clap_weights.json", [])
    beat_segments = load_json(json_dir / "k_beat_segments.json", [])
    phrase_plan = load_json(json_dir / "phrase_plan.json", {})
    phrases = phrase_plan.get("phrases", []) if isinstance(phrase_plan, dict) else []

    segments: list[dict[str, Any]] = []
    if clap_weights:
        for idx, item in enumerate(clap_weights):
            weights = list(item.get("weights", []))
            dominant_idx = int(np.argmax(weights)) if weights else -1
            phrase = phrases[idx] if idx < len(phrases) else {}
            segments.append(
                {
                    "segment_idx": idx,
                    "start_s": float(item["start"]),
                    "end_s": float(item["end"]),
                    "weights": weights,
                    "dominant_anchor": ANCHORS[dominant_idx]
                    if 0 <= dominant_idx < len(ANCHORS)
                    else "",
                    "phrase_index": phrase.get("phrase_index", idx),
                    "section_index": phrase.get("section_index", ""),
                    "section_role": phrase.get("section_role", ""),
                    "motion_mode": phrase.get("motion_mode", ""),
                    "height_level": phrase.get("height_level", ""),
                    "speed_level": phrase.get("speed_level", ""),
                    "vertical_trend": phrase.get("vertical_trend", ""),
                    "transition_style": phrase.get("transition_style", ""),
                }
            )
        return segments

    if beat_segments:
        for idx, item in enumerate(beat_segments):
            start_s, end_s = item
            segments.append(
                {
                    "segment_idx": idx,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "weights": [],
                    "dominant_anchor": "",
                    "phrase_index": idx,
                    "section_index": "",
                    "section_role": "",
                    "motion_mode": "",
                    "height_level": "",
                    "speed_level": "",
                    "vertical_trend": "",
                    "transition_style": "",
                }
            )
        return segments

    t_min = float(np.min(traj[:, 1]))
    t_max = float(np.max(traj[:, 1]))
    return [
        {
            "segment_idx": 0,
            "start_s": t_min,
            "end_s": t_max,
            "weights": [],
            "dominant_anchor": "",
            "phrase_index": 0,
            "section_index": "",
            "section_role": "",
            "motion_mode": "",
            "height_level": "",
            "speed_level": "",
            "vertical_trend": "",
            "transition_style": "",
        }
    ]


def pairwise_distances(points: np.ndarray) -> list[float]:
    distances: list[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(float(np.linalg.norm(points[i] - points[j])))
    return distances


def heading_change(vectors: np.ndarray) -> float:
    changes: list[float] = []
    for drone_id in np.unique(vectors[:, 0]):
        drone = vectors[vectors[:, 0] == drone_id]
        drone = drone[np.argsort(drone[:, 1])]
        v = drone[:, 5:8]
        for prev, cur in zip(v[:-1], v[1:]):
            prev_norm = float(np.linalg.norm(prev))
            cur_norm = float(np.linalg.norm(cur))
            if prev_norm < 1e-9 or cur_norm < 1e-9:
                continue
            cos_theta = float(np.dot(prev, cur) / (prev_norm * cur_norm))
            changes.append(math.acos(max(-1.0, min(1.0, cos_theta))))
    return float(mean(changes)) if changes else 0.0


def segment_features(traj: np.ndarray, segment: dict[str, Any]) -> dict[str, float]:
    start_s = float(segment["start_s"])
    end_s = float(segment["end_s"])
    rows = traj[(traj[:, 1] >= start_s) & (traj[:, 1] < end_s)]
    if rows.size == 0:
        return {key: 0.0 for key in FEATURE_COLUMNS}

    speeds = np.linalg.norm(rows[:, 5:8], axis=1)
    accels = np.linalg.norm(rows[:, 8:11], axis=1)
    positions = rows[:, 2:5]

    jerks: list[float] = []
    for drone_id in np.unique(rows[:, 0]):
        drone = rows[rows[:, 0] == drone_id]
        drone = drone[np.argsort(drone[:, 1])]
        if len(drone) < 2:
            continue
        dt = np.diff(drone[:, 1])
        da = np.diff(drone[:, 8:11], axis=0)
        valid = dt > 1e-9
        if np.any(valid):
            jerks.extend(np.linalg.norm(da[valid] / dt[valid, None], axis=1).tolist())

    by_time: dict[float, np.ndarray] = {}
    for t in np.unique(rows[:, 1]):
        by_time[float(t)] = rows[rows[:, 1] == t][:, 2:5]

    spreads: list[float] = []
    min_pairwise = math.inf
    centroids: list[np.ndarray] = []
    formation_vectors: list[np.ndarray] = []
    for t in sorted(by_time):
        pts = by_time[t]
        centroid = np.mean(pts, axis=0)
        centroids.append(centroid)
        spreads.append(float(np.mean(np.linalg.norm(pts - centroid, axis=1))))
        distances = pairwise_distances(pts)
        if distances:
            min_pairwise = min(min_pairwise, min(distances))
            formation_vectors.append(np.asarray(distances, dtype=float))

    centroid_path_length = 0.0
    if len(centroids) > 1:
        centroid_path_length = float(
            np.sum(np.linalg.norm(np.diff(np.vstack(centroids), axis=0), axis=1))
        )

    formation_change = 0.0
    formation_deltas: list[float] = []
    for prev, cur in zip(formation_vectors[:-1], formation_vectors[1:]):
        if len(prev) == len(cur):
            formation_deltas.append(float(np.linalg.norm(cur - prev)))
    if formation_deltas:
        formation_change = float(mean(formation_deltas))

    lower_margin = positions - BOUNDS_LO
    upper_margin = BOUNDS_HI - positions
    boundary_margin = float(np.min(np.hstack([lower_margin, upper_margin])))

    return {
        "mean_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "mean_accel": float(np.mean(accels)),
        "mean_jerk": float(mean(jerks)) if jerks else 0.0,
        "mean_altitude": float(np.mean(rows[:, 4])),
        "altitude_range": float(np.max(rows[:, 4]) - np.min(rows[:, 4])),
        "swarm_spread": float(mean(spreads)) if spreads else 0.0,
        "min_pairwise_dist": float(min_pairwise) if math.isfinite(min_pairwise) else 0.0,
        "boundary_margin": boundary_margin,
        "centroid_path_length": centroid_path_length,
        "heading_change": heading_change(rows),
        "formation_change": formation_change,
    }


def evaluate_run(run: Run) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traj = read_trajectory(run.trajectory_path)
    segments = load_segments(run.path, traj)

    feature_rows: list[dict[str, Any]] = []
    for segment in segments:
        features = segment_features(traj, segment)
        row = {
            "condition": run.name,
            "segment_idx": segment["segment_idx"],
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
            "dominant_anchor": segment["dominant_anchor"],
            "phrase_index": segment["phrase_index"],
            "section_index": segment["section_index"],
            "section_role": segment["section_role"],
            "motion_mode": segment["motion_mode"],
            "height_level": segment["height_level"],
            "speed_level": segment["speed_level"],
            "vertical_trend": segment["vertical_trend"],
            "transition_style": segment["transition_style"],
        }
        for idx, anchor in enumerate(ANCHORS):
            weights = segment.get("weights", [])
            row[f"weight_{anchor}"] = float(weights[idx]) if idx < len(weights) else ""
        row.update(features)
        feature_rows.append(row)

    safety = {
        "min_pairwise_dist": min(row["min_pairwise_dist"] for row in feature_rows),
        "min_boundary_margin": min(row["boundary_margin"] for row in feature_rows),
        "max_speed": max(row["max_speed"] for row in feature_rows),
        "max_accel": max(row["mean_accel"] for row in feature_rows),
        "segments": len(feature_rows),
        "duration_s": float(np.max(traj[:, 1]) - np.min(traj[:, 1])),
        "drone_count": int(len(np.unique(traj[:, 0]))),
    }
    return feature_rows, safety


def zscore_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows])
    std = np.std(matrix, axis=0)
    std[std < 1e-9] = 1.0
    return (matrix - np.mean(matrix, axis=0)) / std


def summarize(all_rows: list[dict[str, Any]], safety_by_run: dict[str, Any]) -> dict[str, Any]:
    z = zscore_matrix(all_rows)
    for row, values in zip(all_rows, z):
        for col, value in zip(FEATURE_COLUMNS, values):
            row[f"z_{col}"] = float(value)

    by_condition: dict[str, list[int]] = {}
    for idx, row in enumerate(all_rows):
        by_condition.setdefault(str(row["condition"]), []).append(idx)

    condition_summary: dict[str, Any] = {}
    for condition, indices in by_condition.items():
        values = z[indices]
        transition_costs = [
            float(np.linalg.norm(values[i] - values[i - 1]))
            for i in range(1, len(values))
        ]
        expressive_range = float(np.mean(np.var(values, axis=0))) if len(values) > 1 else 0.0
        pairwise: list[float] = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairwise.append(float(np.linalg.norm(values[i] - values[j])))

        rows = [all_rows[i] for i in indices]
        phrase_groups: dict[str, list[np.ndarray]] = {}
        for local_idx, row in enumerate(rows):
            key = str(row.get("section_index") or row.get("phrase_index") or local_idx)
            phrase_groups.setdefault(key, []).append(values[local_idx])
        within_phrase_vars = [
            float(np.mean(np.var(np.vstack(group), axis=0)))
            for group in phrase_groups.values()
            if len(group) > 1
        ]
        mode_switches = 0
        modes = [str(row.get("motion_mode", "")) for row in rows]
        for prev, cur in zip(modes[:-1], modes[1:]):
            if prev and cur and prev != cur:
                mode_switches += 1

        condition_summary[condition] = {
            "expressive_range_mean_feature_variance": expressive_range,
            "mean_pairwise_segment_distance": float(mean(pairwise)) if pairwise else 0.0,
            "responsiveness_transition_cost": float(mean(transition_costs))
            if transition_costs
            else 0.0,
            "phrase_within_group_variance": float(mean(within_phrase_vars))
            if within_phrase_vars
            else 0.0,
            "mode_switches": mode_switches,
            "safety": safety_by_run[condition],
        }

    return {
        "feature_columns": FEATURE_COLUMNS,
        "condition_summary": condition_summary,
        "notes": {
            "expressive_range": "Mean variance across z-scored motion features; higher means broader motion vocabulary.",
            "responsiveness_transition_cost": "Mean Euclidean feature change between adjacent segments.",
            "phrase_within_group_variance": "Mean z-scored feature variance within section_index/phrase groups; lower means more coherent local grouping.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_anchor_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        for anchor in ANCHORS:
            group = [
                row
                for row in rows
                if row["condition"] == condition and row.get("dominant_anchor") == anchor
            ]
            if not group:
                continue
            summary = {
                "condition": condition,
                "dominant_anchor": anchor,
                "n_segments": len(group),
            }
            for col in FEATURE_COLUMNS:
                summary[col] = float(mean(float(row[col]) for row in group))
            summary_rows.append(summary)
    write_csv(out_dir / "emotion_feature_means.csv", summary_rows)


def maybe_write_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for condition in sorted({str(row["condition"]) for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        if not group:
            continue
        matrix = np.asarray([[float(row[f"z_{col}"]) for row in group] for col in FEATURE_COLUMNS])
        fig, ax = plt.subplots(figsize=(max(8, len(group) * 0.35), 5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(f"{condition}: motion-feature heatmap")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Feature")
        ax.set_yticks(range(len(FEATURE_COLUMNS)))
        ax.set_yticklabels(FEATURE_COLUMNS)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([str(row["segment_idx"]) for row in group], rotation=90)
        fig.colorbar(im, ax=ax, label="z-score")
        fig.tight_layout()
        fig.savefig(out_dir / f"{condition}_motion_heatmap.png", dpi=180)
        plt.close(fig)

    summary_path = out_dir / "emotion_feature_means.csv"
    if summary_path.exists():
        with summary_path.open() as f:
            emotion_rows = list(csv.DictReader(f))
        for condition in sorted({row["condition"] for row in emotion_rows}):
            group = [row for row in emotion_rows if row["condition"] == condition]
            matrix = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in group])
            if matrix.size == 0:
                continue
            col_min = np.min(matrix, axis=0)
            col_max = np.max(matrix, axis=0)
            denom = np.where((col_max - col_min) < 1e-9, 1.0, col_max - col_min)
            normalized = (matrix - col_min) / denom
            fig, ax = plt.subplots(figsize=(9, max(3, len(group) * 0.45)))
            im = ax.imshow(normalized, aspect="auto", cmap="magma")
            ax.set_title(f"{condition}: emotion x motion features")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Dominant anchor")
            ax.set_xticks(range(len(FEATURE_COLUMNS)))
            ax.set_xticklabels(FEATURE_COLUMNS, rotation=60, ha="right")
            ax.set_yticks(range(len(group)))
            ax.set_yticklabels([row["dominant_anchor"] for row in group])
            fig.colorbar(im, ax=ax, label="condition-normalized value")
            fig.tight_layout()
            fig.savefig(out_dir / f"{condition}_emotion_feature_heatmap.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song_folder", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    song_folder = args.song_folder.resolve()
    out_dir = args.out.resolve() if args.out else song_folder / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(song_folder)
    all_rows: list[dict[str, Any]] = []
    safety_by_run: dict[str, Any] = {}
    for run in runs:
        rows, safety = evaluate_run(run)
        all_rows.extend(rows)
        safety_by_run[run.name] = safety

    summary = summarize(all_rows, safety_by_run)
    write_csv(out_dir / "segment_features.csv", all_rows)
    write_anchor_summary(out_dir, all_rows)
    with (out_dir / "summary_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    maybe_write_plots(out_dir, all_rows)

    print(f"Evaluated {len(runs)} run(s), {len(all_rows)} segment(s)")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
