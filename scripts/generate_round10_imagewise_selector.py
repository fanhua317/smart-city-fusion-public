from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from fusion_common import ensure_dir, gray_float, list_pairs, load_image
from generate_round7_leader_target import (
    METRICS,
    calibration_from_official,
    load_leaderboard,
    metric_rank,
    predict_official,
)
from local_metrics import HIGHER_IS_BETTER, metrics_for_pair


MANUAL_POOL = [
    "results/candidates_round4/04_micro_edge_broad44",
    "results/candidates_round4/02_micro_edge_balanced43",
    "results/candidates_round4/10_source_edge_avg_low30",
    "results/candidates_round4/11_source_edge_broad_detail",
    "results/candidates_round2/33_mix_avg21_45",
    "results/candidates/18_low_artifact_color",
    "results/candidates/21_crossblend_35_color",
    "results/candidates_round2/11_vi_only",
    "results/candidates_round2/19_vi_anchor_safe_qabf",
    "results/candidates_round5/08_transfer18_mid",
    "results/candidates_round5/15_vi_repair_with18",
    "results/candidates_round7/13_project_leader_struct",
    "results/candidates_round9/18_source_pick_s450",
    "results/candidates_round9/17_source_pick_s442",
    "results/candidates_round9/16_source_pick_s434",
    "results/candidates_remote/crossfuse_target",
]


def candidate_dirs() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for metrics_path in sorted(Path("results").glob("candidates*/**/metrics_summary.json")):
        path = metrics_path.parent
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def predicted_rank(local: dict[str, float], leaderboard: list[dict[str, str]], calibration: dict[str, tuple[float, float, float]]) -> float:
    pred = predict_official(local, calibration)
    ranks = [metric_rank(leaderboard, metric, pred[metric]) for metric in METRICS]
    return float(np.mean(ranks))


def predicted_detail(local: dict[str, float], leaderboard: list[dict[str, str]], calibration: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    pred = predict_official(local, calibration)
    out = {f"Pred_{metric}": float(pred[metric]) for metric in METRICS}
    out["PredInsertedRank"] = float(np.mean([metric_rank(leaderboard, metric, pred[metric]) for metric in METRICS]))
    return out


def aggregate_pool_scores(
    dirs: list[Path], leaderboard: list[dict[str, str]], calibration: dict[str, tuple[float, float, float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in dirs:
        metrics_path = path / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        local = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not all(metric in local for metric in METRICS):
            continue
        rank = predicted_rank({metric: float(local[metric]) for metric in METRICS}, leaderboard, calibration)
        row: dict[str, Any] = {"candidate": path.name, "path": str(path), "PoolPredInsertedRank": rank}
        row.update({metric: float(local[metric]) for metric in METRICS})
        rows.append(row)
    rows.sort(key=lambda row: float(row["PoolPredInsertedRank"]))
    return rows


def choose_pool(
    all_dirs: list[Path],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, tuple[float, float, float]],
    pool_size: int,
) -> list[Path]:
    scored = aggregate_pool_scores(all_dirs, leaderboard, calibration)
    chosen: list[Path] = []
    seen: set[Path] = set()
    for row in scored[:pool_size]:
        path = Path(str(row["path"]))
        chosen.append(path)
        seen.add(path.resolve())
    for raw in MANUAL_POOL:
        path = Path(raw)
        if path.exists() and path.resolve() not in seen:
            chosen.append(path)
            seen.add(path.resolve())
    return chosen


def complete_dirs(paths: list[Path], names: list[str]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if all((path / name).exists() for name in names):
            out.append(path)
    return out


def rank_points(values: np.ndarray, higher: bool) -> np.ndarray:
    order = np.argsort(values)
    if higher:
        order = order[::-1]
    points = np.empty_like(values, dtype=np.float64)
    # Best value gets 1.0, worst gets 0.0.
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        points[idx] = 1.0 - rank / denom
    return points


def selector_by_rank_points(metrics_cube: np.ndarray, metric_index: dict[str, int], profile: str) -> np.ndarray:
    weights_by_profile = {
        "balanced": {
            "AG": 1.0,
            "CC": 0.9,
            "EN": 0.8,
            "MI": 0.3,
            "MSE": 0.8,
            "Nabf": 0.9,
            "PSNR": 0.3,
            "Qabf": 1.4,
            "SCD": 1.0,
            "SD": 0.8,
            "SF": 1.0,
            "SSIM": 1.1,
            "VIF": 0.4,
        },
        "detail": {
            "AG": 1.7,
            "CC": 0.6,
            "EN": 1.0,
            "MI": 0.2,
            "MSE": 0.6,
            "Nabf": 0.7,
            "PSNR": 0.2,
            "Qabf": 1.8,
            "SCD": 1.0,
            "SD": 1.2,
            "SF": 1.6,
            "SSIM": 0.8,
            "VIF": 0.3,
        },
        "safe_qabf": {
            "AG": 0.9,
            "CC": 0.9,
            "EN": 0.7,
            "MI": 0.2,
            "MSE": 0.9,
            "Nabf": 1.5,
            "PSNR": 0.3,
            "Qabf": 2.2,
            "SCD": 0.9,
            "SD": 0.7,
            "SF": 0.9,
            "SSIM": 1.2,
            "VIF": 0.4,
        },
    }
    weights = weights_by_profile[profile]
    n_images, n_candidates, _ = metrics_cube.shape
    selection = np.zeros(n_images, dtype=np.int32)
    for image_idx in range(n_images):
        score = np.zeros(n_candidates, dtype=np.float64)
        for metric, weight in weights.items():
            values = metrics_cube[image_idx, :, metric_index[metric]]
            score += weight * rank_points(values, HIGHER_IS_BETTER[metric])
        selection[image_idx] = int(np.argmax(score))
    return selection


def aggregate_metrics(metrics_cube: np.ndarray, selection: np.ndarray, metric_index: dict[str, int]) -> dict[str, float]:
    picked = metrics_cube[np.arange(metrics_cube.shape[0]), selection, :]
    return {metric: float(np.mean(picked[:, metric_index[metric]])) for metric in METRICS}


def objective(
    local: dict[str, float],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, tuple[float, float, float]],
    mode: str,
) -> float:
    pred = predict_official(local, calibration)
    ranks = [metric_rank(leaderboard, metric, pred[metric]) for metric in METRICS]
    rank = float(np.mean(ranks))
    detail_loss = (
        max(0.0, 6.15 - pred["AG"]) / 1.4
        + max(0.0, 7.40 - pred["SF"]) / 1.2
        + max(0.0, 44.0 - pred["SD"]) / 4.5
        + max(0.0, 0.58 - pred["Qabf"]) * 8.0
    )
    structure_loss = (
        max(0.0, 0.735 - pred["SSIM"]) * 7.0
        + max(0.0, pred["Nabf"] - 0.040) * 12.0
        + max(0.0, 0.620 - pred["CC"]) * 6.0
        + max(0.0, 1.68 - pred["SCD"]) * 4.0
    )
    if mode == "rank":
        return rank
    if mode == "balanced":
        return rank + 0.25 * detail_loss + 0.35 * structure_loss
    if mode == "detail":
        return rank + 0.45 * detail_loss + 0.20 * structure_loss
    if mode == "safe":
        return rank + 0.20 * detail_loss + 0.55 * structure_loss
    raise ValueError(f"unknown objective mode: {mode}")


def greedy_improve(
    initial: np.ndarray,
    metrics_cube: np.ndarray,
    metric_index: dict[str, int],
    leaderboard: list[dict[str, str]],
    calibration: dict[str, tuple[float, float, float]],
    mode: str,
    passes: int,
) -> tuple[np.ndarray, float]:
    selection = initial.copy()
    n_images, n_candidates, _ = metrics_cube.shape
    agg_vec = metrics_cube[np.arange(n_images), selection, :].mean(axis=0)
    local = {metric: float(agg_vec[metric_index[metric]]) for metric in METRICS}
    best = objective(local, leaderboard, calibration, mode)
    for _ in range(passes):
        changed = False
        for image_idx in range(n_images):
            current_idx = int(selection[image_idx])
            current_vec = metrics_cube[image_idx, current_idx, :]
            for candidate_idx in range(n_candidates):
                if candidate_idx == current_idx:
                    continue
                trial_vec = agg_vec + (metrics_cube[image_idx, candidate_idx, :] - current_vec) / n_images
                trial_local = {metric: float(trial_vec[metric_index[metric]]) for metric in METRICS}
                trial = objective(trial_local, leaderboard, calibration, mode)
                if trial + 1e-9 < best:
                    selection[image_idx] = candidate_idx
                    agg_vec = trial_vec
                    current_idx = candidate_idx
                    current_vec = metrics_cube[image_idx, candidate_idx, :]
                    best = trial
                    changed = True
        if not changed:
            break
    return selection, best


def write_candidate(
    out_dir: Path,
    names: list[str],
    pool: list[Path],
    selection: np.ndarray,
    local: dict[str, float],
    extra: dict[str, Any],
) -> None:
    ensure_dir(out_dir)
    manifest = []
    for image_idx, name in enumerate(names):
        source = pool[int(selection[image_idx])] / name
        shutil.copy2(source, out_dir / name)
        manifest.append({"image": name, "source": str(pool[int(selection[image_idx])]), "source_candidate": pool[int(selection[image_idx])].name})
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics_summary.json").write_text(json.dumps(local, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "params.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round10")
    parser.add_argument("--score-out", default="results/round10_local_scores.csv")
    parser.add_argument("--pool-out", default="results/round10_pool.csv")
    parser.add_argument("--pool-size", type=int, default=45)
    parser.add_argument("--leaderboard", default="leaderboard_20260605_234547.csv")
    parser.add_argument("--official-csv", default="results/score_tables/current_official_scores.csv")
    args = parser.parse_args()

    names = list_pairs(args.ir_dir, args.vi_dir)
    leaderboard = load_leaderboard(Path(args.leaderboard))
    calibration = calibration_from_official(Path(args.official_csv))
    all_dirs = complete_dirs(candidate_dirs(), names)
    pool = choose_pool(all_dirs, leaderboard, calibration, args.pool_size)
    pool = complete_dirs(pool, names)
    if not pool:
        raise RuntimeError("no complete candidate dirs")
    print(f"pool candidates: {len(pool)}")

    pool_rows = aggregate_pool_scores(pool, leaderboard, calibration)
    pool_out = Path(args.pool_out)
    ensure_dir(pool_out.parent)
    pool_fields = ["candidate", "path", "PoolPredInsertedRank"] + METRICS
    with pool_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pool_fields)
        writer.writeheader()
        for row in pool_rows:
            writer.writerow({key: row.get(key, "") for key in pool_fields})

    metric_index = {metric: idx for idx, metric in enumerate(METRICS)}
    metrics_cube = np.zeros((len(names), len(pool), len(METRICS)), dtype=np.float64)
    for image_idx, name in enumerate(names):
        ir = gray_float(load_image(Path(args.ir_dir) / name))
        vi = gray_float(load_image(Path(args.vi_dir) / name))
        for candidate_idx, candidate_dir in enumerate(pool):
            fused = gray_float(load_image(candidate_dir / name))
            row = metrics_for_pair(ir, vi, fused)
            for metric in METRICS:
                metrics_cube[image_idx, candidate_idx, metric_index[metric]] = row[metric]
        if (image_idx + 1) % 10 == 0 or image_idx + 1 == len(names):
            print(f"computed {image_idx + 1}/{len(names)}")

    out_root = ensure_dir(args.out_root)
    rows: list[dict[str, Any]] = []

    base_by_name = {path.name: idx for idx, path in enumerate(pool)}
    starts: list[tuple[str, np.ndarray, str]] = []
    for profile in ["balanced", "detail", "safe_qabf"]:
        starts.append((f"rankpoints_{profile}", selector_by_rank_points(metrics_cube, metric_index, profile), profile))
    for name in ["04_micro_edge_broad44", "02_micro_edge_balanced43", "33_mix_avg21_45", "18_source_pick_s450"]:
        if name in base_by_name:
            starts.append((f"fixed_{name}", np.full(len(names), base_by_name[name], dtype=np.int32), "fixed"))

    produced: set[tuple[int, ...]] = set()
    candidate_idx = 1
    for start_name, initial, profile in starts:
        for mode in ["rank", "balanced", "detail", "safe"]:
            selection, obj = greedy_improve(initial, metrics_cube, metric_index, leaderboard, calibration, mode=mode, passes=3)
            key = tuple(int(x) for x in selection)
            if key in produced:
                continue
            produced.add(key)
            local = aggregate_metrics(metrics_cube, selection, metric_index)
            pred = predicted_detail(local, leaderboard, calibration)
            source_counts: dict[str, int] = {}
            for selected in selection:
                source_counts[pool[int(selected)].name] = source_counts.get(pool[int(selected)].name, 0) + 1
            candidate_name = f"{candidate_idx:02d}_{start_name}_{mode}"
            out_dir = out_root / candidate_name
            write_candidate(
                out_dir,
                names,
                pool,
                selection,
                local,
                {
                    "method": "imagewise_selector",
                    "start": start_name,
                    "profile": profile,
                    "objective_mode": mode,
                    "objective_value": obj,
                    "pool_size": len(pool),
                    "source_counts": source_counts,
                },
            )
            row: dict[str, Any] = {
                "candidate": candidate_name,
                "path": str(out_dir),
                "start": start_name,
                "objective_mode": mode,
                "Objective": obj,
                "SourceCount": len(source_counts),
                "TopSources": json.dumps(sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:8], ensure_ascii=False),
            }
            row.update(local)
            row.update(pred)
            rows.append(row)
            print(
                f"{candidate_name} pred={row['PredInsertedRank']:.3f} "
                f"AG={row['Pred_AG']:.2f} Q={row['Pred_Qabf']:.3f} N={row['Pred_Nabf']:.3f} "
                f"SD={row['Pred_SD']:.1f} SF={row['Pred_SF']:.2f} SS={row['Pred_SSIM']:.3f} "
                f"sources={len(source_counts)}"
            )
            candidate_idx += 1

    rows.sort(key=lambda row: (float(row["PredInsertedRank"]), float(row["Objective"])))
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    fieldnames = [
        "candidate",
        "path",
        "start",
        "objective_mode",
        "Objective",
        "SourceCount",
        "TopSources",
    ] + METRICS + [f"Pred_{metric}" for metric in METRICS] + ["PredInsertedRank"]
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("best")
    for row in rows[:8]:
        print(
            f"{row['PredInsertedRank']:.3f} {row['candidate']} "
            f"AG={row['Pred_AG']:.2f} Q={row['Pred_Qabf']:.3f} N={row['Pred_Nabf']:.3f} "
            f"SD={row['Pred_SD']:.1f} SF={row['Pred_SF']:.2f} SS={row['Pred_SSIM']:.3f}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
