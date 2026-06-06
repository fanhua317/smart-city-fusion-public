from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from fusion_common import (
    corrcoef,
    ensure_dir,
    gray_float,
    list_pairs,
    load_image,
    recombine_y_with_vi_color,
    save_image_like,
)
from generate_round7_leader_target import (
    METRICS,
    calibration_from_official,
    load_leaderboard,
    metric_rank,
    predict_official,
)
from local_metrics import HIGHER_IS_BETTER, metrics_for_pair


SOURCE_DEFAULTS = {
    "base33": "results/candidates_round2/33_mix_avg21_45",
    "avg12": "results/candidates_round2/12_avg_raw",
    "vi11": "results/candidates_round2/11_vi_only",
    "vi19": "results/candidates_round2/19_vi_anchor_safe_qabf",
    "r4_02": "results/candidates_round4/02_micro_edge_balanced43",
    "r4_04": "results/candidates_round4/04_micro_edge_broad44",
    "r5_08": "results/candidates_round5/08_transfer18_mid",
}


def weighted_sum(values: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(float(v) for v in weights.values())
    if total <= 1e-8:
        raise ValueError("weights must sum to a positive number")
    out = None
    for key, weight in weights.items():
        value = values[key] * (float(weight) / total)
        out = value if out is None else out + value
    return out.astype(np.float32)


def candidate_params() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, family: str, **params: Any) -> None:
        base = {
            "family": family,
            "saturation": 0.98,
            "sigma": 5.0,
            "low_amount": 0.30,
            "weights": {"vi11": 1.0},
            "gamma": 1.0,
            "mean_source": "vi",
        }
        base.update(params)
        out.append((name, base))

    for w in [5, 10, 16, 24, 35, 50, 70]:
        add(
            f"raw_r4_vi11_w{w}",
            "linear",
            weights={"r4_02": w / 100.0, "vi11": (100 - w) / 100.0},
        )
    for w in [8, 16, 28, 40]:
        add(
            f"raw_r404_vi11_w{w}",
            "linear",
            weights={"r4_04": w / 100.0, "vi11": (100 - w) / 100.0},
        )
    for amount in [15, 25, 35, 45, 60, 80]:
        add(
            f"vi11_low_r4_a{amount}",
            "low_replace",
            base_key="vi11",
            low_weights={"r4_02": 0.72, "base33": 0.18, "avg12": 0.10},
            low_amount=amount / 100.0,
            sigma=5.0,
        )
    for amount in [18, 30, 42, 55]:
        add(
            f"vi11_low_mix_a{amount}",
            "low_replace",
            base_key="vi11",
            low_weights={"r4_02": 0.52, "base33": 0.18, "avg12": 0.10, "vi19": 0.20},
            low_amount=amount / 100.0,
            sigma=6.5,
        )
    for amount in [20, 35, 50]:
        add(
            f"vi11_low_r5_a{amount}",
            "low_replace",
            base_key="vi11",
            low_weights={"r4_02": 0.46, "r5_08": 0.26, "base33": 0.16, "avg12": 0.12},
            low_amount=amount / 100.0,
            sigma=5.5,
        )
    return out


def fuse_luma(vi: np.ndarray, values: dict[str, np.ndarray], params: dict[str, Any]) -> np.ndarray:
    vals = dict(values)
    vals["vi"] = vi
    family = str(params["family"])
    if family == "linear":
        fused = weighted_sum(vals, params["weights"])
    elif family == "low_replace":
        base = vals[str(params["base_key"])]
        sigma = float(params["sigma"])
        source_low = ndimage.gaussian_filter(base, sigma=sigma, mode="reflect")
        target_low = ndimage.gaussian_filter(weighted_sum(vals, params["low_weights"]), sigma=sigma, mode="reflect")
        fused = base + float(params["low_amount"]) * (target_low - source_low)
    else:
        raise ValueError(f"unknown family: {family}")
    if float(params["gamma"]) != 1.0:
        fused = np.power(np.clip(fused, 0.0, 1.0), float(params["gamma"]))
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def add_scores(rows: list[dict[str, float | str]], leaderboard: list[dict[str, str]], calibration: dict[str, tuple[float, float, float]]) -> None:
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_local_{metric}"] = rank
    for row in rows:
        local = {metric: float(row[metric]) for metric in METRICS}
        pred = predict_official(local, calibration)
        ranks = [metric_rank(leaderboard, metric, pred[metric]) for metric in METRICS]
        for metric in METRICS:
            row[f"Pred_{metric}"] = pred[metric]
            row[f"Pred_R_{metric}"] = metric_rank(leaderboard, metric, pred[metric])
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in METRICS]))
        row["PredInsertedRank"] = float(np.mean(ranks))
        row["Round9bScore"] = float(row["PredInsertedRank"]) + 0.35 * (
            max(0.0, 0.58 - pred["Qabf"]) * 10.0
            + max(0.0, 0.735 - pred["SSIM"]) * 6.0
            + max(0.0, pred["Nabf"] - 0.040) * 12.0
            + abs(pred["AG"] - 6.16) / 1.6
            + abs(pred["SF"] - 7.43) / 1.4
            + abs(pred["SD"] - 46.0) / 6.0
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round9b")
    parser.add_argument("--score-out", default="results/round9b_local_scores.csv")
    parser.add_argument("--leaderboard", default="leaderboard_20260605_234547.csv")
    parser.add_argument("--official-csv", default="results/score_tables/current_official_scores.csv")
    args = parser.parse_args()

    dirs = {"ir": Path(args.ir_dir), "vi": Path(args.vi_dir)}
    dirs.update({key: Path(value) for key, value in SOURCE_DEFAULTS.items()})
    for key, path in dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} missing: {path}")

    candidates = candidate_params()
    out_root = ensure_dir(args.out_root)
    cand_dirs = []
    for idx, (name, params) in enumerate(candidates, start=1):
        cand_dir = ensure_dir(out_root / f"{idx:02d}_{name}")
        cand_dirs.append(cand_dir)
        (cand_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    names = list_pairs(dirs["ir"], dirs["vi"])
    metric_rows: list[list[dict[str, float]]] = [[] for _ in candidates]
    base_corrs: list[list[float]] = [[] for _ in candidates]
    for image_idx, image_name in enumerate(names, start=1):
        ir_img = load_image(dirs["ir"] / image_name)
        vi_img = load_image(dirs["vi"] / image_name)
        ir = gray_float(ir_img)
        vi = gray_float(vi_img)
        values = {key: gray_float(load_image(path / image_name)) for key, path in dirs.items() if key not in {"ir", "vi"}}
        for idx, ((_, params), cand_dir) in enumerate(zip(candidates, cand_dirs)):
            fused_y = fuse_luma(vi, values, params)
            save_image_like(
                recombine_y_with_vi_color(vi_img, fused_y, saturation=float(params["saturation"])),
                cand_dir / image_name,
            )
            metric_rows[idx].append(metrics_for_pair(ir, vi, fused_y))
            base_corrs[idx].append(corrcoef(values["base33"], fused_y))
        if image_idx % 10 == 0 or image_idx == len(names):
            print(f"processed {image_idx}/{len(names)}")

    rows: list[dict[str, float | str]] = []
    for (name, params), cand_dir, per_image, corrs in zip(candidates, cand_dirs, metric_rows, base_corrs):
        summary = {metric: float(np.mean([row[metric] for row in per_image])) for metric in HIGHER_IS_BETTER}
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row: dict[str, float | str] = {
            "candidate": cand_dir.name,
            "path": str(cand_dir),
            "family": str(params["family"]),
            "Base33Corr": float(np.mean(corrs)),
        }
        row.update(summary)
        rows.append(row)

    leaderboard = load_leaderboard(Path(args.leaderboard))
    calibration = calibration_from_official(Path(args.official_csv))
    add_scores(rows, leaderboard, calibration)
    rows.sort(key=lambda row: (float(row["Round9bScore"]), float(row["PredInsertedRank"])))

    fieldnames = [
        "candidate",
        "path",
        "family",
        "Round9bScore",
        "PredInsertedRank",
        "LocalPseudoRank",
        "Base33Corr",
    ] + METRICS + [f"Pred_{metric}" for metric in METRICS] + [f"Pred_R_{metric}" for metric in METRICS] + [
        f"R_local_{metric}" for metric in METRICS
    ]
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{float(row['Round9bScore']):.3f} pred={float(row['PredInsertedRank']):.3f} "
            f"q={float(row['Qabf']):.4f}/{float(row['Pred_Qabf']):.3f} "
            f"ag={float(row['Pred_AG']):.2f} sf={float(row['Pred_SF']):.2f} "
            f"sd={float(row['Pred_SD']):.1f} nabf={float(row['Pred_Nabf']):.3f} "
            f"ssim={float(row['Pred_SSIM']):.3f} {row['candidate']}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
