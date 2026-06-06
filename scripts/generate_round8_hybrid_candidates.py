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
    gradient_mag,
    gray_float,
    list_pairs,
    load_image,
    recombine_y_with_vi_color,
    robust_match,
    save_image_like,
)
from generate_round7_leader_target import (
    METRICS,
    calibration_from_official,
    load_leaderboard,
    metric_rank,
    predict_official,
)
from local_metrics import HIGHER_IS_BETTER, evaluate_dir


SOURCE_DEFAULTS = {
    "r4_02": "results/candidates_round4/02_micro_edge_balanced43",
    "r5_08": "results/candidates_round5/08_transfer18_mid",
    "r7_13": "results/candidates_round7/13_project_leader_struct",
    "r7_10": "results/candidates_round7/10_project_leader_mix1",
    "base33": "results/candidates_round2/33_mix_avg21_45",
    "avg12": "results/candidates_round2/12_avg_raw",
}


def soft_std_lift(x: np.ndarray, reference: np.ndarray, target_std: float, amount: float) -> np.ndarray:
    current = float(np.std(x))
    if current < 1e-6:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    target = (x - float(np.mean(x))) / current * target_std + float(np.mean(reference))
    return np.clip((1.0 - amount) * x + amount * target, 0.0, 1.0).astype(np.float32)


def cap_gradient(fused: np.ndarray, low: np.ndarray, ir: np.ndarray, vi: np.ndarray, cap: float, amount: float) -> np.ndarray:
    gf = gradient_mag(fused)
    source_max = np.maximum(gradient_mag(ir), gradient_mag(vi))
    excess = np.maximum(gf - cap * source_max, 0.0)
    hi = float(np.max(excess))
    mask = excess / hi if hi > 1e-8 else np.zeros_like(excess)
    smooth = ndimage.gaussian_filter(fused, sigma=0.75, mode="reflect")
    return np.clip((1.0 - amount * mask) * fused + amount * mask * (0.70 * smooth + 0.30 * low), 0.0, 1.0)


def gray_source(name: str, key: str, dirs: dict[str, Path]) -> np.ndarray:
    return gray_float(load_image(dirs[key] / name))


def weighted_sum(values: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(float(v) for v in weights.values())
    out = None
    for key, weight in weights.items():
        value = values[key] * (float(weight) / total)
        out = value if out is None else out + value
    return out.astype(np.float32)


def candidate_params() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, **params: Any) -> None:
        base = {
            "weights": {"r4_02": 0.75, "r7_13": 0.25},
            "detail_source": "r7_13",
            "detail_gain": 0.0,
            "detail_sigma": 1.3,
            "target_std": 42.0 / 255.0,
            "std_amount": 0.30,
            "grad_cap": 1.055,
            "cap_amount": 0.60,
            "saturation": 0.98,
        }
        base.update(params)
        out.append((name, base))

    for pct, std, cap in [(15, 41.5, 1.045), (25, 42.0, 1.055), (35, 42.8, 1.065), (45, 43.5, 1.075)]:
        add(
            f"r4_r7struct_w{pct}",
            weights={"r4_02": (100 - pct) / 100, "r7_13": pct / 100},
            target_std=std / 255.0,
            std_amount=0.32 + pct / 200.0,
            grad_cap=cap,
            cap_amount=0.62 - pct / 200.0,
        )
    for pct, std in [(18, 41.8), (28, 42.5), (38, 43.2)]:
        add(
            f"r4_r7struct_detail_w{pct}",
            weights={"r4_02": (100 - pct) / 100, "r7_13": pct / 100},
            detail_source="r7_13",
            detail_gain=0.06 + pct / 500.0,
            target_std=std / 255.0,
            std_amount=0.36 + pct / 220.0,
            grad_cap=1.06 + pct / 1000.0,
            cap_amount=0.56,
        )
    add(
        "r4_r7mix1_lowrisk",
        weights={"r4_02": 0.58, "r5_08": 0.18, "r7_13": 0.16, "avg12": 0.08},
        detail_source="r7_13",
        detail_gain=0.055,
        target_std=42.5 / 255.0,
        std_amount=0.42,
        grad_cap=1.06,
        cap_amount=0.58,
    )
    add(
        "r4_r7mix2_target",
        weights={"r4_02": 0.48, "r5_08": 0.18, "r7_13": 0.24, "avg12": 0.10},
        detail_source="r7_13",
        detail_gain=0.085,
        target_std=43.5 / 255.0,
        std_amount=0.52,
        grad_cap=1.08,
        cap_amount=0.52,
    )
    add(
        "r4_r7mix3_ag",
        weights={"r4_02": 0.42, "r7_13": 0.34, "r7_10": 0.10, "avg12": 0.14},
        detail_source="r7_10",
        detail_gain=0.08,
        target_std=44.0 / 255.0,
        std_amount=0.56,
        grad_cap=1.09,
        cap_amount=0.50,
    )
    return out


def fuse_pair(name: str, dirs: dict[str, Path], params: dict[str, Any]) -> object:
    vi_img = load_image(dirs["vi"] / name)
    vi = gray_float(vi_img)
    ir = robust_match(gray_float(load_image(dirs["ir"] / name)), vi)
    values = {key: gray_source(name, key, dirs) for key in SOURCE_DEFAULTS}
    fused = weighted_sum(values, params["weights"])
    low = weighted_sum(values, {"r4_02": 0.64, "base33": 0.24, "avg12": 0.12})
    if float(params["detail_gain"]):
        source = values[str(params["detail_source"])]
        detail = source - ndimage.gaussian_filter(source, sigma=float(params["detail_sigma"]), mode="reflect")
        fused += float(params["detail_gain"]) * detail
    fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    fused = cap_gradient(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    return recombine_y_with_vi_color(vi_img, np.clip(fused, 0.0, 1.0), saturation=float(params["saturation"]))


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
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in METRICS]))
        row["PredInsertedRank"] = float(np.mean(ranks))
        row["Round8Score"] = float(row["PredInsertedRank"]) + 0.4 * (
            abs(pred["AG"] - 6.16) / 1.2
            + abs(pred["SF"] - 7.43) / 1.0
            + abs(pred["SD"] - 46.0) / 5.0
            + max(0.0, 0.52 - pred["Qabf"]) * 6.0
            + max(0.0, pred["Nabf"] - 0.055) * 8.0
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round8")
    parser.add_argument("--score-out", default="results/round8_local_scores.csv")
    parser.add_argument("--leaderboard", default="leaderboard_20260605_234547.csv")
    parser.add_argument("--official-csv", default="results/score_tables/current_official_scores.csv")
    args = parser.parse_args()

    dirs = {"ir": Path(args.ir_dir), "vi": Path(args.vi_dir)}
    dirs.update({key: Path(value) for key, value in SOURCE_DEFAULTS.items()})
    for key, path in dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} missing: {path}")

    out_root = ensure_dir(args.out_root)
    names = list_pairs(dirs["ir"], dirs["vi"])
    rows: list[dict[str, float | str]] = []
    for idx, (name, params) in enumerate(candidate_params(), start=1):
        cand_dir = ensure_dir(out_root / f"{idx:02d}_{name}")
        print(cand_dir)
        for image_name in names:
            save_image_like(fuse_pair(image_name, dirs, params), cand_dir / image_name)
        (cand_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        _, summary = evaluate_dir(dirs["ir"], dirs["vi"], cand_dir)
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        corr_vals = []
        for image_name in names:
            corr_vals.append(corrcoef(gray_source(image_name, "base33", dirs), gray_float(load_image(cand_dir / image_name))))
        row: dict[str, float | str] = {
            "candidate": cand_dir.name,
            "path": str(cand_dir),
            "family": "hybrid",
            "Base33Corr": float(np.mean(corr_vals)),
        }
        row.update(summary)
        rows.append(row)

    leaderboard = load_leaderboard(Path(args.leaderboard))
    calibration = calibration_from_official(Path(args.official_csv))
    add_scores(rows, leaderboard, calibration)
    rows.sort(key=lambda row: (float(row["Round8Score"]), float(row["PredInsertedRank"])))

    fieldnames = [
        "candidate",
        "path",
        "family",
        "Round8Score",
        "PredInsertedRank",
        "LocalPseudoRank",
        "Base33Corr",
    ] + METRICS + [f"Pred_{metric}" for metric in METRICS] + [f"R_local_{metric}" for metric in METRICS]
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{float(row['Round8Score']):.3f} pred={float(row['PredInsertedRank']):.3f} "
            f"AG={float(row['Pred_AG']):.2f} Q={float(row['Pred_Qabf']):.3f} "
            f"SD={float(row['Pred_SD']):.1f} SF={float(row['Pred_SF']):.2f} {row['candidate']}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
