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
    normalize01,
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
from local_metrics import HIGHER_IS_BETTER, metrics_for_pair


SOURCE_DEFAULTS = {
    "base33": "results/candidates_round2/33_mix_avg21_45",
    "avg12": "results/candidates_round2/12_avg_raw",
    "vi11": "results/candidates_round2/11_vi_only",
    "vi19": "results/candidates_round2/19_vi_anchor_safe_qabf",
    "detail18": "results/candidates/18_low_artifact_color",
    "cross21": "results/candidates/21_crossblend_35_color",
    "r4_02": "results/candidates_round4/02_micro_edge_balanced43",
    "r4_04": "results/candidates_round4/04_micro_edge_broad44",
    "r5_08": "results/candidates_round5/08_transfer18_mid",
    "r5_15": "results/candidates_round5/15_vi_repair_with18",
    "r7_13": "results/candidates_round7/13_project_leader_struct",
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


def soft_std_lift(x: np.ndarray, reference: np.ndarray, target_std: float, amount: float) -> np.ndarray:
    current = float(np.std(x))
    if current < 1e-8 or amount <= 0.0:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    target = (x - float(np.mean(x))) / current * target_std + float(np.mean(reference))
    return np.clip((1.0 - amount) * x + amount * target, 0.0, 1.0).astype(np.float32)


def source_gate(ir_matched: np.ndarray, vi: np.ndarray, edge_floor: float, thermal_weight: float, sigma: float) -> np.ndarray:
    g_ir = normalize01(gradient_mag(ir_matched))
    g_vi = normalize01(gradient_mag(vi))
    thermal = normalize01(np.abs(ir_matched - ndimage.gaussian_filter(ir_matched, sigma=9.0, mode="reflect")))
    gate = np.maximum(g_ir - edge_floor * g_vi, 0.0) + thermal_weight * thermal
    gate = normalize01(gate)
    return ndimage.gaussian_filter(gate, sigma=sigma, mode="reflect").astype(np.float32)


def cap_gradient(fused: np.ndarray, low: np.ndarray, ir: np.ndarray, vi: np.ndarray, cap: float, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return np.clip(fused, 0.0, 1.0).astype(np.float32)
    gf = gradient_mag(fused)
    source_max = np.maximum(gradient_mag(ir), gradient_mag(vi))
    excess = np.maximum(gf - cap * source_max, 0.0)
    hi = float(np.max(excess))
    mask = excess / hi if hi > 1e-8 else np.zeros_like(excess)
    smooth = ndimage.gaussian_filter(fused, sigma=0.72, mode="reflect")
    pulled = (1.0 - amount * mask) * fused + amount * mask * (0.72 * smooth + 0.28 * low)
    return np.clip(pulled, 0.0, 1.0).astype(np.float32)


def high_detail(x: np.ndarray, sigma: float) -> np.ndarray:
    return (x - ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")).astype(np.float32)


def candidate_params() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, family: str, **updates: Any) -> None:
        params: dict[str, Any] = {
            "family": family,
            "target_std": 43.0 / 255.0,
            "std_amount": 0.20,
            "grad_cap": 1.055,
            "cap_amount": 0.46,
            "saturation": 0.98,
            "edge_floor": 0.88,
            "thermal_weight": 0.10,
            "gate_sigma": 1.05,
            "detail_sigma": 1.25,
            "ir_gate_gain": 0.0,
            "vi_detail_gain": 0.0,
            "source_detail_gain": 0.0,
            "source_detail_key": "detail18",
            "gamma": 1.0,
        }
        params.update(updates)
        out.append((name, params))

    for pct, std, cap_amount in [
        (10, 42.0, 0.56),
        (16, 42.6, 0.52),
        (22, 43.2, 0.48),
        (30, 44.0, 0.44),
        (38, 44.8, 0.40),
    ]:
        add(
            f"r4_vi11_blend_w{pct}",
            "direct_blend",
            weights={"r4_02": (100 - pct) / 100, "vi11": pct / 100},
            low_weights={"r4_02": 0.74, "base33": 0.18, "avg12": 0.08},
            target_std=std / 255.0,
            std_amount=0.12 + pct / 250.0,
            grad_cap=1.045 + pct / 1000.0,
            cap_amount=cap_amount,
        )

    for pct, vi_gain, ir_gain, std in [
        (14, 0.08, 0.030, 42.8),
        (20, 0.10, 0.040, 43.4),
        (26, 0.12, 0.055, 44.0),
        (32, 0.15, 0.070, 44.7),
    ]:
        add(
            f"qprotect_edge_w{pct}",
            "direct_blend",
            weights={"r4_02": (100 - pct) / 100, "vi11": pct / 100},
            low_weights={"r4_02": 0.66, "base33": 0.22, "avg12": 0.12},
            vi_detail_gain=vi_gain,
            ir_gate_gain=ir_gain,
            target_std=std / 255.0,
            std_amount=0.22 + pct / 260.0,
            grad_cap=1.055 + pct / 1200.0,
            cap_amount=0.48,
            edge_floor=0.82,
            thermal_weight=0.12,
        )

    for pct, src_gain, std in [
        (12, 0.025, 43.0),
        (18, 0.035, 43.6),
        (24, 0.045, 44.2),
    ]:
        add(
            f"qprotect_transfer18_w{pct}",
            "direct_blend",
            weights={"r4_02": (100 - pct) / 100, "vi11": pct / 100},
            low_weights={"r4_02": 0.70, "base33": 0.20, "avg12": 0.10},
            source_detail_key="detail18",
            source_detail_gain=src_gain,
            vi_detail_gain=0.08,
            target_std=std / 255.0,
            std_amount=0.24 + pct / 300.0,
            grad_cap=1.055 + pct / 1300.0,
            cap_amount=0.50,
        )

    add(
        "decomp_vi_low_struct",
        "decomposition",
        low_weights={"r4_02": 0.52, "base33": 0.20, "avg12": 0.12, "vi11": 0.16},
        high_weights={"vi11": 0.52, "vi": 0.30, "r4_02": 0.12, "ir": 0.06},
        high_gain=0.86,
        low_sigma=3.2,
        target_std=44.0 / 255.0,
        std_amount=0.36,
        grad_cap=1.07,
        cap_amount=0.46,
        ir_gate_gain=0.035,
    )
    add(
        "decomp_vi_qabf_high",
        "decomposition",
        low_weights={"r4_02": 0.42, "base33": 0.18, "avg12": 0.10, "vi11": 0.30},
        high_weights={"vi11": 0.56, "vi": 0.34, "r4_04": 0.06, "ir": 0.04},
        high_gain=0.94,
        low_sigma=3.8,
        target_std=45.0 / 255.0,
        std_amount=0.40,
        grad_cap=1.085,
        cap_amount=0.40,
        ir_gate_gain=0.045,
    )
    add(
        "decomp_r5_repair",
        "decomposition",
        low_weights={"r4_02": 0.46, "r5_08": 0.18, "base33": 0.18, "vi11": 0.18},
        high_weights={"vi11": 0.44, "vi": 0.28, "r5_15": 0.18, "ir": 0.10},
        high_gain=0.82,
        low_sigma=3.5,
        target_std=44.2 / 255.0,
        std_amount=0.34,
        grad_cap=1.075,
        cap_amount=0.45,
        ir_gate_gain=0.050,
    )

    for std, pick_gain, blend_gain in [(43.4, 0.12, 0.40), (44.2, 0.16, 0.48), (45.0, 0.20, 0.56)]:
        add(
            f"source_pick_s{int(std * 10):03d}",
            "source_pick",
            low_weights={"r4_02": 0.58, "base33": 0.18, "avg12": 0.10, "vi11": 0.14},
            detail_keys=["vi11", "vi", "r4_04", "ir"],
            detail_weights={"vi11": 0.42, "vi": 0.28, "r4_04": 0.18, "ir": 0.12},
            pick_gain=pick_gain,
            blend_gain=blend_gain,
            target_std=std / 255.0,
            std_amount=0.36,
            grad_cap=1.07,
            cap_amount=0.48,
            edge_floor=0.84,
            thermal_weight=0.12,
        )
    return out


def fuse_luma(ir_raw: np.ndarray, vi: np.ndarray, values: dict[str, np.ndarray], params: dict[str, Any]) -> np.ndarray:
    ir_matched = robust_match(ir_raw, vi)
    vals = dict(values)
    vals["ir"] = ir_matched
    vals["vi"] = vi
    gate = source_gate(
        ir_matched,
        vi,
        float(params["edge_floor"]),
        float(params["thermal_weight"]),
        float(params["gate_sigma"]),
    )

    family = str(params["family"])
    if family == "direct_blend":
        fused = weighted_sum(vals, params["weights"])
    elif family == "decomposition":
        low_sigma = float(params["low_sigma"])
        low_bank = {key: ndimage.gaussian_filter(vals[key], sigma=low_sigma, mode="reflect") for key in params["low_weights"]}
        high_bank = {key: high_detail(vals[key], float(params["detail_sigma"])) for key in params["high_weights"]}
        fused = weighted_sum(low_bank, params["low_weights"]) + float(params["high_gain"]) * weighted_sum(
            high_bank, params["high_weights"]
        )
    elif family == "source_pick":
        low = weighted_sum(vals, params["low_weights"])
        detail_bank = {key: high_detail(vals[key], float(params["detail_sigma"])) for key in params["detail_keys"]}
        grad_bank = {key: normalize01(gradient_mag(vals[key])) for key in params["detail_keys"]}
        stack = np.stack([grad_bank[key] for key in params["detail_keys"]], axis=0)
        idx = np.argmax(stack, axis=0)
        details = np.stack([detail_bank[key] for key in params["detail_keys"]], axis=0)
        picked = np.take_along_axis(details, idx[None, ...], axis=0)[0]
        fused = low + float(params["pick_gain"]) * gate * picked + float(params["blend_gain"]) * weighted_sum(
            detail_bank, params["detail_weights"]
        )
    else:
        raise ValueError(f"unknown family: {family}")

    if float(params["vi_detail_gain"]):
        fused += float(params["vi_detail_gain"]) * high_detail(vi, float(params["detail_sigma"]))
    if float(params["ir_gate_gain"]):
        fused += float(params["ir_gate_gain"]) * gate * high_detail(ir_matched, float(params["detail_sigma"]))
    if float(params["source_detail_gain"]):
        key = str(params["source_detail_key"])
        fused += float(params["source_detail_gain"]) * high_detail(vals[key], float(params["detail_sigma"]))
    if float(params["gamma"]) != 1.0:
        fused = np.power(np.clip(fused, 0.0, 1.0), float(params["gamma"]))

    low_ref = weighted_sum(vals, params.get("low_weights", {"r4_02": 1.0}))
    fused = soft_std_lift(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    fused = cap_gradient(fused, low_ref, ir_raw, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def add_round9_scores(rows: list[dict[str, float | str]], leaderboard: list[dict[str, str]], calibration: dict[str, tuple[float, float, float]]) -> None:
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
        row["Round9Score"] = float(row["PredInsertedRank"]) + 0.45 * (
            max(0.0, 0.60 - pred["Qabf"]) * 8.0
            + max(0.0, pred["Nabf"] - 0.055) * 10.0
            + max(0.0, 0.735 - pred["SSIM"]) * 6.0
            + abs(pred["AG"] - 6.16) / 1.4
            + abs(pred["SF"] - 7.43) / 1.2
            + abs(pred["SD"] - 46.0) / 5.0
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--out-root", default="results/candidates_round9")
    parser.add_argument("--score-out", default="results/round9_local_scores.csv")
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
        ir_raw = gray_float(ir_img)
        vi = gray_float(vi_img)
        values = {key: gray_float(load_image(path / image_name)) for key, path in dirs.items() if key not in {"ir", "vi"}}
        for idx, ((_, params), cand_dir) in enumerate(zip(candidates, cand_dirs)):
            fused_y = fuse_luma(ir_raw, vi, values, params)
            save_image_like(
                recombine_y_with_vi_color(vi_img, fused_y, saturation=float(params["saturation"])),
                cand_dir / image_name,
            )
            metric_rows[idx].append(metrics_for_pair(ir_raw, vi, fused_y))
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
    add_round9_scores(rows, leaderboard, calibration)
    rows.sort(key=lambda row: (float(row["Round9Score"]), float(row["PredInsertedRank"])))

    fieldnames = [
        "candidate",
        "path",
        "family",
        "Round9Score",
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
            f"{float(row['Round9Score']):.3f} pred={float(row['PredInsertedRank']):.3f} "
            f"q={float(row['Qabf']):.4f}/{float(row['Pred_Qabf']):.3f} "
            f"ag={float(row['Pred_AG']):.2f} sf={float(row['Pred_SF']):.2f} "
            f"sd={float(row['Pred_SD']):.1f} nabf={float(row['Pred_Nabf']):.3f} "
            f"ssim={float(row['Pred_SSIM']):.3f} {row['candidate']}"
        )
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
