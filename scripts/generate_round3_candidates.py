from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from fusion_common import (
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
from local_metrics import HIGHER_IS_BETTER, evaluate_dir


def match_std_to_reference(x: np.ndarray, reference: np.ndarray, target_std: float, amount: float) -> np.ndarray:
    adjusted = (x - float(x.mean())) / (float(x.std()) + 1e-6) * target_std + float(reference.mean())
    return np.clip((1.0 - amount) * x + amount * adjusted, 0.0, 1.0).astype(np.float32)


def source_gate(ir: np.ndarray, vi: np.ndarray, edge_floor: float, thermal_weight: float) -> np.ndarray:
    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    thermal = normalize01(np.abs(ir - ndimage.gaussian_filter(ir, sigma=10.0, mode="reflect")))
    edge = normalize01(np.maximum(g_ir - edge_floor * g_vi, 0.0))
    return ndimage.gaussian_filter(np.clip(edge + thermal_weight * thermal, 0.0, 1.0), sigma=1.0, mode="reflect")


def fuse_pair(name: str, dirs: dict[str, Path], params: dict[str, float | str]) -> object:
    ir_img = load_image(dirs["ir"] / name)
    vi_img = load_image(dirs["vi"] / name)
    ir_raw = gray_float(ir_img)
    vi = gray_float(vi_img)
    ir_matched = robust_match(ir_raw, vi)
    base33 = gray_float(load_image(dirs["base33"] / name))
    avg_raw = gray_float(load_image(dirs["avg_raw"] / name))
    cross21 = gray_float(load_image(dirs["cross21"] / name))
    deep = gray_float(load_image(dirs["deep"] / name)) if dirs.get("deep") and (dirs["deep"] / name).exists() else cross21

    family = str(params["family"])
    if family == "mix33_luma_gain":
        fused = match_std_to_reference(base33, vi, float(params["target_std"]), float(params["amount"]))
    elif family == "source_consistent_edge":
        low = (1.0 - float(params["avg_low_weight"])) * base33 + float(params["avg_low_weight"]) * avg_raw
        detail_sigma = float(params["detail_sigma"])
        ir_detail = ir_matched - ndimage.gaussian_filter(ir_matched, sigma=detail_sigma, mode="reflect")
        vi_detail = vi - ndimage.gaussian_filter(vi, sigma=detail_sigma, mode="reflect")
        gate = source_gate(ir_matched, vi, float(params["edge_floor"]), float(params["thermal_weight"]))
        source_detail = float(params["vi_detail_weight"]) * vi_detail + float(params["ir_detail_weight"]) * gate * ir_detail
        fused = low + source_detail
        fused = match_std_to_reference(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    elif family == "decomposition_gate":
        low_sigma = float(params["low_sigma"])
        low_base = (1.0 - float(params["avg_low_weight"])) * base33 + float(params["avg_low_weight"]) * avg_raw
        low = ndimage.gaussian_filter(low_base, sigma=low_sigma, mode="reflect")
        vi_detail = vi - ndimage.gaussian_filter(vi, sigma=float(params["vi_sigma"]), mode="reflect")
        ir_detail = ir_matched - ndimage.gaussian_filter(ir_matched, sigma=float(params["ir_sigma"]), mode="reflect")
        gate = source_gate(ir_matched, vi, float(params["edge_floor"]), float(params["thermal_weight"]))
        fused = low + float(params["vi_detail_weight"]) * vi_detail + float(params["ir_detail_weight"]) * gate * ir_detail
        fused = match_std_to_reference(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    elif family == "deep_aux":
        fused = (
            (1.0 - float(params["avg_weight"]) - float(params["deep_weight"])) * base33
            + float(params["avg_weight"]) * avg_raw
            + float(params["deep_weight"]) * robust_match(deep, vi)
        )
        fused = match_std_to_reference(fused, vi, float(params["target_std"]), float(params["std_amount"]))
    else:
        raise ValueError(f"Unknown family: {family}")

    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidate_params() -> list[tuple[str, dict[str, float | str]]]:
    out: list[tuple[str, dict[str, float | str]]] = []

    def add(name: str, family: str, **updates: float | str) -> None:
        params: dict[str, float | str] = {"family": family, "saturation": 0.98}
        params.update(updates)
        out.append((name, params))

    for target in (44, 46, 48, 50):
        add(f"mix33_luma_gain_sd{target}", "mix33_luma_gain", target_std=target / 255.0, amount=1.0)
    add("mix33_luma_gain_soft46", "mix33_luma_gain", target_std=46 / 255.0, amount=0.65)
    add("mix33_luma_gain_soft48", "mix33_luma_gain", target_std=48 / 255.0, amount=0.65)

    edge_base = {
        "avg_low_weight": 0.15,
        "detail_sigma": 1.3,
        "edge_floor": 0.82,
        "thermal_weight": 0.20,
        "vi_detail_weight": 0.30,
        "ir_detail_weight": 0.06,
        "target_std": 45 / 255.0,
        "std_amount": 0.55,
    }
    add("source_edge_vi030_ir006", "source_consistent_edge", **edge_base)
    p = dict(edge_base)
    p.update({"vi_detail_weight": 0.42, "ir_detail_weight": 0.08, "target_std": 46 / 255.0})
    add("source_edge_vi042_ir008", "source_consistent_edge", **p)
    p = dict(edge_base)
    p.update({"edge_floor": 0.92, "thermal_weight": 0.12, "vi_detail_weight": 0.36, "ir_detail_weight": 0.04})
    add("source_edge_safe_qabf", "source_consistent_edge", **p)
    p = dict(edge_base)
    p.update({"avg_low_weight": 0.30, "vi_detail_weight": 0.34, "ir_detail_weight": 0.10, "target_std": 47 / 255.0})
    add("source_edge_avg_low30", "source_consistent_edge", **p)
    p = dict(edge_base)
    p.update({"detail_sigma": 2.0, "vi_detail_weight": 0.48, "ir_detail_weight": 0.08, "target_std": 48 / 255.0})
    add("source_edge_broad_detail", "source_consistent_edge", **p)

    decomp_base = {
        "low_sigma": 5.0,
        "vi_sigma": 1.2,
        "ir_sigma": 1.8,
        "avg_low_weight": 0.35,
        "edge_floor": 0.80,
        "thermal_weight": 0.20,
        "vi_detail_weight": 0.85,
        "ir_detail_weight": 0.06,
        "target_std": 45 / 255.0,
        "std_amount": 0.55,
    }
    add("decomp_avg35_vi085_ir006", "decomposition_gate", **decomp_base)
    p = dict(decomp_base)
    p.update({"avg_low_weight": 0.50, "vi_detail_weight": 0.95, "target_std": 46 / 255.0})
    add("decomp_avg50_vi095", "decomposition_gate", **p)
    p = dict(decomp_base)
    p.update({"avg_low_weight": 0.20, "vi_detail_weight": 1.05, "ir_detail_weight": 0.04, "edge_floor": 0.90})
    add("decomp_qabf_visible", "decomposition_gate", **p)
    p = dict(decomp_base)
    p.update({"thermal_weight": 0.32, "ir_detail_weight": 0.12, "target_std": 47 / 255.0})
    add("decomp_thermal_targets", "decomposition_gate", **p)

    add("deep_aux_10", "deep_aux", avg_weight=0.15, deep_weight=0.10, target_std=45 / 255.0, std_amount=0.60)
    add("deep_aux_20", "deep_aux", avg_weight=0.15, deep_weight=0.20, target_std=46 / 255.0, std_amount=0.65)
    add("deep_aux_30", "deep_aux", avg_weight=0.10, deep_weight=0.30, target_std=47 / 255.0, std_amount=0.70)
    return out


def add_local_rank_scores(rows: list[dict[str, float | str]]) -> None:
    for metric, higher in HIGHER_IS_BETTER.items():
        order = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]), reverse=higher)
        for rank, idx in enumerate(order, start=1):
            rows[idx][f"R_local_{metric}"] = rank
    for row in rows:
        row["LocalPseudoRank"] = float(np.mean([float(row[f"R_local_{metric}"]) for metric in HIGHER_IS_BETTER]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--base33-dir", default="results/candidates_round2/33_mix_avg21_45")
    parser.add_argument("--avg-raw-dir", default="results/candidates_round2/12_avg_raw")
    parser.add_argument("--cross21-dir", default="results/candidates/21_crossblend_35_color")
    parser.add_argument("--deep-dir", default="remote_sync/results/deep_crossfuse_raw")
    parser.add_argument("--out-root", default="results/candidates_round3")
    parser.add_argument("--score-out", default="results/round3_local_scores.csv")
    args = parser.parse_args()

    dirs = {
        "ir": Path(args.ir_dir),
        "vi": Path(args.vi_dir),
        "base33": Path(args.base33_dir),
        "avg_raw": Path(args.avg_raw_dir),
        "cross21": Path(args.cross21_dir),
    }
    deep_dir = Path(args.deep_dir)
    if deep_dir.exists():
        dirs["deep"] = deep_dir

    out_root = ensure_dir(args.out_root)
    rows: list[dict[str, float | str]] = []
    names = list_pairs(dirs["ir"], dirs["vi"])
    for idx, (name, params) in enumerate(candidate_params(), start=1):
        cand_dir = out_root / f"{idx:02d}_{name}"
        ensure_dir(cand_dir)
        print(cand_dir)
        for image_name in names:
            save_image_like(fuse_pair(image_name, dirs, params), cand_dir / image_name)
        (cand_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        _, summary = evaluate_dir(dirs["ir"], dirs["vi"], cand_dir)
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row: dict[str, float | str] = {"candidate": cand_dir.name, "path": str(cand_dir), "family": str(params["family"])}
        row.update(summary)
        rows.append(row)

    add_local_rank_scores(rows)
    rows.sort(key=lambda row: float(row["LocalPseudoRank"]))
    fieldnames = ["candidate", "path", "family", "LocalPseudoRank"] + list(HIGHER_IS_BETTER) + [
        f"R_local_{metric}" for metric in HIGHER_IS_BETTER
    ]
    score_out = Path(args.score_out)
    ensure_dir(score_out.parent)
    with score_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['LocalPseudoRank']:.3f} {row['candidate']} {row['family']}")
    print(f"wrote {score_out}")


if __name__ == "__main__":
    main()
