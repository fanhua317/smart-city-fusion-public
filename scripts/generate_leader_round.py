from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
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


def soft_match_std(x: np.ndarray, reference: np.ndarray, target_std: float, mean_mix: float) -> np.ndarray:
    ref_mean = float(reference.mean())
    ref_std = float(reference.std()) + 1e-6
    x_mean = float(x.mean())
    x_std = float(x.std()) + 1e-6
    matched = (x - x_mean) / x_std * target_std + (mean_mix * ref_mean + (1.0 - mean_mix) * x_mean)
    if target_std <= 0:
        matched = x
    return np.clip(matched, 0.0, 1.0).astype(np.float32)


def edge_gate(ir: np.ndarray, vi: np.ndarray, edge_floor: float, thermal_gain: float) -> np.ndarray:
    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    ir_advantage = np.maximum(g_ir - edge_floor * g_vi, 0.0)
    thermal = normalize01(np.abs(ir - ndimage.gaussian_filter(ir, sigma=10.0, mode="reflect")))
    gate = normalize01(0.65 * ir_advantage + thermal_gain * thermal)
    return ndimage.gaussian_filter(np.clip(gate, 0.0, 1.0), sigma=1.0, mode="reflect")


def optional_deep_y(name: str, deep_dir: Path | None, vi_y: np.ndarray) -> np.ndarray | None:
    if deep_dir is None:
        return None
    path = deep_dir / name
    if not path.exists():
        return None
    deep = gray_float(load_image(path))
    return robust_match(deep, vi_y)


def fuse_pair(name: str, ir_img: Image.Image, vi_img: Image.Image, params: dict[str, float | str], deep_dir: Path | None) -> Image.Image:
    ir_raw = gray_float(ir_img)
    vi_y = gray_float(vi_img)
    ir = robust_match(ir_raw, vi_y)

    ir_low = ndimage.gaussian_filter(ir, sigma=float(params["low_sigma"]), mode="reflect")
    vi_low = ndimage.gaussian_filter(vi_y, sigma=float(params["low_sigma"]), mode="reflect")
    ir_detail = ir - ir_low
    vi_detail = vi_y - vi_low

    base_ir = float(params["base_ir"])
    base = base_ir * ir_low + (1.0 - base_ir) * vi_low

    g_ir = gradient_mag(ir)
    g_vi = gradient_mag(vi_y)
    detail_bias = float(params["detail_ir_bias"])
    detail_w = (detail_bias * g_ir + 1e-4) / (detail_bias * g_ir + g_vi + 2e-4)
    detail_w = ndimage.gaussian_filter(np.clip(detail_w, 0.0, 1.0), sigma=0.8, mode="reflect")
    detail = detail_w * ir_detail + (1.0 - detail_w) * vi_detail

    gate = edge_gate(ir, vi_y, float(params["edge_floor"]), float(params["thermal_gate"]))
    fused = base + float(params["detail_strength"]) * (0.35 + 0.65 * gate) * detail

    if float(params["raw_ir_residual"]):
        fused += float(params["raw_ir_residual"]) * (ir_raw - ndimage.gaussian_filter(ir_raw, sigma=6.0, mode="reflect")) * gate

    deep = optional_deep_y(name, deep_dir, vi_y)
    if deep is not None and float(params["deep_weight"]):
        deep_low = ndimage.gaussian_filter(deep, sigma=2.0, mode="reflect")
        fused = (1.0 - float(params["deep_weight"])) * fused + float(params["deep_weight"]) * deep_low

    sharpen = float(params["sharpen"])
    if sharpen:
        fused += sharpen * (fused - ndimage.gaussian_filter(fused, sigma=1.1, mode="reflect")) * (0.5 + 0.5 * gate)

    mean_mix = float(params["mean_mix"])
    target_std = float(params["target_std"])
    fused = soft_match_std(fused, vi_y, target_std, mean_mix)

    gamma = float(params["gamma"])
    if abs(gamma - 1.0) > 1e-4:
        fused = np.power(np.clip(fused, 0.0, 1.0), gamma)
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

    color_mode = str(params["color_mode"])
    if color_mode == "gray" or vi_img.mode == "L":
        return Image.fromarray(np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def candidate_params() -> list[tuple[str, dict[str, float | str]]]:
    base: dict[str, float | str] = {
        "base_ir": 0.46,
        "low_sigma": 3.0,
        "detail_strength": 0.12,
        "detail_ir_bias": 1.0,
        "edge_floor": 0.85,
        "thermal_gate": 0.20,
        "raw_ir_residual": 0.0,
        "deep_weight": 0.0,
        "sharpen": 0.0,
        "target_std": 0.182,
        "mean_mix": 0.85,
        "gamma": 1.0,
        "saturation": 0.90,
        "color_mode": "rgb",
        "family": "leader_struct",
    }
    variants: list[tuple[str, dict[str, float | str]]] = []

    def add(name: str, **updates: float | str) -> None:
        params = dict(base)
        params.update(updates)
        variants.append((name, params))

    add("leader_struct_vi58_ir42", base_ir=0.42, detail_strength=0.08, target_std=0.176)
    add("leader_struct_balanced", base_ir=0.48, detail_strength=0.10, target_std=0.182)
    add("leader_struct_ir52", base_ir=0.52, detail_strength=0.10, target_std=0.184)
    add("leader_struct_softdeep08", base_ir=0.46, detail_strength=0.08, deep_weight=0.08, target_std=0.180)

    add(
        "leader_qabf_gate18",
        family="leader_qabf",
        base_ir=0.46,
        detail_strength=0.18,
        detail_ir_bias=1.18,
        edge_floor=0.75,
        thermal_gate=0.28,
        raw_ir_residual=0.015,
        sharpen=0.015,
        target_std=0.188,
    )
    add(
        "leader_qabf_gate24",
        family="leader_qabf",
        base_ir=0.48,
        detail_strength=0.24,
        detail_ir_bias=1.28,
        edge_floor=0.70,
        thermal_gate=0.30,
        raw_ir_residual=0.020,
        sharpen=0.020,
        target_std=0.192,
    )
    add(
        "leader_qabf_viscolor",
        family="leader_qabf",
        base_ir=0.43,
        detail_strength=0.20,
        detail_ir_bias=1.12,
        edge_floor=0.72,
        thermal_gate=0.24,
        target_std=0.186,
        saturation=1.0,
    )

    add(
        "leader_mi_deep15",
        family="leader_mi",
        base_ir=0.45,
        detail_strength=0.14,
        detail_ir_bias=1.05,
        deep_weight=0.15,
        raw_ir_residual=0.015,
        target_std=0.186,
    )
    add(
        "leader_mi_deep22",
        family="leader_mi",
        base_ir=0.48,
        detail_strength=0.16,
        detail_ir_bias=1.10,
        deep_weight=0.22,
        raw_ir_residual=0.020,
        target_std=0.190,
    )
    add(
        "leader_mi_graydeep18",
        family="leader_mi",
        base_ir=0.48,
        detail_strength=0.16,
        detail_ir_bias=1.12,
        deep_weight=0.18,
        target_std=0.188,
        color_mode="gray",
    )
    return variants


def generate_candidate(ir_dir: Path, vi_dir: Path, out_dir: Path, params: dict[str, float | str], deep_dir: Path | None) -> int:
    ensure_dir(out_dir)
    names = list_pairs(ir_dir, vi_dir)
    for name in names:
        ir_img = load_image(ir_dir / name)
        vi_img = load_image(vi_dir / name)
        fused = fuse_pair(name, ir_img, vi_img, params, deep_dir)
        if fused.size != ir_img.size:
            raise RuntimeError(f"Size drift for {name}: {fused.size} != {ir_img.size}")
        save_image_like(fused, out_dir / name)
    (out_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(names)


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
    parser.add_argument("--deep-dir", default="remote_sync/results/deep_crossfuse_raw")
    parser.add_argument("--out-root", default="results/candidates_round2")
    parser.add_argument("--score-out", default="results/round2_local_scores.csv")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    deep_dir = Path(args.deep_dir)
    deep_arg = deep_dir if deep_dir.exists() else None
    out_root = ensure_dir(args.out_root)

    rows: list[dict[str, float | str]] = []
    variants = candidate_params()
    for idx, (name, params) in enumerate(variants, start=1):
        cand_dir = out_root / f"{idx:02d}_{name}"
        print(f"[{idx}/{len(variants)}] {cand_dir}")
        count = generate_candidate(ir_dir, vi_dir, cand_dir, params, deep_arg)
        _, summary = evaluate_dir(ir_dir, vi_dir, cand_dir)
        (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row: dict[str, float | str] = {
            "candidate": cand_dir.name,
            "path": str(cand_dir),
            "family": str(params["family"]),
            "count": count,
        }
        row.update(summary)
        rows.append(row)

    add_local_rank_scores(rows)
    rows.sort(key=lambda row: float(row["LocalPseudoRank"]))
    fieldnames = (
        ["candidate", "path", "family", "count", "LocalPseudoRank"]
        + list(HIGHER_IS_BETTER)
        + [f"R_local_{metric}" for metric in HIGHER_IS_BETTER]
    )
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
