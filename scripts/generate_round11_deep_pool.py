from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from fusion_common import (
    ensure_dir,
    gray_float,
    list_pairs,
    load_image,
    recombine_y_with_vi_color,
    save_image_like,
)
from generate_round7_leader_target import cap_gradient_to_sources, soft_std_lift
from inspect_dataset import inspect_dataset, write_profile
from local_metrics import evaluate_dir, write_csv


DEEP_DIRS = {
    "crossfuse": "results/candidates_remote/crossfuse_target",
    "cddfuse": "results/candidates_remote/cddfuse_target",
    "seafusion": "results/candidates_remote/seafusion_target",
    "densefuse": "results/candidates_remote/densefuse_target",
    "u2fusion": "results/candidates_remote/u2fusion_target",
    "tardal": "results/candidates_remote/tardal_target",
    "swinfusion": "results/candidates_remote/swinfusion_target",
}

ANCHORS = {
    "r03": [
        "results/candidates_round10/03_rankpoints_detail_rank",
        "results/candidates_round11/03_rankpoints_detail_rank",
    ],
    "r33": ["results/candidates_round2/33_mix_avg21_45"],
}


def complete_dir(path: Path, ir_dir: Path, vi_dir: Path, names: list[str]) -> bool:
    if not path.exists():
        return False
    for name in names:
        f_path = path / name
        if not f_path.exists():
            return False
        try:
            src = load_image(ir_dir / name)
            fused = load_image(f_path)
            vi = load_image(vi_dir / name)
        except Exception:  # noqa: BLE001 - directory is simply not eligible.
            return False
        if fused.size != src.size or vi.size != src.size:
            return False
    return True


def first_anchor(key: str, names: list[str]) -> Path | None:
    for raw in ANCHORS[key]:
        path = Path(raw)
        if path.exists() and all((path / name).exists() for name in names):
            return path
    return None


def gray_from(path: Path | None, name: str, fallback: np.ndarray) -> np.ndarray:
    if path is None or not (path / name).exists():
        return fallback
    return gray_float(load_image(path / name))


def variants() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("low03_highdeep_s45_cap", {"low_r03": 0.62, "low_r33": 0.18, "low_deep": 0.10, "low_vi": 0.10, "high_gain": 0.42, "target_std": 45.0, "std_amount": 0.42, "grad_cap": 1.050, "cap_amount": 0.68, "sat": 0.90}),
        ("low03_highdeep_s46", {"low_r03": 0.55, "low_r33": 0.22, "low_deep": 0.12, "low_vi": 0.11, "high_gain": 0.50, "target_std": 46.0, "std_amount": 0.48, "grad_cap": 1.060, "cap_amount": 0.62, "sat": 0.90}),
        ("low33_highdeep_s45", {"low_r03": 0.22, "low_r33": 0.56, "low_deep": 0.10, "low_vi": 0.12, "high_gain": 0.44, "target_std": 45.0, "std_amount": 0.40, "grad_cap": 1.045, "cap_amount": 0.70, "sat": 0.88}),
        ("low33_highdeep_s47", {"low_r03": 0.25, "low_r33": 0.48, "low_deep": 0.15, "low_vi": 0.12, "high_gain": 0.56, "target_std": 47.0, "std_amount": 0.54, "grad_cap": 1.070, "cap_amount": 0.60, "sat": 0.90}),
        ("mix_anchor_highdeep_s48", {"low_r03": 0.38, "low_r33": 0.34, "low_deep": 0.16, "low_vi": 0.12, "high_gain": 0.62, "target_std": 48.0, "std_amount": 0.58, "grad_cap": 1.080, "cap_amount": 0.56, "sat": 0.92}),
        ("deep_low_softcap_s46", {"low_r03": 0.30, "low_r33": 0.24, "low_deep": 0.34, "low_vi": 0.12, "high_gain": 0.36, "target_std": 46.0, "std_amount": 0.45, "grad_cap": 1.045, "cap_amount": 0.72, "sat": 0.90}),
        ("deep_low_detail_s47", {"low_r03": 0.28, "low_r33": 0.20, "low_deep": 0.40, "low_vi": 0.12, "high_gain": 0.46, "target_std": 47.0, "std_amount": 0.54, "grad_cap": 1.060, "cap_amount": 0.64, "sat": 0.92}),
        ("std45_gradient_cap", {"low_r03": 0.46, "low_r33": 0.28, "low_deep": 0.14, "low_vi": 0.12, "high_gain": 0.54, "target_std": 45.0, "std_amount": 0.50, "grad_cap": 1.035, "cap_amount": 0.78, "sat": 0.88}),
        ("std48_detail_probe", {"low_r03": 0.36, "low_r33": 0.28, "low_deep": 0.22, "low_vi": 0.14, "high_gain": 0.70, "target_std": 48.0, "std_amount": 0.62, "grad_cap": 1.095, "cap_amount": 0.52, "sat": 0.92}),
        ("anchor_balanced_s465", {"low_r03": 0.44, "low_r33": 0.34, "low_deep": 0.10, "low_vi": 0.12, "high_gain": 0.48, "target_std": 46.5, "std_amount": 0.50, "grad_cap": 1.055, "cap_amount": 0.66, "sat": 0.90}),
    ]


def fuse_deep_pair(
    name: str,
    ir_dir: Path,
    vi_dir: Path,
    deep_dir: Path,
    r03_dir: Path | None,
    r33_dir: Path | None,
    params: dict[str, Any],
) -> Image.Image:
    ir_img = load_image(ir_dir / name)
    vi_img = load_image(vi_dir / name)
    ir = gray_float(ir_img)
    vi = gray_float(vi_img)
    deep = gray_float(load_image(deep_dir / name))
    r03 = gray_from(r03_dir, name, vi)
    r33 = gray_from(r33_dir, name, vi)
    low_sigma = 4.0
    lows = {
        "r03": ndimage.gaussian_filter(r03, sigma=low_sigma, mode="reflect"),
        "r33": ndimage.gaussian_filter(r33, sigma=low_sigma, mode="reflect"),
        "deep": ndimage.gaussian_filter(deep, sigma=low_sigma, mode="reflect"),
        "vi": ndimage.gaussian_filter(vi, sigma=low_sigma, mode="reflect"),
    }
    total = float(params["low_r03"]) + float(params["low_r33"]) + float(params["low_deep"]) + float(params["low_vi"])
    low = (
        float(params["low_r03"]) * lows["r03"]
        + float(params["low_r33"]) * lows["r33"]
        + float(params["low_deep"]) * lows["deep"]
        + float(params["low_vi"]) * lows["vi"]
    ) / max(total, 1e-6)
    deep_high = deep - ndimage.gaussian_filter(deep, sigma=1.25, mode="reflect")
    vi_high = vi - ndimage.gaussian_filter(vi, sigma=1.25, mode="reflect")
    fused = low + float(params["high_gain"]) * deep_high + 0.12 * vi_high
    fused = soft_std_lift(fused, vi, float(params["target_std"]) / 255.0, float(params["std_amount"]))
    fused = cap_gradient_to_sources(fused, low, ir, vi, float(params["grad_cap"]), float(params["cap_amount"]))
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    if vi_img.mode == "L":
        return Image.fromarray(np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["sat"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--profile", default="results/dataset_profile.csv")
    parser.add_argument("--out-root", default="results/candidates_round11_deep")
    parser.add_argument("--score-out", default="results/round11_deep_pool_scores.csv")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    names = list_pairs(ir_dir, vi_dir)
    profile = Path(args.profile)
    if not profile.exists():
        write_profile(inspect_dataset(ir_dir, vi_dir), profile)
    r03_dir = first_anchor("r03", names)
    r33_dir = first_anchor("r33", names)
    out_root = ensure_dir(args.out_root)
    rows: list[dict[str, Any]] = []

    available = {model: Path(raw) for model, raw in DEEP_DIRS.items() if complete_dir(Path(raw), ir_dir, vi_dir, names)}
    print(f"available_deep={sorted(available)}")
    for model, deep_dir in available.items():
        for idx, (variant_name, params) in enumerate(variants(), 1):
            cand_name = f"{model}_{idx:02d}_{variant_name}"
            cand_dir = ensure_dir(out_root / cand_name)
            print(cand_dir)
            for name in names:
                fused = fuse_deep_pair(name, ir_dir, vi_dir, deep_dir, r03_dir, r33_dir, params)
                save_image_like(fused, cand_dir / name)
            manifest = {"model": model, "source": str(deep_dir), "variant": variant_name, "params": params}
            (cand_dir / "params.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            metric_rows, summary = evaluate_dir(ir_dir, vi_dir, cand_dir)
            write_csv(cand_dir / "metrics.csv", metric_rows)
            (cand_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            row: dict[str, Any] = {"candidate": cand_name, "path": str(cand_dir), "model": model, "variant": variant_name}
            row.update(summary)
            rows.append(row)

    fieldnames = ["candidate", "path", "model", "variant", "AG", "CC", "EN", "MI", "MSE", "Nabf", "PSNR", "Qabf", "SCD", "SD", "SF", "SSIM", "VIF"]
    ensure_dir(Path(args.score_out).parent)
    with Path(args.score_out).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated={len(rows)} scores={args.score_out}")


if __name__ == "__main__":
    main()
