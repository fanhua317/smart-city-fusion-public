from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from fusion_common import (
    ensure_dir,
    IMAGE_EXTS,
    gradient_mag,
    gray_float,
    list_pairs,
    load_image,
    normalize01,
    percentile_stretch,
    recombine_y_with_vi_color,
    robust_match,
    save_image_like,
)
from generate_round7_leader_target import cap_gradient_to_sources, soft_std_lift, source_gate
from inspect_dataset import inspect_dataset, write_profile
from local_metrics import evaluate_dir, write_csv


BASE_SOURCES = {
    "r03": [
        "results/candidates_round10/03_rankpoints_detail_rank",
        "results/candidates_round11/03_rankpoints_detail_rank",
    ],
    "r33": ["results/candidates_round2/33_mix_avg21_45"],
}


def read_profile(path: str | Path, ir_dir: str | Path, vi_dir: str | Path) -> dict[str, set[str]]:
    path = Path(path)
    if not path.exists():
        rows = inspect_dataset(ir_dir, vi_dir)
        write_profile(rows, path)
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image"]] = {tag for tag in row["group"].split(";") if tag}
    return out


def first_complete_source(key: str, names: list[str]) -> Path | None:
    for raw in BASE_SOURCES[key]:
        path = Path(raw)
        if path.exists() and all((path / name).exists() for name in names):
            return path
    return None


def source_gray(source_dir: Path | None, name: str, fallback: np.ndarray) -> np.ndarray:
    if source_dir is None:
        return fallback
    path = source_dir / name
    if not path.exists():
        return fallback
    return gray_float(load_image(path))


def detail_pick(ir: np.ndarray, vi: np.ndarray, gate: np.ndarray, sigma: float, mode: str, ir_bias: float) -> np.ndarray:
    ir_low = ndimage.gaussian_filter(ir, sigma=sigma, mode="reflect")
    vi_low = ndimage.gaussian_filter(vi, sigma=sigma, mode="reflect")
    ir_detail = ir - ir_low
    vi_detail = vi - vi_low
    g_ir = normalize01(gradient_mag(ir))
    g_vi = normalize01(gradient_mag(vi))
    if mode == "max":
        use_ir = g_ir * ir_bias > g_vi
        return np.where(use_ir, gate * ir_detail, vi_detail).astype(np.float32)
    if mode == "vi_gate":
        return (vi_detail + ir_bias * gate * ir_detail).astype(np.float32)
    if mode == "ir_gate":
        return (0.58 * vi_detail + ir_bias * gate * ir_detail).astype(np.float32)
    return (0.78 * vi_detail + 0.22 * gate * ir_detail).astype(np.float32)


def prep_vi_for_night(vi: np.ndarray, gamma: float, stretch: float) -> np.ndarray:
    low = max(0.05, 1.0 - stretch)
    high = min(99.95, 99.0 + stretch)
    out = percentile_stretch(vi, low, high)
    if abs(gamma - 1.0) > 1e-4:
        out = np.power(np.clip(out, 0.0, 1.0), gamma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fuse_adaptive_pair(
    name: str,
    ir_dir: Path,
    vi_dir: Path,
    r03_dir: Path | None,
    r33_dir: Path | None,
    tags: set[str],
    spec: dict[str, Any],
) -> Image.Image:
    ir_img = load_image(ir_dir / name)
    vi_img = load_image(vi_dir / name)
    vi0 = gray_float(vi_img)
    ir = robust_match(gray_float(ir_img), vi0)
    r03 = source_gray(r03_dir, name, vi0)
    r33 = source_gray(r33_dir, name, vi0)

    active = spec["target_group"] in tags
    params = spec if active else spec["fallback"]
    vi = vi0
    if params.get("night_preprocess"):
        vi = prep_vi_for_night(vi0, float(params["vi_gamma"]), float(params["vi_stretch"]))

    low_sigma = float(params["low_sigma"])
    ir_low = ndimage.gaussian_filter(ir, sigma=low_sigma, mode="reflect")
    vi_low = ndimage.gaussian_filter(vi, sigma=low_sigma, mode="reflect")
    r03_low = ndimage.gaussian_filter(r03, sigma=low_sigma, mode="reflect")
    r33_low = ndimage.gaussian_filter(r33, sigma=low_sigma, mode="reflect")
    gate = source_gate(ir, vi, float(params["edge_floor"]), float(params["thermal_weight"]), float(params["gate_sigma"]))
    low = (
        float(params["w_ir"]) * ir_low
        + float(params["w_vi"]) * vi_low
        + float(params["w_r03"]) * r03_low
        + float(params["w_r33"]) * r33_low
    )
    low /= max(1e-6, float(params["w_ir"]) + float(params["w_vi"]) + float(params["w_r03"]) + float(params["w_r33"]))

    detail = detail_pick(
        ir,
        vi,
        gate,
        sigma=float(params["detail_sigma"]),
        mode=str(params["detail_mode"]),
        ir_bias=float(params["detail_ir_bias"]),
    )
    fused = low + float(params["detail_gain"]) * detail
    if float(params.get("local_contrast", 0.0)):
        fused += float(params["local_contrast"]) * (fused - ndimage.gaussian_filter(fused, sigma=5.0, mode="reflect"))

    fused = soft_std_lift(fused, vi0, float(params["target_std"]) / 255.0, float(params["std_amount"]))
    fused = cap_gradient_to_sources(
        fused,
        low,
        ir,
        vi0,
        cap=float(params["grad_cap"]),
        amount=float(params["cap_amount"]),
    )
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

    if params.get("gray_output") or vi_img.mode == "L":
        return Image.fromarray(np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    return recombine_y_with_vi_color(vi_img, fused, saturation=float(params["saturation"]))


def fallback_params() -> dict[str, Any]:
    return {
        "low_sigma": 4.2,
        "detail_sigma": 1.25,
        "edge_floor": 0.88,
        "thermal_weight": 0.12,
        "gate_sigma": 1.1,
        "w_ir": 0.10,
        "w_vi": 0.28,
        "w_r03": 0.36,
        "w_r33": 0.26,
        "detail_mode": "blend",
        "detail_ir_bias": 0.32,
        "detail_gain": 0.72,
        "target_std": 45.0,
        "std_amount": 0.48,
        "grad_cap": 1.075,
        "cap_amount": 0.58,
        "local_contrast": 0.02,
        "saturation": 0.92,
        "gray_output": False,
    }


def candidate_specs() -> list[tuple[str, dict[str, Any]]]:
    fallback = fallback_params()
    specs: list[tuple[str, dict[str, Any]]] = []

    for idx, (base_ir, detail_gain, target_std, cap) in enumerate(product([0.50, 0.56, 0.61, 0.66], [0.76, 0.88], [45.0, 47.5], [1.055]), 1):
        spec = {
            "target_group": "FLIR",
            "preset": "flir",
            "fallback": fallback,
            "low_sigma": 4.4,
            "detail_sigma": 1.22,
            "edge_floor": 0.78,
            "thermal_weight": 0.20,
            "gate_sigma": 1.05,
            "w_ir": base_ir,
            "w_vi": 0.26,
            "w_r03": 0.20,
            "w_r33": 0.12,
            "detail_mode": "max",
            "detail_ir_bias": 1.12,
            "detail_gain": detail_gain,
            "target_std": target_std,
            "std_amount": 0.58,
            "grad_cap": cap,
            "cap_amount": 0.58,
            "local_contrast": 0.035,
            "saturation": 0.88,
            "gray_output": False,
        }
        specs.append((f"A_FLIR_flir_ir{int(base_ir*100):02d}_std{int(target_std*10):03d}_d{int(detail_gain*100):02d}_{idx:02d}", spec))

    for idx, (w_vi, ir_gate, target_std, cap_amount) in enumerate(product([0.46, 0.54, 0.62, 0.68], [0.18, 0.26], [43.8, 45.0], [0.60]), 1):
        spec = {
            "target_group": "DN_D",
            "preset": "day_vi_structure",
            "fallback": fallback,
            "low_sigma": 4.8,
            "detail_sigma": 1.35,
            "edge_floor": 0.92,
            "thermal_weight": 0.07,
            "gate_sigma": 1.35,
            "w_ir": 0.06,
            "w_vi": w_vi,
            "w_r03": 0.20,
            "w_r33": 0.28,
            "detail_mode": "vi_gate",
            "detail_ir_bias": ir_gate,
            "detail_gain": 0.66,
            "target_std": target_std,
            "std_amount": 0.42,
            "grad_cap": 1.055,
            "cap_amount": cap_amount,
            "local_contrast": 0.015,
            "saturation": 0.90,
            "gray_output": False,
        }
        specs.append((f"B_D_day_vi{int(w_vi*100):02d}_gate{int(ir_gate*100):02d}_std{int(target_std*10):03d}_{idx:02d}", spec))

    for idx, (gamma, stretch, ir_w, target_std) in enumerate(product([0.86, 0.92], [1.4, 2.2], [0.26, 0.36, 0.46, 0.54], [46.0]), 1):
        spec = {
            "target_group": "DN_N",
            "preset": "night_ir_saliency",
            "fallback": fallback,
            "night_preprocess": True,
            "vi_gamma": gamma,
            "vi_stretch": stretch,
            "low_sigma": 4.0,
            "detail_sigma": 1.15,
            "edge_floor": 0.78,
            "thermal_weight": 0.22,
            "gate_sigma": 1.05,
            "w_ir": ir_w,
            "w_vi": 0.32,
            "w_r03": 0.26,
            "w_r33": 0.16,
            "detail_mode": "ir_gate",
            "detail_ir_bias": 0.78,
            "detail_gain": 0.82,
            "target_std": target_std,
            "std_amount": 0.58,
            "grad_cap": 1.070,
            "cap_amount": 0.62,
            "local_contrast": 0.04,
            "saturation": 0.92,
            "gray_output": False,
        }
        specs.append((f"C_N_night_g{int(gamma*100):02d}_str{int(stretch*10):02d}_ir{int(ir_w*100):02d}_{idx:02d}", spec))

    for idx, (w_vi, detail_gain, target_std, grad_cap) in enumerate(product([0.40, 0.50, 0.60, 0.68], [0.60, 0.76], [43.8, 45.3], [1.045]), 1):
        spec = {
            "target_group": "GRAY_VI",
            "preset": "gray_vi_struct",
            "fallback": fallback,
            "low_sigma": 4.8,
            "detail_sigma": 1.35,
            "edge_floor": 0.90,
            "thermal_weight": 0.10,
            "gate_sigma": 1.30,
            "w_ir": 0.08,
            "w_vi": w_vi,
            "w_r03": 0.24,
            "w_r33": 0.28,
            "detail_mode": "vi_gate",
            "detail_ir_bias": 0.22,
            "detail_gain": detail_gain,
            "target_std": target_std,
            "std_amount": 0.44,
            "grad_cap": grad_cap,
            "cap_amount": 0.64,
            "local_contrast": 0.01,
            "saturation": 0.0,
            "gray_output": True,
        }
        specs.append((f"D_GRAY_gray_vi{int(w_vi*100):02d}_std{int(target_std*10):03d}_d{int(detail_gain*100):02d}_{idx:02d}", spec))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="target/ir")
    parser.add_argument("--vi-dir", default="target/vi")
    parser.add_argument("--profile", default="results/dataset_profile.csv")
    parser.add_argument("--out-root", default="results/candidates_round11_adaptive")
    parser.add_argument("--score-out", default="results/round11_adaptive_scores.csv")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    names = list_pairs(ir_dir, vi_dir)
    profile = read_profile(args.profile, ir_dir, vi_dir)
    r03_dir = first_complete_source("r03", names)
    r33_dir = first_complete_source("r33", names)
    out_root = ensure_dir(args.out_root)
    rows: list[dict[str, Any]] = []

    for idx, (name, spec) in enumerate(candidate_specs(), 1):
        cand_name = f"{idx:02d}_{name}"
        cand_dir = ensure_dir(out_root / cand_name)
        print(cand_dir)
        summary_path = cand_dir / "metrics_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            image_count = sum(1 for child in cand_dir.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTS)
            if image_count != len(names):
                for image_name in names:
                    fused = fuse_adaptive_pair(image_name, ir_dir, vi_dir, r03_dir, r33_dir, profile[image_name], spec)
                    save_image_like(fused, cand_dir / image_name)
            (cand_dir / "params.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
            metric_rows, summary = evaluate_dir(ir_dir, vi_dir, cand_dir)
            write_csv(cand_dir / "metrics.csv", metric_rows)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        row: dict[str, Any] = {
            "candidate": cand_name,
            "path": str(cand_dir),
            "target_group": spec["target_group"],
            "preset": spec["preset"],
        }
        row.update(summary)
        rows.append(row)

    fieldnames = ["candidate", "path", "target_group", "preset"] + list(next(iter(rows)).keys() - {"candidate", "path", "target_group", "preset"})
    fieldnames = ["candidate", "path", "target_group", "preset", "AG", "CC", "EN", "MI", "MSE", "Nabf", "PSNR", "Qabf", "SCD", "SD", "SF", "SSIM", "VIF"]
    with Path(args.score_out).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated={len(rows)} out={args.out_root} scores={args.score_out}")


if __name__ == "__main__":
    main()
